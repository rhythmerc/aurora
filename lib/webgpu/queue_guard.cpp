#include "queue_guard.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>

#include <dawn/native/VulkanBackend.h>
#include <webgpu/webgpu.h>

namespace aurora::webgpu {
namespace {

std::atomic_bool g_shared{false};
WGPUDevice g_device = nullptr;
// The queue itself: the XR thread's submits and OpenXR calls, and Dawn's
// submits (QueueGuard).
std::recursive_mutex g_queueMutex;
// Dawn's side: held through every wrapped Dawn call (DawnGuard), so nothing
// else reaches Dawn's queue while a submit's command buffers are translated
// with Dawn's flush held. The XR thread never takes it.
std::recursive_mutex g_dawnMutex;
std::atomic<int64_t> g_waitMaxNs{0};
std::atomic<uint32_t> g_waitsOver1ms{0};
// The longest hold and its holder: written with the queue lock held.
int64_t g_holdMaxNs = 0;
const char* g_holdMaxSite = "";

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

} // namespace

void set_queue_shared(bool shared, WGPUDevice device) noexcept {
  g_device = device;
  g_shared.store(shared, std::memory_order_release);
}
bool queue_shared() noexcept { return g_shared.load(std::memory_order_acquire); }

QueueGuard::QueueGuard(bool timed, const char* site) noexcept : m_site(site) {
  if (!queue_shared())
    return;
  if (!timed) {
    g_queueMutex.lock();
  } else if (!g_queueMutex.try_lock()) {
    const int64_t start = now_ns();
    g_queueMutex.lock();
    const int64_t ns = now_ns() - start;
    int64_t prev = g_waitMaxNs.load(std::memory_order_relaxed);
    while (ns > prev && !g_waitMaxNs.compare_exchange_weak(prev, ns, std::memory_order_relaxed)) {}
    if (ns > 1000000)
      g_waitsOver1ms.fetch_add(1, std::memory_order_relaxed);
  }
  m_locked = true;
  m_since = now_ns();
}

QueueGuard::~QueueGuard() {
  if (!m_locked)
    return;
  if (const int64_t held = now_ns() - m_since; held > g_holdMaxNs) {
    g_holdMaxNs = held;
    g_holdMaxSite = m_site;
  }
  g_queueMutex.unlock();
}

QueueGuardStats queue_guard_stats() noexcept {
  QueueGuardStats s;
  s.waitMaxNs = g_waitMaxNs.exchange(0, std::memory_order_relaxed);
  s.waitsOver1ms = g_waitsOver1ms.exchange(0, std::memory_order_relaxed);
  std::lock_guard lock{g_queueMutex};
  s.holdMaxNs = g_holdMaxNs;
  s.holdMaxSite = g_holdMaxSite;
  g_holdMaxNs = 0;
  g_holdMaxSite = "";
  return s;
}

} // namespace aurora::webgpu

#ifdef AURORA_QUEUE_GUARD_WRAP
// The Dawn entry points that can reach the Vulkan queue, wrapped at link time
// (ld --wrap, cmake/aurora_xr.cmake): every caller's reference to wgpuX
// resolves to __wrap_wgpuX, which holds the locks around Dawn's (__real_).
namespace {

using aurora::webgpu::g_dawnMutex;
using aurora::webgpu::queue_shared;
using aurora::webgpu::QueueGuard;

class DawnGuard {
public:
  DawnGuard() noexcept : m_locked(queue_shared()) {
    if (m_locked)
      g_dawnMutex.lock();
  }
  ~DawnGuard() {
    if (m_locked)
      g_dawnMutex.unlock();
  }
  DawnGuard(const DawnGuard&) = delete;
  DawnGuard& operator=(const DawnGuard&) = delete;

private:
  bool m_locked;
};

} // namespace

#define AURORA_WRAP_VOID(name, params, args)                                                                           \
  extern "C" void __real_##name params;                                                                                \
  extern "C" void __wrap_##name params {                                                                               \
    const DawnGuard dawn;                                                                                              \
    const QueueGuard queue{false, #name};                                                                              \
    __real_##name args;                                                                                                \
  }
#define AURORA_WRAP(ret, name, params, args)                                                                           \
  extern "C" ret __real_##name params;                                                                                 \
  extern "C" ret __wrap_##name params {                                                                                \
    const DawnGuard dawn;                                                                                              \
    const QueueGuard queue{false, #name};                                                                              \
    return __real_##name args;                                                                                         \
  }

AURORA_WRAP_VOID(wgpuQueueWriteBuffer, (WGPUQueue q, WGPUBuffer b, uint64_t o, void const* d, size_t s),
                 (q, b, o, d, s))
AURORA_WRAP_VOID(wgpuQueueWriteTexture,
                 (WGPUQueue q, WGPUTexelCopyTextureInfo const* dst, void const* d, size_t s,
                  WGPUTexelCopyBufferLayout const* l, WGPUExtent3D const* e),
                 (q, dst, d, s, l, e))
AURORA_WRAP(WGPUFuture, wgpuQueueOnSubmittedWorkDone, (WGPUQueue q, WGPUQueueWorkDoneCallbackInfo i), (q, i))
AURORA_WRAP(WGPUFuture, wgpuBufferMapAsync,
            (WGPUBuffer b, WGPUMapMode m, size_t o, size_t s, WGPUBufferMapCallbackInfo i), (b, m, o, s, i))
AURORA_WRAP_VOID(wgpuBufferUnmap, (WGPUBuffer b), (b))
AURORA_WRAP_VOID(wgpuBufferDestroy, (WGPUBuffer b), (b))
AURORA_WRAP_VOID(wgpuInstanceProcessEvents, (WGPUInstance i), (i))
AURORA_WRAP_VOID(wgpuDeviceTick, (WGPUDevice d), (d))
AURORA_WRAP_VOID(wgpuDeviceDestroy, (WGPUDevice d), (d))
AURORA_WRAP_VOID(wgpuDeviceRelease, (WGPUDevice d), (d))
AURORA_WRAP(WGPUStatus, wgpuSurfacePresent, (WGPUSurface s), (s))
AURORA_WRAP_VOID(wgpuSurfaceGetCurrentTexture, (WGPUSurface s, WGPUSurfaceTexture* t), (s, t))
AURORA_WRAP_VOID(wgpuSurfaceConfigure, (WGPUSurface s, WGPUSurfaceConfiguration const* c), (s, c))
AURORA_WRAP_VOID(wgpuSurfaceUnconfigure, (WGPUSurface s), (s))
AURORA_WRAP(WGPUStatus, wgpuSharedTextureMemoryBeginAccess,
            (WGPUSharedTextureMemory m, WGPUTexture t, WGPUSharedTextureMemoryBeginAccessDescriptor const* d),
            (m, t, d))
AURORA_WRAP(WGPUStatus, wgpuSharedTextureMemoryEndAccess,
            (WGPUSharedTextureMemory m, WGPUTexture t, WGPUSharedTextureMemoryEndAccessState* d), (m, t, d))

// A submit translates its command buffers with Dawn's flush held, which
// doesn't touch the queue, and only the flush that hands them to Vulkan holds
// the queue: the XR thread waits for that alone, not for a frame's
// translation. The Dawn lock keeps every other wrapped call (any of which may
// flush or wait) out until the flush is released.
extern "C" void __real_wgpuQueueSubmit(WGPUQueue q, size_t n, WGPUCommandBuffer const* c);
extern "C" void __wrap_wgpuQueueSubmit(WGPUQueue q, size_t n, WGPUCommandBuffer const* c) {
  if (!queue_shared()) {
    __real_wgpuQueueSubmit(q, n, c);
    return;
  }
  const DawnGuard dawn;
  dawn::native::vulkan::HoldQueueFlush(aurora::webgpu::g_device, true);
  __real_wgpuQueueSubmit(q, n, c);
  const QueueGuard queue{false, "wgpuQueueSubmit"};
  dawn::native::vulkan::HoldQueueFlush(aurora::webgpu::g_device, false);
}

// A wait holds the locks only while it submits what it waits for (a zero
// timeout flushes the awaited serials); the wait itself then touches no
// queue, and other threads, the XR thread above all, may use it meanwhile.
extern "C" WGPUWaitStatus __real_wgpuInstanceWaitAny(WGPUInstance i, size_t n, WGPUFutureWaitInfo* f, uint64_t t);
extern "C" WGPUWaitStatus __wrap_wgpuInstanceWaitAny(WGPUInstance i, size_t n, WGPUFutureWaitInfo* f, uint64_t t) {
  if (!queue_shared())
    return __real_wgpuInstanceWaitAny(i, n, f, t);
  WGPUWaitStatus status;
  {
    const DawnGuard dawn;
    const QueueGuard queue{false, "wgpuInstanceWaitAny"};
    status = __real_wgpuInstanceWaitAny(i, n, f, 0);
  }
  if (t == 0 || status != WGPUWaitStatus_TimedOut ||
      std::any_of(f, f + n, [](const WGPUFutureWaitInfo& w) { return w.completed != 0; }))
    return status;
  return __real_wgpuInstanceWaitAny(i, n, f, t);
}
#endif
