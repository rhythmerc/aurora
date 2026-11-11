#pragma once

// One Vulkan queue shared by Dawn, the OpenXR runtime and the XR thread.
//
// The XR presenter gives the runtime (and its own submits) a queue of their
// own in Dawn's queue family when the family has a spare one (the Quest's
// Adreno driver does). Mesa's Turnip (the Steam Frame) and RADV have one
// queue for graphics, so there all three share Dawn's, and Vulkan requires
// every use of a queue to be externally synchronized. With sharing on
// (lib/xr, xr::device_created), this lock is held around:
//   - every Dawn entry point that may reach the queue: submits, writes and
//     buffer maps that can flush, event processing and waits, presents,
//     shared texture memory access, device destruction. On Linux these are
//     wrapped at link time (cmake/aurora_xr.cmake, queue_guard.cpp), so every
//     caller is covered: aurora, the ImGui and RmlUi backends, the game.
//   - the XR thread's own submits, and the OpenXR calls that may use the
//     queue (xrBeginFrame, xrEndFrame, swapchain acquire and release).
// Off (the queue isn't shared), the guard does nothing.
//
// Two locks: Dawn's side holds one through each wrapped call, so a submit can
// translate its command buffers with Dawn's flush held (HoldQueueFlush),
// touching no queue, while nothing else reaches Dawn; the queue lock is held
// only where the queue is used (the flush, the other wrapped calls, the XR
// thread's). The XR thread waits for the queue, never for a translation.
//
// Lock order: Dawn's side, then the queue, then Dawn's device lock, never
// the other way. Dawn calls its callbacks with its own lock released, but
// they can run with these held (inside a wrapped event processing call), so
// a callback must not wait for another thread that is waiting for them.

#include <cstdint>

#include <webgpu/webgpu.h>

namespace aurora::webgpu {

// Sharing on or off, for this device; set once, when the device is created,
// before any thread but the creating one uses it.
void set_queue_shared(bool shared, WGPUDevice device) noexcept;
bool queue_shared() noexcept;

class QueueGuard {
public:
  // timed: the wait for the lock is counted (queue_guard_stats), for the
  // XR thread, whose frames it can make late. site: who holds it, for
  // queue_guard_stats' longest hold.
  explicit QueueGuard(bool timed = false, const char* site = "XR thread") noexcept;
  ~QueueGuard();
  QueueGuard(const QueueGuard&) = delete;
  QueueGuard& operator=(const QueueGuard&) = delete;

private:
  bool m_locked = false;
  const char* m_site = nullptr;
  int64_t m_since = 0;
};

// Since the last call: the XR thread's longest wait for the queue and how
// many took over a millisecond, and the longest anyone held it, and where
// (a wrapped Dawn entry point's name, or the XR thread). Resets them.
struct QueueGuardStats {
  int64_t waitMaxNs = 0;
  uint32_t waitsOver1ms = 0;
  int64_t holdMaxNs = 0;
  const char* holdMaxSite = "";
};
QueueGuardStats queue_guard_stats() noexcept;

} // namespace aurora::webgpu
