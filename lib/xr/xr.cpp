#include "xr.hpp"

#include <aurora/xr.h>

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/recording.hpp"
#include "../gfx/xr_replay.hpp"
#include "../gx/gx.hpp"

#include <aurora/gfx.hpp>

#include <SDL3/SDL.h>

#ifdef __ANDROID__
#define VK_USE_PLATFORM_ANDROID_KHR
#define XR_USE_PLATFORM_ANDROID
#include <android/hardware_buffer.h>
#include <android/trace.h>
#include <dlfcn.h>
#include <jni.h>
#endif
#define XR_USE_GRAPHICS_API_VULKAN
#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
// After vulkan.h: the fork's single-device hooks (github.com/encounter/dawn
// plus the melee-xr patches; see docs/quest-xr.md).
#include <dawn/native/VulkanBackend.h>

#include <poll.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#ifdef __ANDROID__
#include "renderdoc_app.h"
#endif
#include <bit>
#include <cmath>
#include <limits>
#include <initializer_list>
#include <cstdio>
#include <cstdlib>
#include <complex>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <map>
#include <unordered_map>
#include <vector>

namespace aurora::xr {
namespace {
Module Log("aurora::xr");

// ---------------------------------------------------------------- platform
//
// One Vulkan device: the OpenXR runtime creates Dawn's VkInstance and
// VkDevice (XR_KHR_vulkan_enable2, through the Dawn fork's
// SetExternalVulkanHooks), and Dawn renders straight into the runtime's
// swapchain images. Dawn submits on queue 0 of its family; the XR thread and
// the runtime use queue 1. A finished image comes back from Dawn with a
// semaphore (SYNC_FD on Android, OPAQUE_FD on Linux) that the XR thread waits
// on, on its queue, before releasing the image.
#ifdef __ANDROID__
constexpr VkExternalSemaphoreHandleTypeFlagBits kSemHandle = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
constexpr VkSemaphoreImportFlags kSemImportFlags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT; // required for SYNC_FD
constexpr wgpu::SharedFenceType kDawnFenceType = wgpu::SharedFenceType::SyncFD;
using DawnFenceExportInfo = wgpu::SharedFenceSyncFDExportInfo;
#else
constexpr VkExternalSemaphoreHandleTypeFlagBits kSemHandle = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
constexpr VkSemaphoreImportFlags kSemImportFlags = 0;
constexpr wgpu::SharedFenceType kDawnFenceType = wgpu::SharedFenceType::VkSemaphoreOpaqueFD;
using DawnFenceExportInfo = wgpu::SharedFenceVkSemaphoreOpaqueFDExportInfo;
#endif

// ---------------------------------------------------------------- state
//
// Three image streams go from the render worker to the XR thread, each with
// its own swapchain:
//   Screen  the presented frame, on the virtual screen (menus, or any time
//           the game draws no fight)
//   Stereo  3D fights: both eyes side by side, submitted as a projection
//           layer with the head poses they were rendered for
//   Hud     the HUD of a 3D fight, on a quad above the arena
//
// A slot is one swapchain image. The XR thread acquires images ahead of the
// render worker, which draws into the oldest one through Dawn and hands it
// back; the XR thread then releases them in acquisition order, as OpenXR
// requires.

enum class SlotState {
  Released,  // the runtime's
  Acquired,  // acquired by the XR thread; xrWaitSwapchainImage still pending
  Free,      // acquired and waited: the render worker may take it
  Rendering, // the render worker is drawing into it
  Ready,     // drawn; the XR thread releases it next
};

struct PendingAccess {
  // Dawn's release of the image: semaphores to wait on and the layout it left.
  std::vector<int> fds;
  VkImageLayout oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageLayout newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
};

struct Slot {
  VkImage image = VK_NULL_HANDLE; // the runtime's swapchain image, or a staged stream's own (ring)
  VkDeviceMemory memory = VK_NULL_HANDLE; // ring images: theirs

  // Render worker only.
  wgpu::SharedTextureMemory stm;
  wgpu::Texture texture;

  // Guarded by g_mutex.
  SlotState state = SlotState::Released;
  uint64_t acquireSeq = 0;       // acquisition order
  PendingAccess fromDawn;        // set when Ready
  std::array<XrView, 2> views{}; // Stereo: the eye poses this image was rendered for
  std::array<int32_t, 2> rect{};  // Stereo, dynamic resolution: the rendered size (0: all)
  uint64_t tickFrame = 0;         // lock-step: the display frame of the tick that started its game frame
  uint64_t readySeen = 0;         // XR thread: the display frame it was first seen Ready (0: not yet)
  bool claimed = false;           // Acquired, and the render worker took it before it was waited (g_deferEyes)

  // XR thread only.
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  bool inFlight = false;
  std::vector<VkSemaphore> waitSems;
};

// kMotion and kMotionDepth: Application SpaceWarp's low-res motion vectors
// and depth for the 3D image (XR_FB_space_warp, when the runtime has it;
// AURORA_XR_APPSW=0 leaves them empty). Released with the 3D image of the same frame.
enum StreamId : int { kScreen = 0, kStereo = 1, kHud = 2, kMotion = 3, kMotionDepth = 4, kStreamCount = 5 };

struct Stream {
  const char* name = "";
  std::vector<Slot> slots;        // one per swapchain image; sized before Phase::SlotsReady
  uint32_t width = 0, height = 0; // set before the XR thread starts
  uint32_t layers = 1;            // Stereo with multiview: one per eye
  VkFormat format = VK_FORMAT_R8G8B8A8_UNORM; // what Dawn draws as (bytes already sRGB-encoded)
  int64_t swapFormat = 0;                     // swapchain: the sRGB twin, created mutable-format
  uint64_t acquireSeq = 0;        // g_mutex
  int renderingSlot = -1;         // render worker

  // XR thread only.
  XrSwapchain swapchain = XR_NULL_HANDLE;
  bool haveImage = false;
  std::chrono::steady_clock::time_point lastRelease{};
  uint64_t releases = 0;
  std::array<XrView, 2> shownViews{}; // Stereo: poses of the last released image
  uint64_t shownTick = 0;             // the game tick of the last released image
  std::array<int32_t, 2> shownRect{};  // Stereo: its rendered size (0: all)
  // Debug dump (AURORA_XR_DUMP)
  VkBuffer dumpBuf = VK_NULL_HANDLE;
  VkDeviceMemory dumpMem = VK_NULL_HANDLE;
  void* dumpPtr = nullptr;
  int dumpSlot = -1;
  bool depth = false;     // a depth swapchain (kMotionDepth)
  bool ownFormat = false; // keeps its own format (not the color swapchains' sRGB one)
  bool unwaited = false;  // render worker: its slot was taken before the XR thread waited it
  bool dumped = false;
  // Staged (create_ring): the render worker draws into images of ours, never
  // the runtime's, and the XR thread copies each into a swapchain image as
  // it hands it over (release_staged). Ring slots are Free, Rendering, Ready,
  // then Released while the copy runs.
  bool staged = false;
  std::vector<Slot> ring;
  int dumpRing = -1; // the ring slot whose copy also dumps (AURORA_XR_DUMP)
};

// The images the render worker draws into.
std::vector<Slot>& worker_slots(Stream& st) { return st.staged ? st.ring : st.slots; }

enum class Phase { Idle, Starting, SlotsReady, Imported, Failed };

std::mutex g_mutex;
std::condition_variable g_slotFreeCv; // a slot became Free (XR thread -> render worker)
std::atomic<Phase> g_phase{Phase::Idle};
std::atomic<bool> g_stop{false};
std::atomic<bool> g_sessionRunning{false};
std::thread g_thread;
std::array<Stream, kStreamCount> g_streams;

// Early eyes translated before their image is free (AURORA_XR_DEFER_EYES,
// on on Android): the XR thread acquires the next 3D image as soon as one
// may be, before it may be waited on; the render worker draws into it, Dawn
// translates the eyes with its flush held, and they reach the GPU once the
// XR thread has waited the image (after the last frame's release). At
// latency 4 the eyes' encoding and translation (about 2.5 ms) had come
// after the image, in the 16.7 ms the eyes have between two releases.
std::atomic<bool> g_deferEyes{false};

// Dynamic resolution (AURORA_XR_DYNRES=1, multiview only): the stereo
// swapchain is made at the largest scale, and each frame renders a corner of
// it at `scale` times the runtime's recommended eye size, reported to the
// compositor through imageRect. Render worker only, after setup.
struct DynamicResolution {
  bool on = false;
  bool random = false; // AURORA_XR_DYNRES_RANDOM: a random scale every frame (testing)
  float minScale = 0.8f, maxScale = 1.3f;
  float scale = 1.f;
  uint32_t recWidth = 0, recHeight = 0; // recommended eye size (times AURORA_XR_EYE_SCALE)
  // Controller (update_dynamic_resolution): a live cost table. The scale
  // moves in steps of kDynresStep; each step keeps the eye pass's GPU time
  // measured there (the part that scales with resolution), trusted less the
  // longer ago it was measured, and steps not measured lately are predicted
  // from the ones that were. The rest of the frame (flat passes, HUD,
  // compose) is one running estimate on top: a step's frame is that plus its
  // eye pass.
  //
  // The target is what a game frame leaves us on the GPU: its 16.7 ms less
  // the compositor's work over the display frames it spans (measured with
  // XR_META_performance_metrics, 2.3 ms each otherwise; about 4.6 ms a game
  // frame at 120 Hz), less headroom. The frame times the controller reads
  // include the compositor's preemptions of our passes. Mute City with four
  // fighters at 120 Hz: 10.2 ms frames held every 2-display-frame slot,
  // 12.1 ms frames missed some 12 a second (2026-10-10).
  //
  // Since 2026-10-10 the target comes from the deadline instead: each 3D
  // frame's margin between its GPU work being done (less any wait for a
  // swapchain image) and the release slot of the ideal latency (a game frame
  // plus one display frame after its tick; the pacing probe measures it). The
  // frame may grow by what the low end of recent margins (5th percentile)
  // has beyond AURORA_XR_DYNRES_MARGIN_MS (1), and must shrink by what it
  // lacks. Steps up wait for fresh margins at the new step. Below 1.0 looked
  // too soft to be worth it (AURORA_XR_DYNRES_MIN, default 1.0): a stage that
  // can't make the ideal latency at 1.0 stays there, and the fixed latency
  // goes one display frame later instead.
  double targetNs = 13.0e6; // per game frame, compositor preemptions included
  bool fixedTarget = false; // AURORA_XR_DYNRES_TARGET_MS pins it
  double headroomNs = 1.2e6; // AURORA_XR_DYNRES_HEADROOM_MS
  double bandNs = 0.5e6;    // hysteresis: down only past target + band, up to the target
  double emaNs = 0;         // frame GPU time at the current step, this visit
  double spanNs = 0;        // frame GPU time, running across steps
  double wantMarginNs = 3.0e6;             // AURORA_XR_DYNRES_MARGIN_MS
  std::array<float, 120> margins{};       // ns before the ideal slot, this step
  size_t marginCount = 0, marginNext = 0;
  double marginNs = 0;                    // their 5th percentile (with 30 or more)
  bool haveMargin = false;
  double restNs = 0;        // frame GPU time outside the eye pass, running
  int level = 0;            // the current step: scale = minScale + level * kDynresStep
  uint64_t seenSeq = 0;     // R.timing.frameSeq last read
  int settle = 0;           // frames measured since the last scale change
  int sinceStep = 0;        // frames since the last step up
  int spikeRun = 0;         // readings in a row taken for spikes
  uint32_t pipelinesSeen = 0; // gfx::pipelines_created() last frame
  int compileQuiet = 0;       // frames since a pipeline was last created
  std::array<uint64_t, 4> recentMisses{};
  // The brake: frames done late (g_lateFrames), by frameCount; too many and
  // the scale goes down a step and stays at or under it for a while.
  std::array<uint64_t, 4> recentLate{};
  uint32_t recentLateIndex = 0, lateFramesSeen = 0;
  int ceiling = INT32_MAX;          // highest step allowed until ceilingUntil
  std::chrono::steady_clock::time_point ceilingUntil{};
  int brakes = 0, brakesSinceLog = 0; // in a row (for the hold's length); logged
  // A heavy moment (a KO's explosion took the eye pass from 4.5 to 15 ms for
  // half a second): the step it came from, while one lasts (-1: none).
  int transientFrom = -1;
  int transientTo = -1;     // a step the readings asked for, for the next decision
  int overRun = 0, fitRun = 0; // readings in a row over the target / fitting the old step
  double transientBaseNs = 0; // the frame GPU time before it
  std::chrono::steady_clock::time_point transientSince{}, brakeQuietUntil{};
  uint32_t transients = 0;
  // A heavy effect the game announced (aurora_xr_heavy_effect): the share of
  // pixels its drop keeps, learned from the last one's peak eye cost.
  double hintPixels = 0.5;
  bool transientHinted = false;
  double transientPeakEyeNs = 0; // the moment's worst eye pass, at the step it came from
  uint32_t recentIndex = 0;
  uint64_t frameCount = 0;
  std::chrono::steady_clock::time_point lastFrame{};
  // Logged every 10 s.
  double scaleSum = 0, nsSum = 0;
  uint32_t frames = 0, nsFrames = 0, panics = 0, loneMisses = 0, lateImages = 0;
  uint32_t readings = 0, skipSettle = 0, skipCompile = 0, skipSpike = 0; // GPU timings, and why some were left out
  uint32_t lateSeen = 0;
  float lowest = 99.f, highest = 0.f;
  std::chrono::steady_clock::time_point lastLog{};
} g_dynres;

// One step of the cost table: the frame GPU time measured at that scale and
// how much of it there is, which fades with time since the last visit.
constexpr float kDynresStep = 0.025f;
constexpr int kDynresLevels = 64;
struct DynresCost {
  double ns = 0;          // running average (0: never measured)
  double samples = 0;     // weight behind it, decaying
  uint64_t lastFrame = 0; // frameCount of the last sample
};
// Per stage and mode (mixed reality or VR): costs differ a lot between them.
// Kept for the session. Render worker only.
std::unordered_map<int, std::array<DynresCost, kDynresLevels>> g_dynresCosts;
// The game framebuffer's format and sample count, read by begin_frame before
// the XR thread starts: when they allow it, both eyes render straight into
// the 3D swapchain image, which then has to be in the framebuffer's format.
wgpu::TextureFormat g_sceneFormat = wgpu::TextureFormat::RGBA8Unorm;
uint32_t g_sceneSamples = 1;
// Both eyes in one pass with a view mask over a 2-layer 3D image (Dawn fork
// multiview), decided by begin_frame before the XR thread starts. Otherwise
// the eyes are replayed side by side, one after the other.
bool g_multiview = false;

// Latest predicted eye poses, from the XR thread for the render worker.
std::mutex g_viewMutex;
// The flat screen as audio sees it (aurora_xr_audio_screen): shown or not,
// and where. Under g_viewMutex; written by the XR thread each frame.
struct AudioScreen {
  bool shown = false;
  XrPosef pose{};
  float width = 0.f;
} g_audioScreen;
std::array<XrView, 2> g_latestViews{};
bool g_viewsValid = false;
// How fast the head turns (degrees a second), from the poses publish_views
// locates: the most and the mean since the GPU passes line last logged them
// (under g_viewMutex), and the latest (any thread), for a spike's line.
float g_headTurnMax = 0.f;
double g_headTurnSum = 0.0;
uint32_t g_headTurnN = 0;
XrQuaternionf g_headTurnPrev{0.f, 0.f, 0.f, 1.f};
XrTime g_headTurnPrevTime = 0;
std::atomic<float> g_headTurnLast{0.f};

// Lock-step pacing (aurora_xr_pace): the XR thread ticks every
// g_displayPerGameFrame display frames, the game waits for the tick instead
// of its own 60 Hz timer, so every game frame is shown for the same number
// of display frames. 0 = the display rate is not a multiple of 60; no pacing.
std::mutex g_paceMutex;
std::condition_variable g_paceCv;
uint64_t g_paceTick = 0;
std::atomic<int> g_displayPerGameFrame{0};
std::atomic<float> g_displayHz{0.f}; // the display's refresh rate (0: unknown)
std::atomic<float> g_refreshHz{0.f}; // the display's own, also with SpaceWarp's loop at half of it
// The compositor's GPU time per display frame (ms, XR_META_performance_metrics;
// negative: unknown), polled by the XR thread for dynamic resolution.
std::atomic<float> g_compositorGpuMs{-1.f};
std::atomic<uint64_t> g_lastTickFrame{0}; // the display frame of the newest tick
// 3D images the headset held past their display frames during a fight (XR
// thread): a missed frame as the user sees it. Dynamic resolution reads it.
std::atomic<uint32_t> g_stereoLate{0};

// Where the arena sits in the room (meters, starting head space): A =
// T(pos) · R_y(yaw) · S(scale) takes game units to the room. The XR thread
// moves it while the player drags the arena during a pause; the render
// worker reads it for every 3D frame.
struct ArenaPose {
  XrVector3f pos{0.f, -0.25f, -0.7f};
  float yaw = 0.f;
  float scale = 0.0035f;
};
std::mutex g_arenaMutex;
// AURORA_XR_ARENA_AT_HEAD=1 (unattended perf runs): each stage's arena is
// placed by the default offset from wherever the headset is and looks when
// the stage is set, not from the tracking origin, so a headset left lying
// anywhere still has the stage in view. Applied by the XR thread.
std::atomic<bool> g_arenaToHead{false};
// The current stage's arena (aurora_xr_set_stage).
ArenaPose g_arena;
// Where every stage starts (AURORA_XR_ARENA_POS, _YAW, _SCALE), before its
// own scale.
ArenaPose g_defaultArena;
// A placement the player made on purpose for every stage (aurora_xr_adopt_stage):
// a world pose, before any stage's scale. While set, a stage not placed yet
// starts there instead of in front of the head.
ArenaPose g_adoptedArena;
bool g_adoptedSet = false;
bool g_arenaInit = false;
// The current stage (aurora_xr_set_stage; -1: none yet), and where the
// player has put each stage this session. Placing one stage leaves the
// others where they were.
int g_stage = -1;
std::unordered_map<int, ArenaPose> g_stageArenas;
// The game point (game units) placed at the arena position: a stage whose
// geometry sits far from the origin is centered on its action.
std::array<float, 3> g_arenaCenter{};
// The current stage's size against the default scale.
float g_stageScale = 1.f;
// A zoom on top of the arena's scale, for the 3D view only (the placement
// the player grabs keeps its own): aurora_xr_set_stage_zoom.
float g_stageZoom = 1.f;
// The HUD over the 3D view even in mixed reality, for the current stage
// (aurora_xr_set_hud_on_top).
std::atomic<bool> g_hudOnTop{false};
// The stage's highest floor (game units; NaN: unknown), for the HUD.
float g_stageTop = std::numeric_limits<float>::quiet_NaN();
// The stage's footprint on a real surface (aurora_xr_set_stage_footprint;
// game units), for its shadow there.
struct Footprint {
  bool valid = false;
  float xmin = 0.f, xmax = 0.f, zmin = 0.f, zmax = 0.f, ybottom = 0.f;
};
Footprint g_footprint;

// Set by the game every fight frame (aurora_xr_set_paused): the fight is
// paused, so the controllers point and grab instead of playing.
std::atomic<bool> g_fightPaused{false};
// Set by the game (aurora_xr_set_placing): the fight is held before it
// starts for placing the stage. 0 no; 1 with the legend; 2 with the cards.
std::atomic<int> g_placing{0};

// The pictures shown while placing (aurora_xr_set_placing_image): given on
// the game thread, made into swapchains on the XR thread when first shown.
struct PlacingImage {
  std::vector<uint8_t> rgba; // straight alpha; kept for a new session
  uint32_t width = 0, height = 0;
  XrSwapchain swapchain = XR_NULL_HANDLE;
};
std::mutex g_placingMutex;
// The cards and the legend for Touch controllers, then for tracked hands.
std::array<PlacingImage, 4> g_placingImages;
// Looping clips shown over the cards' pictures (aurora_xr_set_placing_clip):
// a grid of frames, played forward then back. 0-2 on the controllers'
// cards, 3-5 on the hands'.
struct PlacingClip {
  PlacingImage atlas;
  int frames = 0, cols = 0, frameW = 0, frameH = 0;
  float fps = 12.f;
  int x = 0, y = 0, w = 0, h = 0; // on its cards, in their pixels
};
std::array<PlacingClip, 6> g_placingClips;

// The board is being placed on a real surface (board_place_frame): the
// controllers place it, and the game sees a pad with nothing pressed.
std::atomic<bool> g_boardPlacing{false};

// Set by the game (aurora_xr_set_passthrough): mixed reality shows the room,
// full VR doesn't. The XR thread pauses or starts passthrough to match.
std::atomic<bool> g_passthroughWanted{true};

// Controller input, written by the XR thread, read by the game thread.
std::mutex g_padMutex;
PADStatus g_pad{};
bool g_padValid = false;
// Where a laser points on the virtual screen's picture, for the game
// (aurora_xr_screen_pointer). Under g_padMutex.
struct ScreenPointer {
  bool valid = false, pressed = false;
  float x = 0.f, y = 0.f;
};
ScreenPointer g_screenPointer;
// A hand is pressing the screen's Back button (aurora_xr_screen_back).
// Under g_padMutex.
bool g_screenBack = false;
// The frame's size as the game draws it: the picture is letterboxed on the
// screen when its aspect differs from the screen's.
std::atomic<uint32_t> g_contentW{0}, g_contentH{0};

// XR thread only, except where noted.
struct Bridge {
  XrInstance instance = XR_NULL_HANDLE;
  XrSystemId systemId = XR_NULL_SYSTEM_ID;
  XrSession session = XR_NULL_HANDLE;
  XrSpace space = XR_NULL_HANDLE;
  int64_t swapFormat = 0;
  bool hasPassthroughExt = false;
  XrPassthroughFB passthrough = XR_NULL_HANDLE;
  XrPassthroughLayerFB passthroughLayer = XR_NULL_HANDLE;
  bool running = false;

  // Dawn's device (borrowed: Dawn owns it), set by begin_frame before the XR
  // thread starts. `queue` is the spare queue Dawn created for us (index
  // queueIndex of Dawn's family); the runtime gets it in the graphics binding.
  VkInstance vkInstance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice dev = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queueFamily = 0;
  uint32_t queueIndex = 1;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties memProps{};
  PFN_vkImportSemaphoreFdKHR importSemaphoreFd = nullptr;
  // XR_KHR_vulkan_enable2, for Dawn's creation hooks (main thread).
  PFN_xrCreateVulkanInstanceKHR createVulkanInstance = nullptr;
  PFN_xrCreateVulkanDeviceKHR createVulkanDevice = nullptr;
  PFN_xrGetVulkanGraphicsDevice2KHR getVulkanGraphicsDevice = nullptr;
  bool prepared = false; // prepare_device ran and Dawn's hooks are set

  bool hasRefreshRateExt = false;
  bool hasPerfSettingsExt = false;
  // Application SpaceWarp (XR_FB_space_warp, AURORA_XR_APPSW): the runtime
  // runs the app at half the display rate and synthesizes the frames between
  // from each frame's motion vectors and depth, at this size per eye.
  bool hasSpaceWarpExt = false;
  bool appSw = false;
  uint32_t mvWidth = 0, mvHeight = 0;
#ifdef XR_META_performance_metrics
  // The runtime's own counters (XR_META_performance_metrics): the
  // compositor's GPU time per display frame, for dynamic resolution's budget.
  bool hasPerfMetricsExt = false;
  PFN_xrSetPerformanceMetricsStateMETA setPerfMetricsState = nullptr;
  PFN_xrQueryPerformanceMetricsCounterMETA queryPerfMetric = nullptr;
  XrPath compositorGpuPath = XR_NULL_PATH;
#endif
  PFN_xrPerfSettingsSetPerformanceLevelEXT setPerformanceLevel = nullptr;
  PFN_xrEnumerateDisplayRefreshRatesFB enumerateRefreshRates = nullptr;
  PFN_xrRequestDisplayRefreshRateFB requestRefreshRate = nullptr;
  PFN_xrGetDisplayRefreshRateFB getRefreshRate = nullptr;
  uint64_t displayFrame = 0;

  PFN_xrCreatePassthroughFB createPassthrough = nullptr;
  PFN_xrDestroyPassthroughFB destroyPassthrough = nullptr;
  PFN_xrCreatePassthroughLayerFB createPassthroughLayer = nullptr;
  PFN_xrDestroyPassthroughLayerFB destroyPassthroughLayer = nullptr;
  PFN_xrPassthroughStartFB startPassthrough = nullptr;
  PFN_xrPassthroughPauseFB pausePassthrough = nullptr;
  bool passthroughRunning = false;
  // The room's grade (XR worker: the room's grade): passthrough styles, and
  // colour LUTs where XR_META_passthrough_color_lut is offered.
  PFN_xrPassthroughLayerSetStyleFB setPassthroughStyle = nullptr;
  bool hasColorLutExt = false;
  uint32_t maxLutResolution = 0;
  PFN_xrCreatePassthroughColorLutMETA createColorLut = nullptr;
  PFN_xrDestroyPassthroughColorLutMETA destroyColorLut = nullptr;
  PFN_xrUpdatePassthroughColorLutMETA updateColorLut = nullptr;
  // The real surface under the stage (XR worker: the surface under the
  // stage): XR_META_environment_raycast, which needs XR_EXT_future, and
  // XR_EXT_local_floor's floor where there is no raycast.
  bool hasFutureExt = false, hasEnvRaycastExt = false, hasLocalFloorExt = false;
  bool envRaycast = false; // enabled, and the system says it can
  PFN_xrCreateEnvironmentRaycasterAsyncMETA createRaycasterAsync = nullptr;
  PFN_xrCreateEnvironmentRaycasterCompleteMETA createRaycasterComplete = nullptr;
  PFN_xrDestroyEnvironmentRaycasterMETA destroyRaycaster = nullptr;
  PFN_xrPerformEnvironmentRaycastMETA performRaycast = nullptr;
  PFN_xrPollFutureEXT pollFuture = nullptr;
  PFN_xrCancelFutureEXT cancelFuture = nullptr;
  XrSpace floorSpace = XR_NULL_HANDLE; // LOCAL_FLOOR
  // The stage spot's spatial anchor (the stage spot's anchor):
  // XR_FB_spatial_entity to make and locate it, XR_META_spatial_entity_
  // persistence to save it, XR_META_spatial_entity_discovery to find it again.
  bool hasSpatialEntityExt = false, hasSpacePersistenceExt = false, hasSpaceDiscoveryExt = false;
  bool anchors = false; // all three, and the system says it can
  PFN_xrCreateSpatialAnchorFB createSpatialAnchor = nullptr;
  PFN_xrSetSpaceComponentStatusFB setComponentStatus = nullptr;
  PFN_xrGetSpaceComponentStatusFB getComponentStatus = nullptr;
  PFN_xrSaveSpacesMETA saveSpaces = nullptr;
  PFN_xrEraseSpacesMETA eraseSpaces = nullptr;
  PFN_xrDiscoverSpacesMETA discoverSpaces = nullptr;
  PFN_xrRetrieveSpaceDiscoveryResultsMETA retrieveDiscovery = nullptr;
  XrSwapchain shadowSwapchain = XR_NULL_HANDLE;
  // The most composition layers a frame may have (XrSystemGraphicsProperties).
  uint32_t maxLayers = 16;
#ifdef XR_META_boundary_visibility
  // The Guardian boundary, hidden while passthrough shows the room.
  bool hasBoundaryVisibilityExt = false;
  PFN_xrRequestBoundaryVisibilityMETA requestBoundaryVisibility = nullptr;
  bool boundarySuppressed = false; // last request
#endif

  std::string dumpDir; // AURORA_XR_DUMP

  // Controller input
  XrActionSet actionSet = XR_NULL_HANDLE;
  XrAction stickMain = XR_NULL_HANDLE, stickC = XR_NULL_HANDLE;
  XrAction btnA = XR_NULL_HANDLE, btnB = XR_NULL_HANDLE, btnX = XR_NULL_HANDLE, btnY = XR_NULL_HANDLE;
  XrAction btnZ = XR_NULL_HANDLE, btnStart = XR_NULL_HANDLE;
  XrAction trigL = XR_NULL_HANDLE, trigR = XR_NULL_HANDLE;
  // Pointing and grabbing the arena while a fight is paused.
  std::array<XrAction, 2> aim{}, grab{};
  std::array<XrSpace, 2> aimSpace{};
  // Uploads to static swapchains still on the GPU: freed once their fence
  // signals (free_finished_uploads), never waited on mid-session.
  struct Upload {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
  };
  std::vector<Upload> uploads;
  bool pointerMode = false; // last frame: the grips grab (paused fight, or a laser on the screen) instead of pressing Z
  bool triggersClick = false; // last frame: a laser on the screen, so the triggers click instead of pressing L and R
  int reelHand = -1;        // last frame: the controller dragging the screen; its stick pushes and pulls it
  std::array<float, 2> stickY{}; // each controller's raw thumbstick Y (left: control stick, right: C-stick)
  std::array<float, 2> stickX{}; // and X
  bool backButton = false;       // B or Y held (placing the board: back a step)
  // Hand tracking (XR_EXT_hand_tracking): pinches grab the arena too.
  bool hasHandTrackingExt = false;
  PFN_xrCreateHandTrackerEXT createHandTracker = nullptr;
  PFN_xrDestroyHandTrackerEXT destroyHandTracker = nullptr;
  PFN_xrLocateHandJointsEXT locateHandJoints = nullptr;
  std::array<XrHandTrackerEXT, 2> handTrackers{};
  XrVector3f head{};        // between the eyes, from the latest views
  XrQuaternionf headOrientation{0.f, 0.f, 0.f, 1.f};
  bool focused = false;

  // Laser layers: static textures, drawn as quads at display rate.
  XrSwapchain beamSwapchain = XR_NULL_HANDLE, dotSwapchain = XR_NULL_HANDLE;
  XrSwapchain barSwapchain = XR_NULL_HANDLE; // the screen's grab bar
  XrSwapchain backSwapchain = XR_NULL_HANDLE; // the Back button beside it
  // SpaceWarp's motion vectors for an image submitted again: all zero, so
  // the compositor reprojects it for the head without moving anything in it.
  XrSwapchain zeroMotionSwapchain = XR_NULL_HANDLE;
  bool hasColorScaleBias = false;

  uint64_t framesShown = 0, fightFrames = 0;
  std::array<uint64_t, 4> shownHistogram{}; // 3D images shown for 1, 2, 3, 4+ display frames
  // XR loop timing (logged): xrWaitFrame's return to xrEndFrame, the gap
  // between returns, and the time left before the predicted display.
  int64_t loopWaitNs = 0, loopLastWaitNs = 0, loopMaxNs = 0, gapMaxNs = 0, leadMinNs = INT64_MAX;
  uint32_t loopOver4 = 0, gapOver10 = 0;
  // Sections of a slow loop (over 4 ms): begin, then release and acquire per
  // stream, then the rest up to xrEndFrame, and xrEndFrame itself.
  // 0-8 from xrBeginFrame on; 9-12 the loop's start before it (uploads,
  // pacing, input, views).
  std::array<int64_t, 13> sectionNs{}, sectionMaxNs{};
  std::array<uint32_t, 13> sectionWorst{}; // slow loops each section was the largest part of
  std::array<uint64_t, kStreamCount> releasedSinceStats{};
  std::chrono::steady_clock::time_point statsStart;
};
Bridge B;

#define XR_TRY(x)                                                                                                      \
  do {                                                                                                                 \
    XrResult r_ = (x);                                                                                                 \
    if (XR_FAILED(r_)) {                                                                                               \
      Log.error("{} failed: XrResult {}", #x, static_cast<int>(r_));                                                   \
      return false;                                                                                                    \
    }                                                                                                                  \
  } while (0)
#define VK_TRY(x)                                                                                                      \
  do {                                                                                                                 \
    VkResult r_ = (x);                                                                                                 \
    if (r_ != VK_SUCCESS) {                                                                                            \
      Log.error("{} failed: VkResult {}", #x, static_cast<int>(r_));                                                   \
      return false;                                                                                                    \
    }                                                                                                                  \
  } while (0)

bool env_flag(const char* name, bool def) {
  const char* v = std::getenv(name);
  if (!v || !*v)
    return def;
  return !(v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F');
}

float env_float(const char* name, float def) {
  const char* v = std::getenv(name);
  if (!v || !*v)
    return def;
  char* end = nullptr;
  const float f = std::strtof(v, &end);
  return end != v ? f : def;
}

// The room's colour grade (T5) is off unless asked for, and has no launcher
// setting: how dark a room should go depends on how lit it is, which it
// doesn't know yet. Only AURORA_XR_GRADE=1 turns it on (the dev tools'
// Tint page tunes it, with it on).
bool grade_wanted() { return env_flag("AURORA_XR_GRADE", false); }

// How often the GPU timing and dynamic resolution lines are logged:
// AURORA_XR_LOG_PERIOD seconds, 10 by default (1 lines them up with what
// is on screen when profiling a stage).
std::chrono::steady_clock::duration log_period() {
  static const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<float>(std::clamp(env_float("AURORA_XR_LOG_PERIOD", 10.f), 0.25f, 600.f)));
  return period;
}

template <typename F>
F xr_proc(const char* name) {
  PFN_xrVoidFunction fn = nullptr;
  if (XR_FAILED(xrGetInstanceProcAddr(B.instance, name, &fn)))
    return nullptr;
  return reinterpret_cast<F>(fn);
}

uint32_t find_memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
  for (uint32_t i = 0; i < B.memProps.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (B.memProps.memoryTypes[i].propertyFlags & want) == want)
      return i;
  return UINT32_MAX;
}

VkImageMemoryBarrier barrier(VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags srcA,
                             VkAccessFlags dstA, uint32_t srcQ = VK_QUEUE_FAMILY_IGNORED,
                             uint32_t dstQ = VK_QUEUE_FAMILY_IGNORED,
                             VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.srcAccessMask = srcA;
  b.dstAccessMask = dstA;
  b.oldLayout = from;
  b.newLayout = to;
  b.srcQueueFamilyIndex = srcQ;
  b.dstQueueFamilyIndex = dstQ;
  b.image = img;
  b.subresourceRange = {aspect, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS};
  return b;
}

// ---------------------------------------------------------------- XR thread: setup

bool create_instance() {
#ifdef __ANDROID__
  auto* env = static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv());
  JavaVM* vm = nullptr;
  env->GetJavaVM(&vm);
  auto localActivity = static_cast<jobject>(SDL_GetAndroidActivity());
  jobject activity = env->NewGlobalRef(localActivity); // kept for the instance's lifetime
  env->DeleteLocalRef(localActivity);
  PFN_xrInitializeLoaderKHR initLoader = nullptr;
  XR_TRY(xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR",
                               reinterpret_cast<PFN_xrVoidFunction*>(&initLoader)));
  XrLoaderInitInfoAndroidKHR li{XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR};
  li.applicationVM = vm;
  li.applicationContext = activity;
  XR_TRY(initLoader(reinterpret_cast<const XrLoaderInitInfoBaseHeaderKHR*>(&li)));
#endif
  uint32_t n = 0;
  XR_TRY(xrEnumerateInstanceExtensionProperties(nullptr, 0, &n, nullptr));
  std::vector<XrExtensionProperties> props(n, {XR_TYPE_EXTENSION_PROPERTIES});
  XR_TRY(xrEnumerateInstanceExtensionProperties(nullptr, n, &n, props.data()));
  // AURORA_XR_LOG_EXTENSIONS=1: every extension the runtime offers.
  if (env_flag("AURORA_XR_LOG_EXTENSIONS", false)) {
    // In lines of 12: logcat cuts a long line short.
    std::string list;
    for (size_t i = 0; i < props.size(); ++i) {
      list += fmt::format("{}{} v{}", list.empty() ? "" : ", ", props[i].extensionName, props[i].extensionVersion);
      if ((i + 1) % 12 == 0 || i + 1 == props.size()) {
        Log.info("OpenXR extensions offered: {}", list);
        list.clear();
      }
    }
  }
  bool hasVk2 = false;
  for (const auto& p : props) {
    hasVk2 |= !std::strcmp(p.extensionName, XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME);
    B.hasPassthroughExt |= !std::strcmp(p.extensionName, XR_FB_PASSTHROUGH_EXTENSION_NAME);
    B.hasRefreshRateExt |= !std::strcmp(p.extensionName, XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME);
    B.hasColorScaleBias |= !std::strcmp(p.extensionName, XR_KHR_COMPOSITION_LAYER_COLOR_SCALE_BIAS_EXTENSION_NAME);
    B.hasPerfSettingsExt |= !std::strcmp(p.extensionName, XR_EXT_PERFORMANCE_SETTINGS_EXTENSION_NAME);
#ifdef XR_META_performance_metrics
    B.hasPerfMetricsExt |= !std::strcmp(p.extensionName, XR_META_PERFORMANCE_METRICS_EXTENSION_NAME);
#endif
    B.hasHandTrackingExt |= !std::strcmp(p.extensionName, XR_EXT_HAND_TRACKING_EXTENSION_NAME);
    B.hasSpaceWarpExt |= !std::strcmp(p.extensionName, XR_FB_SPACE_WARP_EXTENSION_NAME);
#ifdef XR_META_boundary_visibility
    B.hasBoundaryVisibilityExt |= !std::strcmp(p.extensionName, XR_META_BOUNDARY_VISIBILITY_EXTENSION_NAME);
#endif
    B.hasFutureExt |= !std::strcmp(p.extensionName, XR_EXT_FUTURE_EXTENSION_NAME);
    B.hasEnvRaycastExt |= !std::strcmp(p.extensionName, XR_META_ENVIRONMENT_RAYCAST_EXTENSION_NAME);
    B.hasLocalFloorExt |= !std::strcmp(p.extensionName, XR_EXT_LOCAL_FLOOR_EXTENSION_NAME);
    B.hasColorLutExt |= !std::strcmp(p.extensionName, XR_META_PASSTHROUGH_COLOR_LUT_EXTENSION_NAME);
    B.hasSpatialEntityExt |= !std::strcmp(p.extensionName, XR_FB_SPATIAL_ENTITY_EXTENSION_NAME);
    B.hasSpacePersistenceExt |= !std::strcmp(p.extensionName, XR_META_SPATIAL_ENTITY_PERSISTENCE_EXTENSION_NAME);
    B.hasSpaceDiscoveryExt |= !std::strcmp(p.extensionName, XR_META_SPATIAL_ENTITY_DISCOVERY_EXTENSION_NAME);
  }
  if (!hasVk2) {
    Log.error("OpenXR runtime lacks XR_KHR_vulkan_enable2");
    return false;
  }
  std::vector<const char*> exts{XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME};
  if (B.hasPassthroughExt)
    exts.push_back(XR_FB_PASSTHROUGH_EXTENSION_NAME);
  if (B.hasRefreshRateExt)
    exts.push_back(XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME);
  if (B.hasColorScaleBias)
    exts.push_back(XR_KHR_COMPOSITION_LAYER_COLOR_SCALE_BIAS_EXTENSION_NAME);
  if (B.hasPerfSettingsExt)
    exts.push_back(XR_EXT_PERFORMANCE_SETTINGS_EXTENSION_NAME);
#ifdef XR_META_performance_metrics
  if (B.hasPerfMetricsExt)
    exts.push_back(XR_META_PERFORMANCE_METRICS_EXTENSION_NAME);
#endif
  // On wherever the runtime offers it, unless the host turned it off
  // (AuroraConfig::xrNoSpaceWarp; AURORA_XR_APPSW=0/1 overrides). Multiview
  // is decided later (with Dawn's device): size_stereo_stream.
  B.hasSpaceWarpExt &= env_flag("AURORA_XR_APPSW", !g_config.xrNoSpaceWarp);
  if (B.hasSpaceWarpExt)
    exts.push_back(XR_FB_SPACE_WARP_EXTENSION_NAME);
  // AURORA_XR_HANDS=0 leaves hand tracking off, for measuring what it costs.
  B.hasHandTrackingExt &= env_flag("AURORA_XR_HANDS", true);
  if (B.hasHandTrackingExt)
    exts.push_back(XR_EXT_HAND_TRACKING_EXTENSION_NAME);
#ifdef XR_META_boundary_visibility
  // AURORA_XR_BOUNDARY=1 keeps the Guardian boundary in mixed reality.
  B.hasBoundaryVisibilityExt &= !env_flag("AURORA_XR_BOUNDARY", false);
  if (B.hasBoundaryVisibilityExt)
    exts.push_back(XR_META_BOUNDARY_VISIBILITY_EXTENSION_NAME);
#endif
  // The surface under the stage, for its shadow. Raycast is offered only to
  // apps declaring USE_SCENE, on Quest 3 and 3S; AURORA_XR_RAYCAST=0 leaves
  // it off, AURORA_XR_LOCAL_FLOOR=0 the floor that stands in for it.
  B.hasEnvRaycastExt &= B.hasFutureExt && env_flag("AURORA_XR_RAYCAST", !g_config.xrNoStageShadow);
  if (B.hasEnvRaycastExt) {
    exts.push_back(XR_EXT_FUTURE_EXTENSION_NAME);
    exts.push_back(XR_META_ENVIRONMENT_RAYCAST_EXTENSION_NAME);
  }
  B.hasLocalFloorExt &= env_flag("AURORA_XR_LOCAL_FLOOR", true);
  if (B.hasLocalFloorExt)
    exts.push_back(XR_EXT_LOCAL_FLOOR_EXTENSION_NAME);
  // The room's grade by colour LUT; without it, brightness, contrast and
  // saturation (AURORA_XR_GRADE_LUT=0 forces those, AURORA_XR_GRADE=0 none).
  B.hasColorLutExt &= B.hasPassthroughExt && grade_wanted() &&
                      env_float("AURORA_XR_GRADE_LUT", 16.f) >= 2.f;
  if (B.hasColorLutExt)
    exts.push_back(XR_META_PASSTHROUGH_COLOR_LUT_EXTENSION_NAME);
  // The stage spot kept in the room (offered to apps declaring
  // USE_ANCHOR_API); AURORA_XR_ANCHORS=0 leaves it to the launch.
  const bool anchorExts = B.hasSpatialEntityExt && B.hasSpacePersistenceExt && B.hasSpaceDiscoveryExt &&
                          env_flag("AURORA_XR_ANCHORS", true);
  B.hasSpatialEntityExt &= anchorExts;
  B.hasSpacePersistenceExt &= anchorExts;
  B.hasSpaceDiscoveryExt &= anchorExts;
  if (anchorExts) {
    exts.push_back(XR_FB_SPATIAL_ENTITY_EXTENSION_NAME);
    exts.push_back(XR_META_SPATIAL_ENTITY_PERSISTENCE_EXTENSION_NAME);
    exts.push_back(XR_META_SPATIAL_ENTITY_DISCOVERY_EXTENSION_NAME);
  }
  XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
#ifdef __ANDROID__
  exts.push_back(XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME);
  XrInstanceCreateInfoAndroidKHR aci{XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR};
  aci.applicationVM = vm;
  aci.applicationActivity = activity;
  ci.next = &aci;
#endif
  std::strncpy(ci.applicationInfo.applicationName, "melee-pc", XR_MAX_APPLICATION_NAME_SIZE - 1);
  std::strncpy(ci.applicationInfo.engineName, "aurora", XR_MAX_ENGINE_NAME_SIZE - 1);
  ci.applicationInfo.apiVersion = XR_API_VERSION_1_0;
  ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
  ci.enabledExtensionNames = exts.data();
  XR_TRY(xrCreateInstance(&ci, &B.instance));

  XrInstanceProperties ip{XR_TYPE_INSTANCE_PROPERTIES};
  XR_TRY(xrGetInstanceProperties(B.instance, &ip));
  Log.info("OpenXR runtime: {} {}.{}.{} (XR_FB_passthrough {})", ip.runtimeName, XR_VERSION_MAJOR(ip.runtimeVersion),
           XR_VERSION_MINOR(ip.runtimeVersion), XR_VERSION_PATCH(ip.runtimeVersion),
           B.hasPassthroughExt ? "available" : "absent");

  XrSystemGetInfo si{XR_TYPE_SYSTEM_GET_INFO};
  si.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
  XR_TRY(xrGetSystem(B.instance, &si, &B.systemId));
  {
    XrSystemProperties sp{XR_TYPE_SYSTEM_PROPERTIES};
    XrSystemEnvironmentRaycastPropertiesMETA raycast{XR_TYPE_SYSTEM_ENVIRONMENT_RAYCAST_PROPERTIES_META};
    XrSystemPassthroughColorLutPropertiesMETA lut{XR_TYPE_SYSTEM_PASSTHROUGH_COLOR_LUT_PROPERTIES_META};
    if (B.hasEnvRaycastExt) {
      raycast.next = sp.next;
      sp.next = &raycast;
    }
    if (B.hasColorLutExt) {
      lut.next = sp.next;
      sp.next = &lut;
    }
    if (XR_SUCCEEDED(xrGetSystemProperties(B.instance, B.systemId, &sp))) {
      B.maxLayers = std::max<uint32_t>(sp.graphicsProperties.maxLayerCount, 1);
      B.envRaycast = B.hasEnvRaycastExt && raycast.supportsEnvironmentRaycast;
      B.maxLutResolution = B.hasColorLutExt ? lut.maxColorLutResolution : 0;
    }
    Log.info("System {}: at most {} composition layers a frame; environment raycast {}, local floor {}, "
             "passthrough colour LUTs {}",
             sp.systemName, B.maxLayers,
             B.envRaycast ? "available" : B.hasEnvRaycastExt ? "offered but unsupported" : "not offered",
             B.hasLocalFloorExt ? "available" : "not offered",
             B.maxLutResolution > 0 ? fmt::format("up to {}^3", B.maxLutResolution) : std::string{"unavailable"});
  }
  if (B.hasSpaceWarpExt) {
    XrSystemSpaceWarpPropertiesFB sw{XR_TYPE_SYSTEM_SPACE_WARP_PROPERTIES_FB};
    XrSystemProperties sp{XR_TYPE_SYSTEM_PROPERTIES};
    sp.next = &sw;
    if (XR_SUCCEEDED(xrGetSystemProperties(B.instance, B.systemId, &sp)) && sw.recommendedMotionVectorImageRectWidth > 0) {
      B.mvWidth = sw.recommendedMotionVectorImageRectWidth;
      B.mvHeight = sw.recommendedMotionVectorImageRectHeight;
      B.appSw = true;
      Log.info("Application SpaceWarp on: motion vectors and depth at {}x{} per eye", B.mvWidth, B.mvHeight);
    } else {
      Log.warn("XR_FB_space_warp offered without a motion vector size; SpaceWarp off");
    }
  }
  if (B.hasPassthroughExt) {
    B.createPassthrough = xr_proc<PFN_xrCreatePassthroughFB>("xrCreatePassthroughFB");
    B.destroyPassthrough = xr_proc<PFN_xrDestroyPassthroughFB>("xrDestroyPassthroughFB");
    B.createPassthroughLayer = xr_proc<PFN_xrCreatePassthroughLayerFB>("xrCreatePassthroughLayerFB");
    B.destroyPassthroughLayer = xr_proc<PFN_xrDestroyPassthroughLayerFB>("xrDestroyPassthroughLayerFB");
    B.startPassthrough = xr_proc<PFN_xrPassthroughStartFB>("xrPassthroughStartFB");
    B.pausePassthrough = xr_proc<PFN_xrPassthroughPauseFB>("xrPassthroughPauseFB");
    B.setPassthroughStyle = xr_proc<PFN_xrPassthroughLayerSetStyleFB>("xrPassthroughLayerSetStyleFB");
  }
  if (B.hasColorLutExt) {
    B.createColorLut = xr_proc<PFN_xrCreatePassthroughColorLutMETA>("xrCreatePassthroughColorLutMETA");
    B.destroyColorLut = xr_proc<PFN_xrDestroyPassthroughColorLutMETA>("xrDestroyPassthroughColorLutMETA");
    B.updateColorLut = xr_proc<PFN_xrUpdatePassthroughColorLutMETA>("xrUpdatePassthroughColorLutMETA");
    if (!B.createColorLut || !B.destroyColorLut)
      B.maxLutResolution = 0;
  }
  if (B.hasRefreshRateExt) {
    B.enumerateRefreshRates = xr_proc<PFN_xrEnumerateDisplayRefreshRatesFB>("xrEnumerateDisplayRefreshRatesFB");
    B.requestRefreshRate = xr_proc<PFN_xrRequestDisplayRefreshRateFB>("xrRequestDisplayRefreshRateFB");
    B.getRefreshRate = xr_proc<PFN_xrGetDisplayRefreshRateFB>("xrGetDisplayRefreshRateFB");
  }
  if (B.hasPerfSettingsExt)
    B.setPerformanceLevel = xr_proc<PFN_xrPerfSettingsSetPerformanceLevelEXT>("xrPerfSettingsSetPerformanceLevelEXT");
#ifdef XR_META_performance_metrics
  if (B.hasPerfMetricsExt) {
    B.setPerfMetricsState = xr_proc<PFN_xrSetPerformanceMetricsStateMETA>("xrSetPerformanceMetricsStateMETA");
    B.queryPerfMetric = xr_proc<PFN_xrQueryPerformanceMetricsCounterMETA>("xrQueryPerformanceMetricsCounterMETA");
    auto enumerate =
        xr_proc<PFN_xrEnumeratePerformanceMetricsCounterPathsMETA>("xrEnumeratePerformanceMetricsCounterPathsMETA");
    uint32_t count = 0;
    std::vector<XrPath> paths;
    if (enumerate && XR_SUCCEEDED(enumerate(B.instance, 0, &count, nullptr))) {
      paths.resize(count);
      enumerate(B.instance, count, &count, paths.data());
    }
    std::string list;
    for (XrPath path : paths) {
      char name[XR_MAX_PATH_LENGTH];
      uint32_t len = 0;
      if (XR_FAILED(xrPathToString(B.instance, path, sizeof name, &len, name)))
        continue;
      list += fmt::format("{}{}", list.empty() ? "" : ", ", name);
      if (std::strstr(name, "compositor") && std::strstr(name, "gpu"))
        B.compositorGpuPath = path;
    }
    Log.info("XR_META_performance_metrics counters: {}", list.empty() ? "none" : list);
  }
#endif
  if (B.hasHandTrackingExt) {
    B.createHandTracker = xr_proc<PFN_xrCreateHandTrackerEXT>("xrCreateHandTrackerEXT");
    B.destroyHandTracker = xr_proc<PFN_xrDestroyHandTrackerEXT>("xrDestroyHandTrackerEXT");
    B.locateHandJoints = xr_proc<PFN_xrLocateHandJointsEXT>("xrLocateHandJointsEXT");
  }
  if (B.envRaycast) {
    B.createRaycasterAsync =
        xr_proc<PFN_xrCreateEnvironmentRaycasterAsyncMETA>("xrCreateEnvironmentRaycasterAsyncMETA");
    B.createRaycasterComplete =
        xr_proc<PFN_xrCreateEnvironmentRaycasterCompleteMETA>("xrCreateEnvironmentRaycasterCompleteMETA");
    B.destroyRaycaster = xr_proc<PFN_xrDestroyEnvironmentRaycasterMETA>("xrDestroyEnvironmentRaycasterMETA");
    B.performRaycast = xr_proc<PFN_xrPerformEnvironmentRaycastMETA>("xrPerformEnvironmentRaycastMETA");
    B.pollFuture = xr_proc<PFN_xrPollFutureEXT>("xrPollFutureEXT");
    B.cancelFuture = xr_proc<PFN_xrCancelFutureEXT>("xrCancelFutureEXT");
    B.envRaycast = B.createRaycasterAsync && B.createRaycasterComplete && B.destroyRaycaster && B.performRaycast &&
                   B.pollFuture;
  }
#ifdef XR_META_boundary_visibility
  if (B.hasBoundaryVisibilityExt)
    B.requestBoundaryVisibility = xr_proc<PFN_xrRequestBoundaryVisibilityMETA>("xrRequestBoundaryVisibilityMETA");
  Log.info("XR_META_boundary_visibility {}", B.requestBoundaryVisibility ? "available" : "unavailable");
#endif
  if (B.hasSpatialEntityExt) {
    XrSystemProperties sp{XR_TYPE_SYSTEM_PROPERTIES};
    XrSystemSpatialEntityPropertiesFB se{XR_TYPE_SYSTEM_SPATIAL_ENTITY_PROPERTIES_FB};
    XrSystemSpacePersistencePropertiesMETA sv{XR_TYPE_SYSTEM_SPACE_PERSISTENCE_PROPERTIES_META};
    XrSystemSpaceDiscoveryPropertiesMETA sd{XR_TYPE_SYSTEM_SPACE_DISCOVERY_PROPERTIES_META};
    sp.next = &se;
    se.next = &sv;
    sv.next = &sd;
    const bool can = XR_SUCCEEDED(xrGetSystemProperties(B.instance, B.systemId, &sp)) && se.supportsSpatialEntity &&
                     sv.supportsSpacePersistence && sd.supportsSpaceDiscovery;
    B.createSpatialAnchor = xr_proc<PFN_xrCreateSpatialAnchorFB>("xrCreateSpatialAnchorFB");
    B.setComponentStatus = xr_proc<PFN_xrSetSpaceComponentStatusFB>("xrSetSpaceComponentStatusFB");
    B.getComponentStatus = xr_proc<PFN_xrGetSpaceComponentStatusFB>("xrGetSpaceComponentStatusFB");
    B.saveSpaces = xr_proc<PFN_xrSaveSpacesMETA>("xrSaveSpacesMETA");
    B.eraseSpaces = xr_proc<PFN_xrEraseSpacesMETA>("xrEraseSpacesMETA");
    B.discoverSpaces = xr_proc<PFN_xrDiscoverSpacesMETA>("xrDiscoverSpacesMETA");
    B.retrieveDiscovery = xr_proc<PFN_xrRetrieveSpaceDiscoveryResultsMETA>("xrRetrieveSpaceDiscoveryResultsMETA");
    B.anchors = can && B.createSpatialAnchor && B.setComponentStatus && B.getComponentStatus && B.saveSpaces &&
                B.eraseSpaces && B.discoverSpaces && B.retrieveDiscovery;
  }
  Log.info("Spatial anchors for the stage spot: {}", B.anchors ? "available"
                                                   : B.hasSpatialEntityExt ? "offered but unsupported"
                                                                           : "not offered (USE_ANCHOR_API?)");
  return true;
}

// Dawn's Vulkan instance and device, created through the runtime
// (XR_KHR_vulkan_enable2) so it can add what it needs and pick the GPU.
// Called by Dawn on the main thread during adapter discovery and device
// creation (prepare_device sets these up).
VkResult hook_create_instance(void*, PFN_vkGetInstanceProcAddr getProcAddr, const VkInstanceCreateInfo* info,
                              VkInstance* out) {
  XrVulkanInstanceCreateInfoKHR xici{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
  xici.systemId = B.systemId;
  xici.pfnGetInstanceProcAddr = getProcAddr;
  xici.vulkanCreateInfo = info;
  VkResult vr = VK_SUCCESS;
  const XrResult r = B.createVulkanInstance(B.instance, &xici, out, &vr);
  if (XR_FAILED(r)) {
    Log.error("xrCreateVulkanInstanceKHR failed: XrResult {}", static_cast<int>(r));
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  return vr;
}

VkPhysicalDevice hook_physical_device(void*, VkInstance instance) {
  XrVulkanGraphicsDeviceGetInfoKHR gi{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
  gi.systemId = B.systemId;
  gi.vulkanInstance = instance;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  if (XR_FAILED(B.getVulkanGraphicsDevice(B.instance, &gi, &phys)))
    Log.error("xrGetVulkanGraphicsDevice2KHR failed");
  return phys;
}

VkResult hook_create_device(void*, PFN_vkGetInstanceProcAddr getProcAddr, VkPhysicalDevice phys,
                            const VkDeviceCreateInfo* info, VkDevice* out) {
  XrVulkanDeviceCreateInfoKHR xdci{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
  xdci.systemId = B.systemId;
  xdci.pfnGetInstanceProcAddr = getProcAddr;
  xdci.vulkanPhysicalDevice = phys;
  xdci.vulkanCreateInfo = info;
  VkResult vr = VK_SUCCESS;
  const XrResult r = B.createVulkanDevice(B.instance, &xdci, out, &vr);
  if (XR_FAILED(r)) {
    Log.error("xrCreateVulkanDeviceKHR failed: XrResult {}", static_cast<int>(r));
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  return vr;
}

// XR thread: everything it needs on Dawn's device (handles set by
// begin_frame from dawn::native::vulkan::GetDeviceVkHandles).
bool adopt_dawn_device() {
  if (!B.dev || !B.queue) {
    Log.error("Dawn's device has no spare queue for the OpenXR runtime");
    return false;
  }
  vkGetPhysicalDeviceMemoryProperties(B.phys, &B.memProps);
  VkPhysicalDeviceProperties pp;
  vkGetPhysicalDeviceProperties(B.phys, &pp);
  B.importSemaphoreFd =
      reinterpret_cast<PFN_vkImportSemaphoreFdKHR>(vkGetDeviceProcAddr(B.dev, "vkImportSemaphoreFdKHR"));
  if (!B.importSemaphoreFd) {
    Log.error("vkImportSemaphoreFdKHR missing on Dawn's device");
    return false;
  }
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = B.queueFamily;
  VK_TRY(vkCreateCommandPool(B.dev, &pci, nullptr, &B.pool));
  Log.info("Rendering on Dawn's VkDevice: {}, queue family {} (OpenXR queue {})", pp.deviceName, B.queueFamily,
           B.queueIndex);
  return true;
}

// The image's create info as Dawn should see it: the stream's UNORM format
// over the runtime's sRGB, mutable-format swapchain image, so Dawn stores the
// already sRGB-encoded bytes unchanged and the compositor decodes them.
VkImageCreateInfo swapchain_image_info(const Stream& st) {
  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.flags = st.ownFormat ? 0 : VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = st.format;
  ici.extent = {st.width, st.height, 1};
  ici.mipLevels = 1;
  ici.arrayLayers = st.layers;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  // The SpaceWarp images also take copies of the motion pass (render_eyes),
  // the color ones the staged ring's (release_staged).
  ici.usage = st.depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT
              : st.ownFormat
                  ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                             : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  return ici;
}

bool create_swapchain(Stream& st) {
  if (st.width == 0)
    return true; // not in use (the SpaceWarp streams without AURORA_XR_APPSW)
  XrSwapchainCreateInfo ci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
  ci.usageFlags = st.depth ? XR_SWAPCHAIN_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT |
                                XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT
                  : st.ownFormat ? XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT |
                                       XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT
                                 : XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_SRC_BIT |
                                       XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_MUTABLE_FORMAT_BIT;
  ci.format = st.swapFormat;
  ci.sampleCount = 1;
  ci.width = st.width;
  ci.height = st.height;
  ci.faceCount = 1;
  ci.arraySize = st.layers;
  ci.mipCount = 1;
  XR_TRY(xrCreateSwapchain(B.session, &ci, &st.swapchain));
  uint32_t n = 0;
  XR_TRY(xrEnumerateSwapchainImages(st.swapchain, 0, &n, nullptr));
  std::vector<XrSwapchainImageVulkan2KHR> imgs(n, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR});
  XR_TRY(xrEnumerateSwapchainImages(st.swapchain, n, &n, reinterpret_cast<XrSwapchainImageBaseHeader*>(imgs.data())));
  st.slots = std::vector<Slot>(n);
  Log.info("XR swapchain {}: {} images, {}x{} x{}", static_cast<int>(&st - g_streams.data()), n, st.width, st.height,
           st.layers);
  for (uint32_t i = 0; i < n; ++i) {
    Slot& s = st.slots[i];
    s.image = imgs[i].image;
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = B.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_TRY(vkAllocateCommandBuffers(B.dev, &cai, &s.cmd));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_TRY(vkCreateFence(B.dev, &fci, nullptr, &s.fence));
  }
  return true;
}

// A staged stream's own images: the swapchain's, in the format Dawn draws as
// (copied into the sRGB swapchain bit for bit), also a copy source. Copy
// destination too: Dawn's views of any texture ask for it (its clears).
VkImageCreateInfo ring_image_info(const Stream& st) {
  VkImageCreateInfo ici = swapchain_image_info(st);
  ici.flags = 0;
  ici.usage = (st.depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) |
              VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  return ici;
}

// Staging the eyes (AURORA_XR_STAGED, on with SpaceWarp): drawn into the
// runtime's images, a frame could start only once the compositor gave one
// back, and the render worker waited on it. With SpaceWarp's 60 Hz loop that
// came only as the next loop began, one late frame held its images into the
// next frame's, and four-CPU Brinstar stuck at 33 3D frames a second. Drawn
// into a ring of our own, nothing the render worker does waits on the
// runtime; the XR thread copies a finished image in when it hands it over
// (40 MB for the 3D image). Without SpaceWarp, four-CPU Brinstar finished
// frames 8 ms sooner but kept a little less resolution (1.07 against 1.18,
// the copy) for nothing gained: off. AURORA_XR_STAGED_IMAGES: the ring's size.
bool create_ring(Stream& st) {
  const bool stageable = &st == &g_streams[kStereo] || &st == &g_streams[kHud] || &st == &g_streams[kMotion] ||
                         &st == &g_streams[kMotionDepth];
  if (!stageable || st.width == 0 || !env_flag("AURORA_XR_STAGED", B.appSw))
    return true;
  const int count = std::clamp(static_cast<int>(env_float("AURORA_XR_STAGED_IMAGES", 4.f)), 2, 8);
  const VkImageCreateInfo ici = ring_image_info(st);
  st.ring = std::vector<Slot>(static_cast<size_t>(count));
  for (auto& s : st.ring) {
    VK_TRY(vkCreateImage(B.dev, &ici, nullptr, &s.image));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(B.dev, s.image, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_TRY(vkAllocateMemory(B.dev, &mai, nullptr, &s.memory));
    VK_TRY(vkBindImageMemory(B.dev, s.image, s.memory, 0));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = B.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_TRY(vkAllocateCommandBuffers(B.dev, &cai, &s.cmd));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_TRY(vkCreateFence(B.dev, &fci, nullptr, &s.fence));
    s.state = SlotState::Free;
  }
  st.staged = true;
  return true;
}

// Eye resolution for 3D fights: the runtime's recommendation scaled by
// AURORA_XR_EYE_SCALE (default 1.0). On a Quest 3 the full 1680x1760 per eye
// first cost 21-27 ms of GPU per frame; with the flat present skipped during
// fights, both eyes drawn straight into the shared image and the shader copy,
// it costs about 9-10 ms per game frame at 120 Hz.
bool size_stereo_stream() {
  uint32_t n = 0;
  XR_TRY(xrEnumerateViewConfigurationViews(B.instance, B.systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &n,
                                           nullptr));
  std::vector<XrViewConfigurationView> views(n, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
  XR_TRY(xrEnumerateViewConfigurationViews(B.instance, B.systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, n, &n,
                                           views.data()));
  if (n != 2) {
    Log.error("Expected 2 stereo views, got {}", n);
    return false;
  }
  // A fixed eye resolution from the host (AuroraConfig::xrEyeScale) turns
  // dynamic resolution off; AURORA_XR_EYE_SCALE and AURORA_XR_DYNRES override.
  const bool hostFixed = g_config.xrEyeScale > 0.f;
  const float scale =
      std::clamp(env_float("AURORA_XR_EYE_SCALE", hostFixed ? g_config.xrEyeScale : 1.f), 0.25f, 2.f);
  const auto even = [](float v) { return (static_cast<uint32_t>(v + 0.5f) + 1u) & ~1u; };
  auto& dr = g_dynres;
  dr.recWidth = even(static_cast<float>(views[0].recommendedImageRectWidth) * scale);
  dr.recHeight = even(static_cast<float>(views[0].recommendedImageRectHeight) * scale);
  // Multiview only: the side-by-side path packs both eyes in one image.
#ifdef __ANDROID__
  dr.on = g_multiview && env_flag("AURORA_XR_DYNRES", !hostFixed); // measured there
#else
  dr.on = g_multiview && env_flag("AURORA_XR_DYNRES", false);
#endif
  if (dr.on) {
    // With SpaceWarp the floor is 0.8 (only while it's on, update_dynamic_
    // resolution): a frame it can't fit loses it for the fight, so a little
    // softness is the cheaper trade.
    dr.minScale = std::clamp(env_float("AURORA_XR_DYNRES_MIN", B.appSw ? 0.8f : 1.f), 0.5f, 1.f);
    dr.wantMarginNs = std::clamp(env_float("AURORA_XR_DYNRES_MARGIN_MS", 3.f), 0.f, 8.f) * 1.0e6;
    dr.maxScale = std::clamp(env_float("AURORA_XR_DYNRES_MAX", 1.3f), 1.f, 1.6f);
    // The target follows the display and the compositor (DynamicResolution);
    // AURORA_XR_DYNRES_TARGET_MS pins it instead.
    dr.fixedTarget = std::getenv("AURORA_XR_DYNRES_TARGET_MS") != nullptr;
    dr.targetNs = std::clamp(env_float("AURORA_XR_DYNRES_TARGET_MS", 13.f), 3.f, 16.f) * 1.0e6;
    dr.headroomNs = std::clamp(env_float("AURORA_XR_DYNRES_HEADROOM_MS", 1.2f), 0.f, 8.f) * 1.0e6;
    dr.bandNs = std::clamp(env_float("AURORA_XR_DYNRES_BAND_MS", 0.5f), 0.f, 4.f) * 1.0e6;
    dr.random = env_flag("AURORA_XR_DYNRES_RANDOM", false);
    dr.level = std::max(static_cast<int>((1.f - dr.minScale) / kDynresStep + 0.5f), 0); // start at 1.0
    dr.scale = std::min(dr.minScale + kDynresStep * static_cast<float>(dr.level), dr.maxScale);
  }
  const float alloc = dr.on ? dr.maxScale : 1.f;
  const uint32_t eyeW = even(static_cast<float>(dr.recWidth) * alloc);
  const uint32_t eyeH = even(static_cast<float>(dr.recHeight) * alloc);
  if (dr.on)
    Log.info("Dynamic resolution: {:.2f}-{:.2f} of {}x{} per eye, frame GPU target {} +- {:.1f} ms", dr.minScale,
             dr.maxScale, dr.recWidth, dr.recHeight,
             dr.fixedTarget ? fmt::format("{:.1f}", dr.targetNs / 1.0e6)
                            : fmt::format("from the display, {:.1f} ms headroom", dr.headroomNs / 1.0e6),
             dr.bandNs / 1.0e6);
  else
    Log.info("Eye resolution: {}x{} per eye ({:.2f} of the recommended), dynamic resolution off", eyeW, eyeH,
             scale);
  g_streams[kStereo].width = g_multiview ? eyeW : eyeW * 2;
  g_streams[kStereo].height = eyeH;
  g_streams[kStereo].layers = g_multiview ? 2 : 1;
  if (B.appSw && !g_multiview) {
    Log.warn("Application SpaceWarp off: it needs the multiview eyes");
    B.appSw = false;
  }
  if (B.appSw) {
    for (int i : {kMotion, kMotionDepth}) {
      auto& st = g_streams[i];
      st.width = B.mvWidth;
      st.height = B.mvHeight;
      st.layers = 2;
      st.ownFormat = true;
    }
    g_streams[kMotion].name = "motion";
    g_streams[kMotion].format = VK_FORMAT_R16G16B16A16_SFLOAT;
    g_streams[kMotion].swapFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    g_streams[kMotionDepth].name = "motion depth";
    g_streams[kMotionDepth].depth = true; // its format: create_session, from the runtime's
  }
  return true;
}

bool create_session() {
  XrGraphicsBindingVulkan2KHR gb{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
  gb.instance = B.vkInstance;
  gb.physicalDevice = B.phys;
  gb.device = B.dev;
  gb.queueFamilyIndex = B.queueFamily;
  gb.queueIndex = B.queueIndex; // Dawn's spare queue; Dawn submits on 0
  XrSessionCreateInfo sci{XR_TYPE_SESSION_CREATE_INFO};
  sci.next = &gb;
  sci.systemId = B.systemId;
  XR_TRY(xrCreateSession(B.instance, &sci, &B.session));

  XrReferenceSpaceCreateInfo rsci{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
  rsci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
  rsci.poseInReferenceSpace.orientation.w = 1.f;
  XR_TRY(xrCreateReferenceSpace(B.session, &rsci, &B.space));
  // The floor under LOCAL's origin: only a surface for the stage's shadow
  // when nothing better is known, never where anything is placed.
  if (B.hasLocalFloorExt) {
    rsci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL_FLOOR_EXT;
    if (XR_FAILED(xrCreateReferenceSpace(B.session, &rsci, &B.floorSpace)))
      B.floorSpace = XR_NULL_HANDLE;
  }

  // The game's frames are already sRGB-encoded bytes. Dawn draws them through
  // a UNORM view of a mutable-format *_SRGB swapchain image, which stores the
  // bytes unchanged, and the runtime decodes them correctly; an sRGB view
  // would re-encode them and wash them out.
  uint32_t n = 0;
  XR_TRY(xrEnumerateSwapchainFormats(B.session, 0, &n, nullptr));
  std::vector<int64_t> fmts(n);
  XR_TRY(xrEnumerateSwapchainFormats(B.session, n, &n, fmts.data()));
  for (int64_t want : {VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM}) {
    if (std::find(fmts.begin(), fmts.end(), want) != fmts.end()) {
      B.swapFormat = want;
      break;
    }
  }
  if (!B.swapFormat) {
    Log.error("Runtime offers no R8G8B8A8 swapchain format");
    return false;
  }
  for (auto& st : g_streams)
    if (!st.ownFormat)
      st.swapFormat = B.swapFormat;
  if (B.appSw) {
    // SpaceWarp's depth: 32-bit float if offered, else 24-bit.
    auto& d = g_streams[kMotionDepth];
    for (int64_t f : {int64_t{VK_FORMAT_D32_SFLOAT}, int64_t{VK_FORMAT_D24_UNORM_S8_UINT}})
      if (std::find(fmts.begin(), fmts.end(), f) != fmts.end()) {
        d.format = static_cast<VkFormat>(f);
        d.swapFormat = f;
        break;
      }
    if (d.swapFormat == 0 ||
        std::find(fmts.begin(), fmts.end(), int64_t{VK_FORMAT_R16G16B16A16_SFLOAT}) == fmts.end()) {
      Log.warn("SpaceWarp: no motion vector or depth swapchain format; SpaceWarp off");
      B.appSw = false;
      for (int i : {kMotion, kMotionDepth})
        g_streams[i].width = 0;
    }
    if (B.appSw)
      Log.info("SpaceWarp depth: {}", d.format == VK_FORMAT_D32_SFLOAT ? "D32_SFLOAT" : "D24_UNORM_S8_UINT");
  }
  // The 3D image takes the framebuffer's channel order when the eyes render
  // straight into it (single-sample BGRA8 framebuffers, e.g. desktop NVIDIA).
  auto& stereo = g_streams[kStereo];
  const auto offered = [&](int64_t f) { return std::find(fmts.begin(), fmts.end(), f) != fmts.end(); };
#ifndef __ANDROID__
  if (g_sceneSamples == 1 && g_sceneFormat == wgpu::TextureFormat::BGRA8Unorm && offered(VK_FORMAT_B8G8R8A8_SRGB)) {
    stereo.format = VK_FORMAT_B8G8R8A8_UNORM;
    stereo.swapFormat = VK_FORMAT_B8G8R8A8_SRGB;
  }
#endif
  (void)offered;

  if (B.createPassthrough && B.createPassthroughLayer && env_flag("AURORA_XR_PASSTHROUGH", true)) {
    XrPassthroughCreateInfoFB pci{XR_TYPE_PASSTHROUGH_CREATE_INFO_FB};
    pci.flags = XR_PASSTHROUGH_IS_RUNNING_AT_CREATION_BIT_FB;
    if (XR_SUCCEEDED(B.createPassthrough(B.session, &pci, &B.passthrough))) {
      XrPassthroughLayerCreateInfoFB lci{XR_TYPE_PASSTHROUGH_LAYER_CREATE_INFO_FB};
      lci.passthrough = B.passthrough;
      lci.flags = XR_PASSTHROUGH_IS_RUNNING_AT_CREATION_BIT_FB;
      lci.purpose = XR_PASSTHROUGH_LAYER_PURPOSE_RECONSTRUCTION_FB;
      if (XR_FAILED(B.createPassthroughLayer(B.session, &lci, &B.passthroughLayer)))
        B.passthroughLayer = XR_NULL_HANDLE;
      B.passthroughRunning = true;
    } else {
      B.passthrough = XR_NULL_HANDLE;
    }
  }
  Log.info("Swapchains: screen {}x{}, 3D {}x{} (two {}x{} eyes, format {}), HUD {}x{} (format {}), passthrough {}",
           g_streams[kScreen].width, g_streams[kScreen].height, g_streams[kStereo].width, g_streams[kStereo].height,
           g_multiview ? g_streams[kStereo].width : g_streams[kStereo].width / 2, g_streams[kStereo].height,
           g_streams[kStereo].swapFormat,
           g_streams[kHud].width, g_streams[kHud].height, B.swapFormat, B.passthroughLayer ? "on" : "off");
  return true;
}

bool create_dump_buffer(Stream& st) {
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = static_cast<VkDeviceSize>(st.width) * st.height * st.layers * 4;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VK_TRY(vkCreateBuffer(B.dev, &bci, nullptr, &st.dumpBuf));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(B.dev, st.dumpBuf, &mr);
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = mr.size;
  mai.memoryTypeIndex =
      find_memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_TRY(vkAllocateMemory(B.dev, &mai, nullptr, &st.dumpMem));
  VK_TRY(vkBindBufferMemory(B.dev, st.dumpBuf, st.dumpMem, 0));
  VK_TRY(vkMapMemory(B.dev, st.dumpMem, 0, bci.size, 0, &st.dumpPtr));
  return true;
}

void write_dump(const Stream& st) {
  const auto* px = static_cast<const uint8_t*>(st.dumpPtr);
  const std::string base = B.dumpDir + "/xr_" + st.name;
  if (FILE* f = std::fopen((base + ".ppm").c_str(), "wb")) {
    std::fprintf(f, "P6\n%u %u\n255\n", st.width, st.height * st.layers);
    for (size_t i = 0; i < static_cast<size_t>(st.width) * st.height * st.layers; ++i)
      std::fwrite(px + i * 4, 1, 3, f);
    std::fclose(f);
  }
  if (FILE* f = std::fopen((base + "_alpha.pgm").c_str(), "wb")) {
    std::fprintf(f, "P5\n%u %u\n255\n", st.width, st.height * st.layers);
    for (size_t i = 0; i < static_cast<size_t>(st.width) * st.height * st.layers; ++i)
      std::fputc(px[i * 4 + 3], f);
    std::fclose(f);
  }
  Log.info("Dumped {} to {}.ppm", st.name, base);
}

// ---------------------------------------------------------------- XR thread: controller input
//
// Touch controllers as a GameCube pad (port 1):
//   left stick -> control stick     right stick -> C-stick
//   A / B (right) -> A / B          X / Y (left) -> X / Y
//   left / right trigger -> analog L / R (digital past 90%)
//   either grip -> Z                left menu button -> Start
// While a fight is paused the grips grab the arena instead (Z would retry the
// match in some modes), and each controller shows a laser.

bool create_action(XrAction& action, const char* name, const char* localized, XrActionType type) {
  XrActionCreateInfo ai{XR_TYPE_ACTION_CREATE_INFO};
  std::strncpy(ai.actionName, name, XR_MAX_ACTION_NAME_SIZE - 1);
  std::strncpy(ai.localizedActionName, localized, XR_MAX_LOCALIZED_ACTION_NAME_SIZE - 1);
  ai.actionType = type;
  XR_TRY(xrCreateAction(B.actionSet, &ai, &action));
  return true;
}

XrPath path(const char* str) {
  XrPath p = XR_NULL_PATH;
  xrStringToPath(B.instance, str, &p);
  return p;
}

bool suggest(const char* profile, std::initializer_list<std::pair<XrAction, const char*>> bindings) {
  std::vector<XrActionSuggestedBinding> sb;
  for (const auto& [action, binding] : bindings)
    sb.push_back({action, path(binding)});
  XrInteractionProfileSuggestedBinding isb{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
  isb.interactionProfile = path(profile);
  isb.suggestedBindings = sb.data();
  isb.countSuggestedBindings = static_cast<uint32_t>(sb.size());
  const XrResult r = xrSuggestInteractionProfileBindings(B.instance, &isb);
  if (XR_FAILED(r))
    Log.warn("Bindings for {} rejected: XrResult {}", profile, static_cast<int>(r));
  return XR_SUCCEEDED(r);
}

bool create_input() {
  XrActionSetCreateInfo asi{XR_TYPE_ACTION_SET_CREATE_INFO};
  std::strncpy(asi.actionSetName, "gamecube", XR_MAX_ACTION_SET_NAME_SIZE - 1);
  std::strncpy(asi.localizedActionSetName, "GameCube controller", XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE - 1);
  XR_TRY(xrCreateActionSet(B.instance, &asi, &B.actionSet));
  if (!create_action(B.stickMain, "control_stick", "Control stick", XR_ACTION_TYPE_VECTOR2F_INPUT) ||
      !create_action(B.stickC, "c_stick", "C-stick", XR_ACTION_TYPE_VECTOR2F_INPUT) ||
      !create_action(B.btnA, "a", "A", XR_ACTION_TYPE_BOOLEAN_INPUT) ||
      !create_action(B.btnB, "b", "B", XR_ACTION_TYPE_BOOLEAN_INPUT) ||
      !create_action(B.btnX, "x", "X", XR_ACTION_TYPE_BOOLEAN_INPUT) ||
      !create_action(B.btnY, "y", "Y", XR_ACTION_TYPE_BOOLEAN_INPUT) ||
      !create_action(B.btnZ, "z", "Z", XR_ACTION_TYPE_BOOLEAN_INPUT) ||
      !create_action(B.btnStart, "start", "Start", XR_ACTION_TYPE_BOOLEAN_INPUT) ||
      !create_action(B.trigL, "l", "L", XR_ACTION_TYPE_FLOAT_INPUT) ||
      !create_action(B.trigR, "r", "R", XR_ACTION_TYPE_FLOAT_INPUT) ||
      !create_action(B.aim[0], "aim_left", "Left pointer", XR_ACTION_TYPE_POSE_INPUT) ||
      !create_action(B.aim[1], "aim_right", "Right pointer", XR_ACTION_TYPE_POSE_INPUT) ||
      !create_action(B.grab[0], "grab_left", "Left grab", XR_ACTION_TYPE_FLOAT_INPUT) ||
      !create_action(B.grab[1], "grab_right", "Right grab", XR_ACTION_TYPE_FLOAT_INPUT))
    return false;

  const bool touch = suggest("/interaction_profiles/oculus/touch_controller",
                             {
                                 {B.stickMain, "/user/hand/left/input/thumbstick"},
                                 {B.stickC, "/user/hand/right/input/thumbstick"},
                                 {B.btnA, "/user/hand/right/input/a/click"},
                                 {B.btnB, "/user/hand/right/input/b/click"},
                                 {B.btnX, "/user/hand/left/input/x/click"},
                                 {B.btnY, "/user/hand/left/input/y/click"},
                                 {B.btnZ, "/user/hand/left/input/squeeze/value"},
                                 {B.btnZ, "/user/hand/right/input/squeeze/value"},
                                 {B.btnStart, "/user/hand/left/input/menu/click"},
                                 {B.trigL, "/user/hand/left/input/trigger/value"},
                                 {B.trigR, "/user/hand/right/input/trigger/value"},
                                 {B.aim[0], "/user/hand/left/input/aim/pose"},
                                 {B.aim[1], "/user/hand/right/input/aim/pose"},
                                 {B.grab[0], "/user/hand/left/input/squeeze/value"},
                                 {B.grab[1], "/user/hand/right/input/squeeze/value"},
                             });
  // Pointing and grabbing only, no pad buttons: the Quest reports bare
  // hands as this profile, select being a pinch, and a pinch must grab,
  // not press A or B. (Monado's keyboard/mouse controllers come through it
  // too; the desktop keyboard plays the pad there.)
  suggest("/interaction_profiles/khr/simple_controller", {
                                                             {B.aim[0], "/user/hand/left/input/aim/pose"},
                                                             {B.aim[1], "/user/hand/right/input/aim/pose"},
                                                             {B.grab[0], "/user/hand/left/input/select/click"},
                                                             {B.grab[1], "/user/hand/right/input/select/click"},
                                                         });
  XrSessionActionSetsAttachInfo attach{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
  attach.countActionSets = 1;
  attach.actionSets = &B.actionSet;
  XR_TRY(xrAttachSessionActionSets(B.session, &attach));
  for (int h = 0; h < 2; ++h) {
    XrActionSpaceCreateInfo asci{XR_TYPE_ACTION_SPACE_CREATE_INFO};
    asci.action = B.aim[h];
    asci.poseInActionSpace.orientation.w = 1.f;
    XR_TRY(xrCreateActionSpace(B.session, &asci, &B.aimSpace[h]));
  }
  Log.info("Controller input ready ({})", touch ? "Touch controller bindings" : "simple controller only");
  return true;
}

// One tracker per hand, when the runtime and headset support it.
bool create_hand_trackers() {
  if (!B.hasHandTrackingExt || !B.createHandTracker || !B.locateHandJoints)
    return false;
  XrSystemHandTrackingPropertiesEXT ht{XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT};
  XrSystemProperties sp{XR_TYPE_SYSTEM_PROPERTIES};
  sp.next = &ht;
  XR_TRY(xrGetSystemProperties(B.instance, B.systemId, &sp));
  if (!ht.supportsHandTracking)
    return false;
  for (int h = 0; h < 2; ++h) {
    XrHandTrackerCreateInfoEXT ci{XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT};
    ci.hand = h == 0 ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT;
    ci.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;
    XR_TRY(B.createHandTracker(B.session, &ci, &B.handTrackers[h]));
  }
  Log.info("Hand tracking ready");
  return true;
}

bool action_bool(XrAction a) {
  XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
  gi.action = a;
  XrActionStateBoolean st{XR_TYPE_ACTION_STATE_BOOLEAN};
  return XR_SUCCEEDED(xrGetActionStateBoolean(B.session, &gi, &st)) && st.isActive && st.currentState;
}

float action_float(XrAction a) {
  XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
  gi.action = a;
  XrActionStateFloat st{XR_TYPE_ACTION_STATE_FLOAT};
  return XR_SUCCEEDED(xrGetActionStateFloat(B.session, &gi, &st)) && st.isActive ? st.currentState : 0.f;
}

XrVector2f action_vec2(XrAction a) {
  XrActionStateGetInfo gi{XR_TYPE_ACTION_STATE_GET_INFO};
  gi.action = a;
  XrActionStateVector2f st{XR_TYPE_ACTION_STATE_VECTOR2F};
  if (XR_SUCCEEDED(xrGetActionStateVector2f(B.session, &gi, &st)) && st.isActive)
    return st.currentState;
  return {0.f, 0.f};
}

s8 to_axis(float v) { return static_cast<s8>(std::lround(std::clamp(v, -1.f, 1.f) * 127.f)); }

void update_input() {
  if (!B.actionSet || !B.focused) {
    std::lock_guard lock{g_padMutex};
    g_padValid = false;
    return;
  }
  XrActiveActionSet active{B.actionSet, XR_NULL_PATH};
  XrActionsSyncInfo sync{XR_TYPE_ACTIONS_SYNC_INFO};
  sync.countActiveActionSets = 1;
  sync.activeActionSets = &active;
  if (XR_FAILED(xrSyncActions(B.session, &sync)))
    return;
  PADStatus pad{};
  const XrVector2f main = action_vec2(B.stickMain);
  const XrVector2f c = action_vec2(B.stickC);
  B.stickY = {main.y, c.y};
  B.stickX = {main.x, c.x};
  B.backButton = action_bool(B.btnB) || action_bool(B.btnY);
  if (g_boardPlacing) {
    std::lock_guard lock{g_padMutex};
    g_pad = {};
    g_padValid = true;
    return;
  }
  // A controller dragging the screen reels it in and out with its stick
  // instead of steering the menu.
  if (B.reelHand != 0) {
    pad.stickX = to_axis(main.x);
    pad.stickY = to_axis(main.y);
  }
  if (B.reelHand != 1) {
    pad.substickX = to_axis(c.x);
    pad.substickY = to_axis(c.y);
  }
  const float l = B.triggersClick ? 0.f : action_float(B.trigL);
  const float r = B.triggersClick ? 0.f : action_float(B.trigR);
  pad.triggerLeft = static_cast<u8>(std::lround(std::clamp(l, 0.f, 1.f) * 255.f));
  pad.triggerRight = static_cast<u8>(std::lround(std::clamp(r, 0.f, 1.f) * 255.f));
  u16 buttons = 0;
  buttons |= action_bool(B.btnA) ? PAD_BUTTON_A : 0;
  buttons |= action_bool(B.btnB) ? PAD_BUTTON_B : 0;
  buttons |= action_bool(B.btnX) ? PAD_BUTTON_X : 0;
  buttons |= action_bool(B.btnY) ? PAD_BUTTON_Y : 0;
  buttons |= !B.pointerMode && action_bool(B.btnZ) ? PAD_TRIGGER_Z : 0;
  buttons |= action_bool(B.btnStart) ? PAD_BUTTON_START : 0;
  buttons |= l > 0.9f ? PAD_TRIGGER_L : 0;
  buttons |= r > 0.9f ? PAD_TRIGGER_R : 0;
  pad.button = buttons;
  std::lock_guard lock{g_padMutex};
  g_pad = pad;
  g_padValid = true;
}

// ---------------------------------------------------------------- XR thread: per frame

// The slot's last release has finished on the GPU: its command buffer and
// semaphores can be reused, and a dump it recorded can be written.
void retire_slot(Stream& st, Slot& s, int index) {
  if (!s.inFlight)
    return;
  vkWaitForFences(B.dev, 1, &s.fence, VK_TRUE, UINT64_MAX);
  vkResetFences(B.dev, 1, &s.fence);
  s.inFlight = false;
  for (VkSemaphore sem : s.waitSems)
    vkDestroySemaphore(B.dev, sem, nullptr);
  s.waitSems.clear();
  if (st.dumpSlot == index) {
    st.dumpSlot = -1;
    st.dumped = true;
    write_dump(st);
  }
}

#ifdef __ANDROID__
// A systrace counter (ATrace_setCounter: Android 29, past the app's minimum).
void trace_counter(const char* name, int64_t value) {
  using SetCounter = void (*)(const char*, int64_t);
  static const auto set = reinterpret_cast<SetCounter>(dlsym(RTLD_DEFAULT, "ATrace_setCounter"));
  if (set != nullptr)
    set(name, value);
}
struct TraceSection {
  explicit TraceSection(const char* name) { ATrace_beginSection(name); }
  ~TraceSection() { ATrace_endSection(); }
};
#else
void trace_counter(const char*, int64_t) {}
struct TraceSection {
  explicit TraceSection(const char*) {}
};
#endif

// Keeps one waited-on image ready for the render worker. OpenXR allows one
// waited image per swapchain until it is released, so the next is waited on
// only once the last went back. The wait is short: an image the compositor
// still holds is waited on again next display frame.
bool acquire_ahead(Stream& st, int64_t waitNs = 2'000'000) {
  int pending = -1;
  bool haveWaited = false;
  size_t held = 0;
  {
    std::lock_guard lock{g_mutex};
    for (size_t i = 0; i < st.slots.size(); ++i) {
      const SlotState state = st.slots[i].state;
      haveWaited |= state == SlotState::Free || state == SlotState::Rendering || state == SlotState::Ready;
      if (state == SlotState::Acquired)
        pending = static_cast<int>(i);
      if (state != SlotState::Released)
        ++held;
    }
  }
  // Acquiring needn't wait for the last image's release, only waiting on
  // it does; deferred eyes draw into it in between.
  // SpaceWarp's images go with the 3D one: waiting on them blocked the
  // render worker until the next loop, every frame.
  const bool preAcquire = g_deferEyes && !st.staged &&
                          (&st == &g_streams[kStereo] || &st == &g_streams[kMotion] ||
                           &st == &g_streams[kMotionDepth]);
  if (pending < 0 && held < st.slots.size() && (!haveWaited || preAcquire)) {
    uint32_t index = 0;
    XR_TRY(xrAcquireSwapchainImage(st.swapchain, nullptr, &index));
    std::lock_guard lock{g_mutex};
    st.slots[index].state = SlotState::Acquired;
    st.slots[index].acquireSeq = ++st.acquireSeq;
    st.slots[index].claimed = false;
    pending = static_cast<int>(index);
    g_slotFreeCv.notify_all();
  }
  if (haveWaited || pending < 0)
    return true;
  XrSwapchainImageWaitInfo wi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
  wi.timeout = waitNs; // 2 ms but for a staged frame due (release_staged)
  TraceSection trace{&st == &g_streams[kStereo] ? "Wait 3D image" : "Wait image"};
  const XrResult r = xrWaitSwapchainImage(st.swapchain, &wi);
  if (r == XR_SUCCESS && &st == &g_streams[kStereo])
    trace_counter("XR 3D image waited", static_cast<int64_t>(st.acquireSeq & 1));
  if (r == XR_TIMEOUT_EXPIRED)
    return true;
  XR_TRY(r);
  retire_slot(st, st.slots[pending], pending);
  {
    std::lock_guard lock{g_mutex};
    auto& sl = st.slots[pending];
    sl.state = sl.claimed ? SlotState::Rendering : SlotState::Free;
    sl.claimed = false;
  }
  g_slotFreeCv.notify_all();
  return true;
}

// Fixed-latency presentation (XR thread). Under lock-step pacing a game
// frame starts on a tick and its 3D and HUD images are released `latency`
// display frames after that tick, not as soon as they're ready: released as
// soon as ready, images finishing right around a frame boundary alternated
// between being shown 3 display frames and 1 instead of 2 and 2 (about 20
// such pairs a second on Battlefield). `latency` is the smallest number of
// display frames by which 97% of recent 3D images were ready, at most two
// past a game frame's display frames; it rises at once and falls after four
// steady windows. On Android "ready" is the image's GPU work being done (the
// pacing probe), not its submit: the runtime's xrBeginFrame waits for the
// work of images already handed over, so an image released unfinished
// stalled the XR loop and display frames went stale. Four-CPU Mute City
// finishes about 26 ms after its tick, past latency 3's 25: at 4 it went
// from 2.7 stale frames a second to 0.1, every image on its two display
// frames, for one display frame (8 ms) more latency. On with dynamic resolution (see release_ready);
// AURORA_XR_FIXED_LATENCY=0 turns it off, =<n> forces it on at n.
struct Presentation {
  int forced = -1; // AURORA_XR_FIXED_LATENCY
  int latency = 0;
  std::array<uint8_t, 120> samples{};
  size_t count = 0, next = 0;
  int sinceUpdate = 0, steadyLower = 0;
  uint64_t held = 0; // images held for their slot (logged)
  // Frames finishing after their release at the current latency, over the
  // last 120 (note_lateness): too many, and it's left and kept out of for a
  // while, longer each time.
  std::array<bool, 120> late{};
  int lateLatency = 0, lateCount = 0, lateNext = 0, lateFrames = 0;
  std::array<int64_t, 16> bannedUntilNs{};
  std::array<int, 16> bans{};
  uint64_t raisedForLate = 0; // since the last stats (logged)
} g_present;

// Frames whose eyes went early, and of those, drawn again at the frame's end
// (world draws after the split); logged with the frame stats.
std::atomic<uint32_t> g_earlyEyes{0}, g_earlyResumed{0};
// SpaceWarp: 3D frames dropped for starting too late (render_eyes), logged.
std::atomic<uint32_t> g_appSwDropped{0};
// ... frames submitted although late (tweak late), the next one skipped for it,
// and motion passes that followed a tick that never went out (tweak gap).
std::atomic<uint32_t> g_appSwLate{0}, g_appSwPlannedSkip{0}, g_appSwGapCut{0};
// Staged frames retired unshown, a later one being due too (release_staged),
// and hand-overs that found no swapchain image even after waiting for one.
std::atomic<uint32_t> g_stagedStale{0}, g_stagedNoImage{0};
std::array<std::atomic<uint32_t>, 5> g_noImageBy{};
// XR loops that handed the 3D stream no new image (logged), by why: [0] no
// frame in flight (the game or its recording is late), [1] the oldest still
// being drawn (GPU), [2] finished but held for its slot, [3] finished and due
// but no swapchain image / lost to a newer frame. [4] loops that did release.
std::array<uint32_t, 5> g_missWhy{};
// Loops that submitted SpaceWarp's images from another game tick than the 3D
// image's, per image: [0] motion, [1] depth, [2] HUD (hud is only counted). Logged.
std::array<uint32_t, 3> g_tickMismatch{};
// Live presentation modes (XR thread), from the environment and then, if
// AURORA_XR_TWEAK names a file, its "lead=N set=N cutout=N trans=N epoch=N gap=N late=N eyesort=N bindless=N" (read twice a
// second), so a build can be compared mid-fight:
//   lead: 1 poses predicted for the frame's display time (publish_views).
//   set: 1 a tick's 3D, motion and depth go out together (compute_gate).
//   cutout, trans: 1 alpha-tested / translucent draws in the motion pass.
//   epoch: 1 the previous-pose state SpaceWarp's vectors pair with advances
//     only when a frame's eyes go out (a dropped frame no longer becomes the
//     epoch the next one pairs against, while the runtime holds an older image).
//   gap: 1 after a tick that never went out, the motion pass pairs no draw
//     with its recorded previous matrices (render_eyes).
//   late: 1 a frame over AURORA_XR_APPSW_DROP_MS late is drawn and submitted
//     anyway; the next one, if it is late too, is skipped (render_eyes).
//   eyesort: 1 the eye replay regroups reorderable opaque draws for batching
//     (gx pipeline.cpp; AURORA_XR_EYE_SORT, on by default).
//   bindless: 1 eye batches join across textures through the resource table
//     (AURORA_XR_BINDLESS, on by default; only where the device offers it).
struct Tweaks {
  bool lead = true;
  bool set = true;
  bool cutout = true;
  bool trans = true;
  bool epoch = true;
  bool gap = true;
  bool late = false;
  bool eyesort = true;
  bool bindless = true;
} g_tweak;
// The device offers Dawn's resource table (bindless eye batching can run).
bool g_bindlessOk = false;
bool appsw_on() { return B.appSw; }
// 3D frames done over 1 ms after the slot they're released in, so far
// (pacing probe; dynamic resolution's brake).
std::atomic<uint32_t> g_lateFrames{0};
// Render worker: the longest wait for a swapchain image, and waits over 2 ms,
// per stream (read racily by the frame stats).
std::array<int64_t, kStreamCount> g_acquireWaitMaxNs{};
std::array<uint32_t, kStreamCount> g_acquireWaitsOver2{};

// Pacing probe (Android): where each 3D frame's time goes after its tick,
// and when its GPU work finishes against the display time it's due for. A
// watcher thread polls a dup of Dawn's sync fd, so the finish time is exact.
// Logged with the frame stats: the game waking on the tick, its recording
// starting and ending, the render worker starting the eye replay, the
// submit, the GPU finishing, and the margin before the image's display
// time (tick + latency).
struct PaceProbe {
  struct Display {
    uint64_t frame = 0;
    int64_t waitNs = 0, displayNs = 0, wakeNs = 0; // xrWaitFrame's return; its predicted display; game woken
  };
  enum Mark { Wake, RecordStart, WorldEnd, GameDone, RecordEnd, RenderStart, Submit, Done, Margin, Marks };
  std::mutex mutex;
  std::condition_variable cv;
  std::array<Display, 256> displays{};
  struct Pending {
    uint64_t tickFrame;
    int latency;
    int64_t slotWaitNs;
    std::array<int64_t, Done> ns; // by Mark, up to the submit
    int fd;
  };
  std::deque<Pending> pending;
  std::array<std::vector<float>, Marks> ms; // since the last log, by Mark
  std::array<int64_t, 5> next{};            // render worker: record start, world end, game done, record end, render start
  int64_t slotWaitNs = 0;                   // render worker: this frame's wait for a 3D image
  std::vector<uint8_t> doneLatency;          // display frames from tick to GPU done, for g_present
  std::vector<std::pair<uint8_t, bool>> lateness; // latency drawn at, done over 1 ms after its release
  std::vector<float> idealMargins;           // ns from GPU done to the ideal latency's slot, for g_dynres
  std::vector<std::string> spikes;           // frames done after their slot, by hop (logged, a few)
  uint32_t lateFrames = 0;                   // done over 1 ms after their slot (logged)
  bool started = false;
} g_pace;


int64_t monotonic_ns() { return std::chrono::steady_clock::now().time_since_epoch().count(); }

void pace_display(uint64_t frame, XrTime displayTime) {
  std::lock_guard lock{g_pace.mutex};
  g_pace.displays[frame % g_pace.displays.size()] = {frame, monotonic_ns(), static_cast<int64_t>(displayTime), 0};
}

void pace_wake(uint64_t tickFrame) {
  std::lock_guard lock{g_pace.mutex};
  auto& d = g_pace.displays[tickFrame % g_pace.displays.size()];
  if (d.frame == tickFrame)
    d.wakeNs = monotonic_ns();
}

#ifdef __ANDROID__
void pace_watch() {
  for (;;) {
    PaceProbe::Pending p;
    {
      std::unique_lock lock{g_pace.mutex};
      g_pace.cv.wait(lock, [] { return !g_pace.pending.empty(); });
      p = g_pace.pending.front();
      g_pace.pending.pop_front();
    }
    pollfd pfd{.fd = p.fd, .events = POLLIN};
    const bool done = poll(&pfd, 1, 200) > 0;
    const int64_t doneNs = monotonic_ns();
    close(p.fd);
    if (!done)
      continue;
    std::lock_guard lock{g_pace.mutex};
    const auto& tick = g_pace.displays[p.tickFrame % g_pace.displays.size()];
    if (tick.frame != p.tickFrame || g_pace.ms[PaceProbe::Done].size() >= 4096)
      continue;
    p.ns[PaceProbe::Wake] = tick.wakeNs;
    for (int m = 0; m < PaceProbe::Done; ++m)
      if (p.ns[m] != 0)
        g_pace.ms[m].push_back(static_cast<float>(p.ns[m] - tick.waitNs) / 1e6f);
    g_pace.ms[PaceProbe::Done].push_back(static_cast<float>(doneNs - tick.waitNs) / 1e6f);
    trace_counter("XR 3D done after tick (us)", (doneNs - tick.waitNs) / 1000);
    // The first display frame whose loop could release it finished (half a
    // millisecond of slack): fixed-latency presentation's sample. Less the
    // render worker's wait for a free image, which the latency itself
    // causes: counted, it kept Battlefield at latency 4, where holding
    // images starved the render worker and 3D ran at 30 a second.
    const float hz = g_displayHz > 0.f ? g_displayHz.load() : 120.f;
    const double frames =
        std::ceil(static_cast<double>(doneNs - p.slotWaitNs + 500000 - tick.waitNs) * hz / 1e9);
    if (g_pace.doneLatency.size() < 256)
      g_pace.doneLatency.push_back(static_cast<uint8_t>(std::clamp(frames, 0.0, 15.0)));
    // Dynamic resolution's margin: to the slot of a game frame plus one
    // display frame, whatever the latency is now.
    const int ideal = std::max(g_displayPerGameFrame.load(), 1) + 1;
    const double slot = static_cast<double>(tick.waitNs) + ideal * 1e9 / hz;
    // No more than the margin to the slot it's released in: above the
    // ideal latency, the wait for an image left out above may not go at
    // the ideal one (see note_lateness).
    const double releaseSlot = static_cast<double>(tick.waitNs) + p.latency * 1e9 / hz;
    if (g_pace.idealMargins.size() < 256)
      g_pace.idealMargins.push_back(static_cast<float>(
          std::min(slot - static_cast<double>(doneNs - p.slotWaitNs), releaseSlot - static_cast<double>(doneNs))));
    const bool late = static_cast<double>(doneNs) > releaseSlot + 1e6;
    g_pace.lateFrames += late ? 1 : 0;
    if (late)
      g_lateFrames.fetch_add(1, std::memory_order_relaxed);
    if (g_pace.lateness.size() < 256)
      g_pace.lateness.emplace_back(static_cast<uint8_t>(p.latency), late);
    // A spike: done after the slot it's released in. Each hop, ms after the tick.
    if (static_cast<double>(doneNs) > releaseSlot && g_pace.spikes.size() < 8) {
      const auto ms = [&](int64_t ns) { return ns != 0 ? static_cast<float>(ns - tick.waitNs) / 1e6f : -1.f; };
      g_pace.spikes.push_back(fmt::format("[world {:.1f} game {:.1f} recorded {:.1f} replaying {:.1f} submitted {:.1f} "
                                          "done {:.1f} slot {:.1f} image wait {:.1f}]",
                                          ms(p.ns[PaceProbe::WorldEnd]), ms(p.ns[PaceProbe::GameDone]),
                                          ms(p.ns[PaceProbe::RecordEnd]), ms(p.ns[PaceProbe::RenderStart]),
                                          ms(p.ns[PaceProbe::Submit]), ms(doneNs),
                                          (releaseSlot - static_cast<double>(tick.waitNs)) / 1e6,
                                          p.slotWaitNs / 1e6));
    }
    // The display time of the frame it's released for.
    const auto& due = g_pace.displays[(p.tickFrame + p.latency) % g_pace.displays.size()];
    if (due.frame == p.tickFrame + p.latency)
      g_pace.ms[PaceProbe::Margin].push_back(static_cast<float>(due.displayNs - doneNs) / 1e6f);
  }
}
#endif

// When the XR thread's loop for `frame` began (0: not recorded).
int64_t pace_wait_ns(uint64_t frame) {
  std::lock_guard lock{g_pace.mutex};
  const auto& d = g_pace.displays[frame % g_pace.displays.size()];
  return d.frame == frame ? d.waitNs : 0;
}

// Render worker: the frame about to be replayed for the eyes.
void pace_frame(int64_t recordStartNs, int64_t worldEndNs, int64_t gameDoneNs, int64_t recordEndNs) {
  g_pace.next = {recordStartNs, worldEndNs != 0 ? worldEndNs : recordEndNs, gameDoneNs, recordEndNs,
                 monotonic_ns()};
  g_pace.slotWaitNs = 0;
}

// The render worker's 3D image for `tickFrame`, submitted now (fd: a dup of
// its Dawn sync fd, owned from here).
void pace_submit(uint64_t tickFrame, int fd) {
#ifdef __ANDROID__
  std::lock_guard lock{g_pace.mutex};
  if (!g_pace.started) {
    std::thread(pace_watch).detach();
    g_pace.started = true;
  }
  PaceProbe::Pending p{tickFrame, std::max(g_present.latency, g_displayPerGameFrame.load()), g_pace.slotWaitNs, {}, fd};
  p.ns[PaceProbe::RecordStart] = g_pace.next[0];
  p.ns[PaceProbe::WorldEnd] = g_pace.next[1];
  p.ns[PaceProbe::GameDone] = g_pace.next[2];
  p.ns[PaceProbe::RecordEnd] = g_pace.next[3];
  p.ns[PaceProbe::RenderStart] = g_pace.next[4];
  p.ns[PaceProbe::Submit] = monotonic_ns();
  g_pace.pending.push_back(p);
  g_pace.cv.notify_one();
#else
  close(fd);
#endif
}

std::string pace_summary() {
  std::lock_guard lock{g_pace.mutex};
  auto pct = [](std::vector<float>& v, float q) {
    if (v.empty())
      return 0.f;
    const size_t i = std::min(v.size() - 1, static_cast<size_t>(q * static_cast<float>(v.size())));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i), v.end());
    return v[i];
  };
  if (g_pace.ms[PaceProbe::Done].empty())
    return {};
  static constexpr const char* kNames[] = {"woken",    "recording", "world recorded", "game done",
                                           "recorded", "replaying", "submitted",      "GPU done"};
  std::string s = "; pacing (ms after the tick, median/95%/max):";
  for (int m = 0; m < PaceProbe::Margin; ++m) {
    auto& v = g_pace.ms[m];
    s += fmt::format(" {} {:.1f}/{:.1f}/{:.1f}{}", kNames[m], pct(v, 0.5f), pct(v, 0.95f), pct(v, 1.f),
                     m + 1 < PaceProbe::Margin ? "," : ";");
  }
  if (!g_pace.spikes.empty()) {
    // Its own line: logcat cuts the stats line short.
    std::string spikes;
    for (const auto& sp : g_pace.spikes)
      spikes += " " + sp;
    Log.info("Frames done after their slot (ms after the tick):{}", spikes);
    g_pace.spikes.clear();
  }
  auto& margin = g_pace.ms[PaceProbe::Margin];
  auto under = [&](float ms) { return std::count_if(margin.begin(), margin.end(), [&](float m) { return m < ms; }); };
  s += fmt::format(" {} done over 1 ms after their slot;", g_pace.lateFrames);
  g_pace.lateFrames = 0;
  s += fmt::format(" margin before display {:.1f} median, {:.1f} least; under 8/10/12/14 ms: {}/{}/{}/{} of {}",
                   pct(margin, 0.5f), pct(margin, 0.f), under(8.f), under(10.f), under(12.f), under(14.f),
                   margin.size());
  for (auto& v : g_pace.ms)
    v.clear();
  if (const std::string cpu = gfx::frame_cpu_summary(); !cpu.empty())
    s += "; CPU per frame (ms, median/95%/max): " + cpu;
  return s;
}

// The most display frames past the tick an image is held for: about 33 ms
// (4 at 120 Hz, 2 with SpaceWarp's 60 Hz loop). Holding images longer ties
// up the swapchain's few images, and the render worker then finds none free.
int latency_cap() {
  const float hz = g_displayHz > 0.f ? g_displayHz.load() : 120.f;
  return std::max(std::max(g_displayPerGameFrame.load(), 1) + 1, static_cast<int>(hz / 30.f + 0.01f));
}

void note_ready_latency(uint64_t lat) {
  auto& p = g_present;
  p.samples[p.next++ % p.samples.size()] = static_cast<uint8_t>(std::min<uint64_t>(lat, 15));
  p.count = std::min(p.count + 1, p.samples.size());
  if (p.forced >= 0) {
    p.latency = p.forced;
    return;
  }
  if (++p.sinceUpdate < 30)
    return;
  p.sinceUpdate = 0;
  std::array<int, 16> hist{};
  for (size_t i = 0; i < p.count; ++i)
    ++hist[p.samples[i]];
  int k = 0;
  for (int acc = 0; k < 16; ++k) {
    acc += hist[k];
    if (acc * 100 >= static_cast<int>(p.count) * 97)
      break;
  }
  k = std::min(k, latency_cap());
  // SpaceWarp: two loop frames (33 ms), always. Staged frames often finish
  // within one, and latency 1 then held them to it: dynamic resolution cut
  // the scale to keep it, the latency went back to 2, the scale rose, and so
  // on (0.8 to 1.3 and back on two-CPU Battlefield). The runtime covers the
  // head's latency anyway.
  if (appsw_on())
    k = std::max(k, std::min(2, latency_cap()));
  if (k > p.latency) {
    p.latency = k;
    p.steadyLower = 0;
  } else if (k < p.latency) {
    const int64_t now = monotonic_ns();
    while (k < p.latency && now < p.bannedUntilNs[k])
      ++k;
    if (k < p.latency && ++p.steadyLower >= 4) {
      p.latency = k;
      p.steadyLower = 0;
    }
  } else {
    p.steadyLower = 0;
  }
}

// Android: each 3D frame's GPU work against its release at the latency it
// was drawn at. The picker samples the GPU finishing less the render
// worker's wait for an image, as if the image would come sooner at a lower
// latency; on Mute City and Pokémon Stadium (four CPUs) it often didn't,
// and frames at latency 3 finished at 26-28 ms against 25 and went stale.
// More than 2 of the last 120 frames over a millisecond late, below the
// cap, and the latency goes up one, the one left barred for 15 s (then 30,
// 60, 120 if it fails again; 10 s held clears that).
void note_lateness(int latency, bool late) {
  auto& p = g_present;
  if (p.forced >= 0 || latency != p.latency)
    return;
  if (p.lateLatency != p.latency) {
    p.lateLatency = p.latency;
    p.late = {};
    p.lateCount = p.lateNext = p.lateFrames = 0;
  }
  p.lateCount += (late ? 1 : 0) - (p.late[p.lateNext] ? 1 : 0);
  p.late[p.lateNext] = late;
  p.lateNext = (p.lateNext + 1) % static_cast<int>(p.late.size());
  if (++p.lateFrames == 600)
    p.bans[p.latency] = 0;
  if (p.lateCount > 2 && p.latency < latency_cap()) {
    const int n = std::min(p.bans[p.latency]++, 3);
    p.bannedUntilNs[p.latency] = monotonic_ns() + (int64_t{15'000'000'000} << n);
    ++p.latency;
    p.steadyLower = 0;
    ++p.raisedForLate;
  }
}

// Releases drawn images in acquisition order: waits (on the GPU, on our
// queue) for Dawn's semaphores, puts the image back in the layout the
// runtime expects, then hands it to the runtime.
void write_dump_layout(const std::array<XrView, 2>& views);

// Fixed-latency presentation: hold a 3D or HUD image for its slot (true:
// not yet; g_mutex held).
// On with dynamic resolution, which keeps frames on time: without it,
// frames running late on an overloaded stage were held into slots the
// render worker then lacked (Pokémon Stadium rock, four CPUs: 58.2 ->
// 55.5 game fps). AURORA_XR_FIXED_LATENCY=<n> forces it on.
bool held_for_slot(Stream& st, Slot& sl) {
  const bool fixedLatency = g_present.forced > 0 || (g_present.forced < 0 && g_dynres.on);
  if (!fixedLatency || g_displayPerGameFrame <= 0 || sl.tickFrame == 0 ||
      !(&st == &g_streams[kStereo] || &st == &g_streams[kHud] || &st == &g_streams[kMotion] ||
        &st == &g_streams[kMotionDepth]))
    return false;
  if (sl.readySeen == 0) {
    sl.readySeen = B.displayFrame;
#ifndef __ANDROID__
    // Android samples when the GPU work is done (the pacing probe).
    if (&st == &g_streams[kStereo] && B.displayFrame >= sl.tickFrame)
      note_ready_latency(B.displayFrame - sl.tickFrame);
#endif
  }
  if (B.displayFrame < sl.tickFrame + static_cast<uint64_t>(g_present.latency)) {
    if (&st == &g_streams[kStereo] && sl.readySeen == B.displayFrame)
      ++g_present.held;
    return true;
  }
  return false;
}

bool import_dawn_semaphores(PendingAccess& fromDawn, std::vector<VkSemaphore>& out) {
  for (int fd : fromDawn.fds) {
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore sem = VK_NULL_HANDLE;
    VK_TRY(vkCreateSemaphore(B.dev, &sci, nullptr, &sem));
    VkImportSemaphoreFdInfoKHR imp{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR};
    imp.semaphore = sem;
    imp.flags = kSemImportFlags;
    imp.handleType = kSemHandle;
    imp.fd = fd; // Vulkan owns it on success
    if (B.importSemaphoreFd(B.dev, &imp) != VK_SUCCESS) {
      close(fd);
      vkDestroySemaphore(B.dev, sem, nullptr);
      Log.error("Importing Dawn's semaphore failed");
      return false;
    }
    out.push_back(sem);
  }
  fromDawn.fds.clear();
  return true;
}

bool release_staged(Stream& st, bool& released);

// SpaceWarp frames go out as sets: a game tick's 3D, motion and depth images
// (and its HUD) are handed over together or not at all. Each stream used to
// hand over its own oldest finished frame as its own swapchain image came
// free, so when one stream lacked an image the others went on without it
// and the compositor got motion vectors and depth of another tick than the
// picture (up to a quarter of a bad window's loops on four-CPU Brinstar).
// Each loop picks the newest tick finished in all of the three; older frames
// are retired, and the tick is released only if the three have an image to
// copy into, else it waits as it is (a newer one, if it comes, replaces it).
// AURORA_XR_SET_HANDOVER=0 turns it off.
struct HandoverGate {
  bool on = false;
  uint64_t tick = 0; // 0: no set is due
  bool blocked = false;
  // Ready frames whose tick is missing from one of the three streams (a 3D
  // image missed while its motion went out, say): no set can ever hold them,
  // so they're retired. Kept, they filled the rings and the 3D view froze.
  std::vector<uint64_t> orphans;
} g_gate;

bool gated_stream(const Stream& st) {
  return &st == &g_streams[kStereo] || &st == &g_streams[kHud] || &st == &g_streams[kMotion] ||
         &st == &g_streams[kMotionDepth];
}

// g_mutex held.
bool has_ready_tick(Stream& st, uint64_t tick) {
  for (Slot& sl : st.ring)
    if (sl.state == SlotState::Ready && sl.tickFrame == tick && !held_for_slot(st, sl))
      return true;
  return false;
}

bool has_free_image(Stream& st) {
  std::lock_guard lock{g_mutex};
  for (const Slot& sl : st.slots)
    if (sl.state == SlotState::Free)
      return true;
  return false;
}

// g_mutex held: st has a frame of `tick` in hand (drawing or finished).
bool has_tick(const Stream& st, uint64_t tick) {
  for (const Slot& sl : st.ring)
    if ((sl.state == SlotState::Rendering || sl.state == SlotState::Ready) && sl.tickFrame == tick)
      return true;
  return false;
}

bool compute_gate() {
  g_gate.on = false;
  g_gate.tick = 0;
  g_gate.blocked = false;
  g_gate.orphans.clear();
  if (!g_tweak.set || !appsw_on() || g_displayPerGameFrame <= 0 || !g_streams[kStereo].staged ||
      !g_streams[kMotion].staged || !g_streams[kMotionDepth].staged)
    return true;
  g_gate.on = true;
  {
    std::lock_guard lock{g_mutex};
    for (const int id : {kStereo, kHud, kMotion, kMotionDepth})
      for (const Slot& sl : g_streams[id].ring)
        if (sl.state == SlotState::Ready && sl.tickFrame != 0 &&
            (!has_tick(g_streams[kStereo], sl.tickFrame) || !has_tick(g_streams[kMotion], sl.tickFrame) ||
             !has_tick(g_streams[kMotionDepth], sl.tickFrame)))
          g_gate.orphans.push_back(sl.tickFrame);
    for (Slot& sl : g_streams[kStereo].ring) {
      if (sl.state != SlotState::Ready || sl.tickFrame == 0 || sl.tickFrame <= g_gate.tick ||
          held_for_slot(g_streams[kStereo], sl))
        continue;
      if (has_ready_tick(g_streams[kMotion], sl.tickFrame) && has_ready_tick(g_streams[kMotionDepth], sl.tickFrame))
        g_gate.tick = sl.tickFrame;
    }
  }
  if (g_gate.tick == 0)
    return true;
  // The three images the compositor needs together, the 3D one longest:
  // up to 10 ms (AURORA_XR_GATE_WAIT_MS). There are three, and at latency 2
  // one is shown and one queued, so the third comes back only as the
  // compositor lets go, later when it's loaded (SpaceWarp filling frames
  // under a heavy stage, a fast head turn). At 6 ms the wait ran out 140
  // times in 10 s as Brinstar Depths' Kraid rose: a frame done in time was
  // repeated and the next found the ring full of stale ones (3D at 46-53 a
  // second). At 10 the image always came (8 ms at most), the loop still
  // ending 24 ms before display: 59.9.
  static const int64_t stereoWaitNs =
      static_cast<int64_t>(std::clamp(env_float("AURORA_XR_GATE_WAIT_MS", 10.f), 0.5f, 14.f) * 1e6f);
  for (const int id : {kStereo, kMotion, kMotionDepth}) {
    Stream& st = g_streams[id];
    if (has_free_image(st))
      continue;
    if (!acquire_ahead(st, id == kStereo ? stereoWaitNs : 2'000'000))
      return false;
    if (!has_free_image(st)) {
      g_gate.blocked = true;
      ++g_stagedNoImage;
      ++g_noImageBy[id];
      break;
    }
  }
  return true;
}

bool release_ready(Stream& st, bool& released) {
  if (st.staged)
    return release_staged(st, released);
  released = false;
  for (;;) {
    int index = -1;
    PendingAccess fromDawn;
    std::array<XrView, 2> views{};
    std::array<int32_t, 2> rect{};
    {
      std::lock_guard lock{g_mutex};
      uint64_t oldest = UINT64_MAX;
      for (size_t i = 0; i < st.slots.size(); ++i) {
        const auto& sl = st.slots[i];
        if (sl.state != SlotState::Released && sl.acquireSeq < oldest) {
          oldest = sl.acquireSeq;
          index = static_cast<int>(i);
        }
      }
      if (index < 0 || st.slots[index].state != SlotState::Ready)
        return true;
      auto& sl = st.slots[index];
#ifdef __ANDROID__
      // AURORA_XR_RELEASE_DONE=1: not before its GPU work is done. The
      // runtime's xrBeginFrame waits for the work of images already handed
      // over, so an image released while its eye pass still ran stalled the
      // XR loop for up to 25 ms and display frames went stale (Mute City,
      // four CPUs, at 1.3: 10 a second). Held, those stalls go, but OpenXR
      // allows one waited image per swapchain until it's released, so the
      // render worker then waits and drops 3D frames (30-50 a second); a
      // second swapchain only made it two images in flight. Off.
      static const bool waitDone = env_flag("AURORA_XR_RELEASE_DONE", false);
      if (waitDone)
        for (int fd : sl.fromDawn.fds) {
          pollfd pfd{.fd = fd, .events = POLLIN};
          if (poll(&pfd, 1, 0) <= 0)
            return true;
        }
#endif
      if (held_for_slot(st, sl))
        return true;
      fromDawn = std::move(st.slots[index].fromDawn);
      st.slots[index].fromDawn = {};
      views = st.slots[index].views;
      rect = st.slots[index].rect;
    }
    Slot& s = st.slots[index];
    retire_slot(st, s, index);

    if (!import_dawn_semaphores(fromDawn, s.waitSems))
      return false;

    // OpenXR takes Vulkan color swapchain images back in
    // COLOR_ATTACHMENT_OPTIMAL; Dawn may have left another layout.
    // Depth images (SpaceWarp's) go back in DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
    const VkImageLayout home =
        st.depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    const VkImageAspectFlags aspect =
        !st.depth ? VK_IMAGE_ASPECT_COLOR_BIT
        : st.format == VK_FORMAT_D24_UNORM_S8_UINT ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                                   : VK_IMAGE_ASPECT_DEPTH_BIT;
    const VkImageLayout from = fromDawn.newLayout != VK_IMAGE_LAYOUT_UNDEFINED ? fromDawn.newLayout : home;
    const bool dumpNow = !st.ownFormat && !B.dumpDir.empty() && !st.dumped && st.dumpSlot < 0 &&
                         st.releases >= static_cast<uint64_t>(env_float("AURORA_XR_DUMP_AFTER", 300.f)) &&
                         (st.dumpBuf || create_dump_buffer(st));
    const bool record = dumpNow || from != home;
    if (record) {
      VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
      cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      VK_TRY(vkResetCommandBuffer(s.cmd, 0));
      VK_TRY(vkBeginCommandBuffer(s.cmd, &cbi));
      VkImageLayout layout = from;
      if (dumpNow) {
        const auto toSrc = barrier(s.image, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &toSrc);
        VkBufferImageCopy rb{};
        rb.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, st.layers}; // layers stack in the dump
        rb.imageExtent = {st.width, st.height, 1};
        vkCmdCopyImageToBuffer(s.cmd, s.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, st.dumpBuf, 1, &rb);
        st.dumpSlot = index;
        layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        if (&st == &g_streams[kStereo])
          write_dump_layout(s.views);
      }
      if (layout != home) {
        const auto back =
            barrier(s.image, layout, home,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_TRANSFER_READ_BIT,
                    0, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, aspect);
        vkCmdPipelineBarrier(s.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &back);
      }
      VK_TRY(vkEndCommandBuffer(s.cmd));
    }
    std::vector<VkPipelineStageFlags> stages(s.waitSems.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = static_cast<uint32_t>(s.waitSems.size());
    si.pWaitSemaphores = s.waitSems.data();
    si.pWaitDstStageMask = stages.data();
    si.commandBufferCount = record ? 1 : 0;
    si.pCommandBuffers = &s.cmd;
    VK_TRY(vkQueueSubmit(B.queue, 1, &si, s.fence));
    s.inFlight = true;
    if (&st == &g_streams[kStereo])
      trace_counter("XR 3D released", static_cast<int64_t>(st.releases & 1));
    XR_TRY(xrReleaseSwapchainImage(st.swapchain, nullptr));
    {
      std::lock_guard lock{g_mutex};
      s.state = SlotState::Released;
    }
    st.haveImage = true;
    st.lastRelease = std::chrono::steady_clock::now();
    st.shownViews = views;
    st.shownRect = rect;
    ++st.releases;
    released = true;
  }
}

// A staged stream's hand-over (create_ring): its oldest finished image, once
// its fixed-latency slot comes, is copied into the swapchain image the XR
// thread holds waited (acquire_ahead), after Dawn's work on it, and that is
// released. The ring image goes back to the render worker when the copy is
// done. Frames go in the order the render worker took their images.
bool release_staged(Stream& st, bool& released) {
  released = false;
  bool freed = false;
  for (int i = 0; i < static_cast<int>(st.ring.size()); ++i) {
    Slot& r = st.ring[i];
    if (!r.inFlight || vkGetFenceStatus(B.dev, r.fence) != VK_SUCCESS)
      continue;
    vkResetFences(B.dev, 1, &r.fence);
    for (VkSemaphore sem : r.waitSems)
      vkDestroySemaphore(B.dev, sem, nullptr);
    r.waitSems.clear();
    if (st.dumpRing == i) {
      st.dumpRing = -1;
      st.dumped = true;
      write_dump(st);
    }
    std::lock_guard lock{g_mutex};
    r.inFlight = false; // read by the render worker under g_mutex (take_slot_locked)
    if (r.state == SlotState::Released) // the render worker may have taken it already
      r.state = SlotState::Free;
    freed = true;
  }
  if (freed)
    g_slotFreeCv.notify_all();
  const VkImageLayout home =
      st.depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  const VkImageAspectFlags aspect =
      !st.depth ? VK_IMAGE_ASPECT_COLOR_BIT
      : st.format == VK_FORMAT_D24_UNORM_S8_UINT ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                                 : VK_IMAGE_ASPECT_DEPTH_BIT;
  const VkImageAspectFlags copyAspect = st.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
  for (;;) {
    int index = -1, image = -1;
    bool retire = false;
    PendingAccess fromDawn;
    std::array<XrView, 2> views{};
    std::array<int32_t, 2> rect{};
    {
      std::lock_guard lock{g_mutex};
      for (int i = 0; i < static_cast<int>(st.ring.size()); ++i) {
        const SlotState state = st.ring[i].state;
        if ((state == SlotState::Rendering || state == SlotState::Ready) &&
            (index < 0 || st.ring[i].acquireSeq < st.ring[index].acquireSeq))
          index = i;
      }
      const bool gated = g_gate.on && gated_stream(st);
      if (index < 0 || st.ring[index].state != SlotState::Ready || (!gated && held_for_slot(st, st.ring[index])))
        return true;
      // A later frame due too: this one is stale, retired unshown. The
      // loop hands over one frame per loop, and with SpaceWarp's 60 Hz loop
      // one comes per loop: a loop without an image to copy into left every
      // later frame a loop late for the rest of the fight, and the 3D,
      // HUD and SpaceWarp images, each its own queue, drifted apart (the
      // motion vectors of one frame on the picture of another: SpaceWarp's
      // distortion grew through a Temple fight).
      bool newerDue = false;
      if (gated) {
        // The set's tick goes out (if its images are there); older frames
        // are retired; none newer than it is due yet.
        const uint64_t tick = st.ring[index].tickFrame;
        const bool orphan = std::find(g_gate.orphans.begin(), g_gate.orphans.end(), tick) != g_gate.orphans.end();
        if (!orphan && (g_gate.tick == 0 || tick > g_gate.tick))
          return true;
        newerDue = orphan || tick < g_gate.tick;
        if (!newerDue && g_gate.blocked)
          return true;
      } else {
        for (int i = 0; i < static_cast<int>(st.ring.size()) && !newerDue; ++i)
          newerDue = i != index && st.ring[i].state == SlotState::Ready &&
                     st.ring[i].acquireSeq > st.ring[index].acquireSeq && !held_for_slot(st, st.ring[i]);
      }
      if (newerDue) {
        retire = true;
        fromDawn = std::move(st.ring[index].fromDawn);
        st.ring[index].fromDawn = {};
        st.ring[index].state = SlotState::Released;
      } else {
        for (int i = 0; i < static_cast<int>(st.slots.size()); ++i)
          if (st.slots[i].state == SlotState::Free)
            image = i;
      }
    }
    // A frame due and no swapchain image waited yet: wait for one here, up
    // to 6 ms, instead of letting this loop pass. acquire_ahead's 2 ms often
    // ran out with SpaceWarp, and a lost loop left two frames due at once on
    // the next (Temple: 3D at 52-57 a second, frames done ~20 ms after the
    // tick against a 33 ms slot).
    if (!retire && image < 0 && !g_stop) {
      if (!acquire_ahead(st, 6'000'000))
        return false;
      std::lock_guard lock{g_mutex};
      for (int i = 0; i < static_cast<int>(st.slots.size()); ++i)
        if (st.slots[i].state == SlotState::Free)
          image = i;
      if (image < 0) {
        ++g_stagedNoImage;
        ++g_noImageBy[&st - g_streams.data()];
      }
    }
    if (retire) {
      // Retire: wait for Dawn's work on it, then it's free (release_staged's
      // cleanup), with nothing copied.
      Slot& r = st.ring[index];
      if (!import_dawn_semaphores(fromDawn, r.waitSems))
        return false;
      std::vector<VkPipelineStageFlags> stages(r.waitSems.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
      VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
      si.waitSemaphoreCount = static_cast<uint32_t>(r.waitSems.size());
      si.pWaitSemaphores = r.waitSems.data();
      si.pWaitDstStageMask = stages.data();
      VK_TRY(vkQueueSubmit(B.queue, 1, &si, r.fence));
      {
        std::lock_guard lock{g_mutex};
        r.inFlight = true;
      }
      ++g_stagedStale;
      continue;
    }
    {
      std::lock_guard lock{g_mutex};
      if (image < 0)
        return true; // no swapchain image waited yet: next loop
      fromDawn = std::move(st.ring[index].fromDawn);
      st.ring[index].fromDawn = {};
      views = st.ring[index].views;
      rect = st.ring[index].rect;
    }
    Slot& r = st.ring[index];
    Slot& s = st.slots[image];
    if (!import_dawn_semaphores(fromDawn, r.waitSems))
      return false;
    const VkImageLayout from = fromDawn.newLayout != VK_IMAGE_LAYOUT_UNDEFINED ? fromDawn.newLayout
                               : fromDawn.oldLayout != VK_IMAGE_LAYOUT_UNDEFINED ? fromDawn.oldLayout
                                                                                 : home;
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_TRY(vkResetCommandBuffer(r.cmd, 0));
    VK_TRY(vkBeginCommandBuffer(r.cmd, &cbi));
    constexpr VkAccessFlags written = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    const std::array toCopy{
        barrier(r.image, from, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, written, VK_ACCESS_TRANSFER_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, aspect),
        barrier(s.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, aspect),
    };
    vkCmdPipelineBarrier(r.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, static_cast<uint32_t>(toCopy.size()), toCopy.data());
    // Dynamic resolution: only the corner drawn (the layer shows only that).
    const bool corner = &st == &g_streams[kStereo] && rect[0] > 0 && rect[1] > 0;
    VkImageCopy region{};
    region.srcSubresource = {copyAspect, 0, 0, st.layers};
    region.dstSubresource = region.srcSubresource;
    region.extent = {corner ? static_cast<uint32_t>(rect[0]) : st.width,
                     corner ? static_cast<uint32_t>(rect[1]) : st.height, 1};
    vkCmdCopyImage(r.cmd, r.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    const bool dumpNow = !st.ownFormat && !B.dumpDir.empty() && !st.dumped && st.dumpRing < 0 &&
                         st.releases >= static_cast<uint64_t>(env_float("AURORA_XR_DUMP_AFTER", 300.f)) &&
                         (st.dumpBuf || create_dump_buffer(st));
    if (dumpNow) {
      const auto toSrc = barrier(s.image, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_ACCESS_TRANSFER_READ_BIT);
      vkCmdPipelineBarrier(r.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                           nullptr, 1, &toSrc);
      VkBufferImageCopy rb{};
      rb.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, st.layers}; // layers stack in the dump
      rb.imageExtent = {st.width, st.height, 1};
      vkCmdCopyImageToBuffer(r.cmd, s.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, st.dumpBuf, 1, &rb);
      st.dumpRing = index;
      layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      if (&st == &g_streams[kStereo])
        write_dump_layout(views);
    }
    const auto back = barrier(s.image, layout, home, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, 0,
                              VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, aspect);
    vkCmdPipelineBarrier(r.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &back);
    VK_TRY(vkEndCommandBuffer(r.cmd));
    std::vector<VkPipelineStageFlags> stages(r.waitSems.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = static_cast<uint32_t>(r.waitSems.size());
    si.pWaitSemaphores = r.waitSems.data();
    si.pWaitDstStageMask = stages.data();
    si.commandBufferCount = 1;
    si.pCommandBuffers = &r.cmd;
    VK_TRY(vkQueueSubmit(B.queue, 1, &si, r.fence));
    {
      std::lock_guard lock{g_mutex};
      r.inFlight = true;
    }
    if (&st == &g_streams[kStereo])
      trace_counter("XR 3D released", static_cast<int64_t>(st.releases & 1));
    XR_TRY(xrReleaseSwapchainImage(st.swapchain, nullptr));
    {
      std::lock_guard lock{g_mutex};
      s.state = SlotState::Released;
      r.state = SlotState::Released;
    }
    st.haveImage = true;
    st.lastRelease = std::chrono::steady_clock::now();
    st.shownViews = views;
    st.shownTick = r.tickFrame;
    st.shownRect = rect;
    ++st.releases;
    released = true;
  }
}

// CPU and GPU clock levels (XR_EXT_performance_settings), per domain:
// AURORA_XR_PERF_CPU / AURORA_XR_PERF_GPU = off, low, high, boost.
// - CPU: high by default on Android. Light scenes let the CPU drop to its
//   lowest clock, and the XR thread then misses its 120 Hz submits: 2-player
//   Battlefield showed 6-7 stale frames a second at CPU level 2 and 0.2 at
//   level 3 with the request (2026-10-03).
// - GPU: the runtime's own by default. A sustained-high request changed
//   nothing: passthrough caps a Quest 3's GPU at level 2 (docs/quest-xr.md).
void request_performance_levels() {
  if (!B.setPerformanceLevel) {
    Log.info("XR_EXT_performance_settings unavailable; the runtime picks clock levels");
    return;
  }
  const auto request = [](const char* name, const char* fallback, XrPerfSettingsDomainEXT domain,
                          const char* label) {
    const char* v = std::getenv(name);
    const std::string_view s = v != nullptr && *v != '\0' ? v : fallback;
    if (s == "off")
      return;
    const auto lvl = s == "low"     ? XR_PERF_SETTINGS_LEVEL_SUSTAINED_LOW_EXT
                     : s == "boost" ? XR_PERF_SETTINGS_LEVEL_BOOST_EXT
                                    : XR_PERF_SETTINGS_LEVEL_SUSTAINED_HIGH_EXT;
    const XrResult r = B.setPerformanceLevel(B.session, domain, lvl);
    Log.info("Performance level requested: {} {} ({})", label, static_cast<int>(lvl), static_cast<int>(r));
  };
#ifdef __ANDROID__
  request("AURORA_XR_PERF_CPU", "high", XR_PERF_SETTINGS_DOMAIN_CPU_EXT, "CPU");
#else
  request("AURORA_XR_PERF_CPU", "off", XR_PERF_SETTINGS_DOMAIN_CPU_EXT, "CPU");
#endif
  request("AURORA_XR_PERF_GPU", "off", XR_PERF_SETTINGS_DOMAIN_GPU_EXT, "GPU");
}

void request_refresh_rate() {
  if (!B.enumerateRefreshRates || !B.requestRefreshRate)
    return;
  uint32_t n = 0;
  if (XR_FAILED(B.enumerateRefreshRates(B.session, 0, &n, nullptr)) || n == 0)
    return;
  std::vector<float> rates(n);
  if (XR_FAILED(B.enumerateRefreshRates(B.session, n, &n, rates.data())))
    return;
  std::string list;
  for (float r : rates)
    list += fmt::format("{}{:.0f}", list.empty() ? "" : ", ", r);
  const auto offered = [&](float want) {
    return std::any_of(rates.begin(), rates.end(), [&](float r) { return std::abs(r - want) < 0.5f; });
  };
  float pick = 0.f;
  if (const float wanted = env_float("AURORA_XR_REFRESH", 0.f); wanted > 0.f && offered(wanted))
    pick = wanted;
  else if (offered(60.f))
    pick = 60.f;
  else if (offered(120.f))
    pick = 120.f;
  if (pick > 0.f && XR_SUCCEEDED(B.requestRefreshRate(B.session, pick)))
    Log.info("Display refresh rates offered: {}; requested {:.0f} Hz", list, pick);
  else
    Log.info("Display refresh rates offered: {}; keeping the default", list);
}

// Recompute how many display frames each game frame gets from the current
// display rate (on session start and on XR_FB_display_refresh_rate changes).
// The XR loop's period while Application SpaceWarp runs it at half the
// display rate (xrWaitFrame's predictedDisplayPeriod; 0: the display's).
int64_t g_loopPeriodNs = 0;

void update_pacing() {
  float rate = 0.f;
  if (B.getRefreshRate)
    B.getRefreshRate(B.session, &rate);
  g_refreshHz = rate;
  // SpaceWarp: the loop runs at what xrWaitFrame paces it to, and the game
  // ticks on that.
  if (g_loopPeriodNs > 0)
    rate = static_cast<float>(1e9 / static_cast<double>(g_loopPeriodNs));
  if (rate <= 0.f && !B.getRefreshRate)
    rate = 60.f; // no extension (e.g. some runtimes): assume a 60 Hz display
  g_displayHz = rate;
  const float ratio = rate / 60.f;
  const int per = std::abs(ratio - std::round(ratio)) < 0.02f ? static_cast<int>(std::round(ratio)) : 0;
  const int old = g_displayPerGameFrame.exchange(env_flag("AURORA_XR_LOCKSTEP", true) ? per : 0);
  if (old != g_displayPerGameFrame)
    Log.info("Display at {:.1f} Hz: {}", rate,
             g_displayPerGameFrame ? fmt::format("game paced to every {} display frame(s)", g_displayPerGameFrame.load())
                                   : std::string{"not a multiple of 60 Hz, game keeps its own timer"});
}

// Publish where the eyes will be when the render worker's next 3D frame is
// shown. The frame starts on this loop's tick but, under fixed-latency
// presentation, is handed over `latency` loops later: the pose it carries is
// predicted for that display time, not this loop's, or the compositor has to
// bridge the whole hold (17-50 ms of head motion) by reprojection alone.
// AURORA_XR_POSE_LEAD=0 predicts for this loop's display time.
void publish_views(XrTime displayTime) {
  XrViewLocateInfo vli{XR_TYPE_VIEW_LOCATE_INFO};
  vli.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
  vli.displayTime = displayTime;
  vli.space = B.space;
  XrViewState vs{XR_TYPE_VIEW_STATE};
  std::array<XrView, 2> views{{{XR_TYPE_VIEW}, {XR_TYPE_VIEW}}};
  uint32_t n = 0;
  if (XR_FAILED(xrLocateViews(B.session, &vli, &vs, 2, &n, views.data())) || n != 2)
    return;
  const bool valid = (vs.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) != 0 &&
                     (vs.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) != 0;
  std::array<XrView, 2> renderViews = views;
  const bool lead = g_tweak.lead;
  const int latency = g_present.latency;
  if (valid && lead && latency > 0 && g_displayPerGameFrame > 0) {
    const int64_t period = g_loopPeriodNs > 0 ? g_loopPeriodNs
                                              : static_cast<int64_t>(1e9 / std::max(g_displayHz.load(), 1.f));
    XrViewLocateInfo ahead = vli;
    ahead.displayTime = displayTime + static_cast<int64_t>(latency) * period;
    XrViewState as{XR_TYPE_VIEW_STATE};
    std::array<XrView, 2> aheadViews{{{XR_TYPE_VIEW}, {XR_TYPE_VIEW}}};
    uint32_t an = 0;
    if (XR_SUCCEEDED(xrLocateViews(B.session, &ahead, &as, 2, &an, aheadViews.data())) && an == 2 &&
        (as.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT) != 0 &&
        (as.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) != 0)
      renderViews = aheadViews;
  }
  std::lock_guard lock{g_viewMutex};
  if (valid) {
    g_latestViews = renderViews;
    g_viewsValid = true;
    const XrVector3f &l = views[0].pose.position, &r = views[1].pose.position;
    B.head = {(l.x + r.x) * 0.5f, (l.y + r.y) * 0.5f, (l.z + r.z) * 0.5f};
    B.headOrientation = views[0].pose.orientation;
    // The turn since the last pose: 2 asin |q_rel.xyz|, from the quaternions'
    // dot (no libm calls here: the game's own libm interposes some).
    const XrQuaternionf& q = B.headOrientation;
    const XrQuaternionf& p = g_headTurnPrev;
    const double dt = (displayTime - g_headTurnPrevTime) * 1e-9;
    if (g_headTurnPrevTime != 0 && dt > 1e-4 && dt < 0.1) {
      const float dot = std::min(1.f, std::fabs(q.x * p.x + q.y * p.y + q.z * p.z + q.w * p.w));
      const float deg = 2.f * std::sqrt(std::max(0.f, 1.f - dot * dot)) * 57.2958f; // small angles
      const float speed = static_cast<float>(deg / dt);
      g_headTurnMax = std::max(g_headTurnMax, speed);
      g_headTurnSum += speed;
      ++g_headTurnN;
      g_headTurnLast.store(speed, std::memory_order_relaxed);
    }
    g_headTurnPrev = q;
    g_headTurnPrevTime = displayTime;
  }
}

// ---------------------------------------------------------------- XR thread: arena and screen placement
//
// While a fight is paused each hand, holding a controller or tracked (the
// controller put down), points a laser at the arena, or shows a dot once its
// touch point (the controller's tip, or the pinch) is inside the arena's grab
// box. Squeezing the grip or pinching with the laser on the arena, or inside
// the box, grabs it:
//   one hand   the arena hangs off the laser at the grabbed point, or off
//              the touch point, keeping its heading; letting go leaves it
//   two hands  squeezing or pinching with the other hand too (anywhere)
//              scales the arena and turns it about the vertical axis around
//              the point between the touch points, and the arena follows
//              that point as it moves; letting go of one hand carries on
//              with the other alone
// Controllers and hands work the same way and mix freely.
// Starting position and scale: AURORA_XR_ARENA_POS, AURORA_XR_ARENA_SCALE.
//
// The virtual screen (menus) works like a Quest window, any time it shows.
// Its lasers appear only while they point at it. On the picture a laser
// points for the game (aurora_xr_screen_pointer) and the trigger or a pinch
// clicks; the grips press Z and the triggers L and R unless a laser is on
// the screen. The bar under it is the handle: a grip, trigger or pinch with
// the laser on it, or a hand touching it, takes hold of the screen, and
// while one hand holds it a press of the other with its laser anywhere on
// the screen joins in. It always turns to face the head; one controller
// dragging it also pushes and pulls it with its stick, and two hands resize
// it about its center. While hands are tracked, a Back button left of the
// bar presses B for as long as a pinch on it is held (aurora_xr_screen_back).
// The arena and the screen keep separate poses: moving one never moves the
// other.

XrVector3f operator+(XrVector3f a, XrVector3f b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
XrVector3f operator-(XrVector3f a, XrVector3f b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
XrVector3f operator*(XrVector3f a, float k) { return {a.x * k, a.y * k, a.z * k}; }
float vdot(XrVector3f a, XrVector3f b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
XrVector3f vcross(XrVector3f a, XrVector3f b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
float vlen(XrVector3f a) { return std::sqrt(vdot(a, a)); }
XrVector3f vnorm(XrVector3f a) {
  const float l = vlen(a);
  return l > 1e-6f ? a * (1.f / l) : a;
}
// v rotated by the unit quaternion q.
XrVector3f qrot(const XrQuaternionf& q, XrVector3f v) {
  const XrVector3f u{q.x, q.y, q.z};
  const XrVector3f t = vcross(u, v) * 2.f;
  return v + t * q.w + vcross(u, t);
}
// Rotation by `a` radians about +Y (counter-clockwise seen from above).
XrVector3f rot_y(XrVector3f v, float a) {
  const float c = std::cos(a), s = std::sin(a);
  return {c * v.x + s * v.z, v.y, -s * v.x + c * v.z};
}
XrQuaternionf yaw_quat(float a) { return {0.f, std::sin(a / 2.f), 0.f, std::cos(a / 2.f)}; }
// The rotation whose local axes are x, y, z (orthonormal).
XrQuaternionf quat_from_axes(XrVector3f x, XrVector3f y, XrVector3f z) {
  const float tr = x.x + y.y + z.z;
  XrQuaternionf q;
  if (tr > 0.f) {
    const float s = std::sqrt(tr + 1.f) * 2.f;
    q = {(y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s, s / 4.f};
  } else if (x.x > y.y && x.x > z.z) {
    const float s = std::sqrt(1.f + x.x - y.y - z.z) * 2.f;
    q = {s / 4.f, (y.x + x.y) / s, (z.x + x.z) / s, (y.z - z.y) / s};
  } else if (y.y > z.z) {
    const float s = std::sqrt(1.f + y.y - x.x - z.z) * 2.f;
    q = {(y.x + x.y) / s, s / 4.f, (z.y + y.z) / s, (z.x - x.z) / s};
  } else {
    const float s = std::sqrt(1.f + z.z - x.x - y.y) * 2.f;
    q = {(z.x + x.z) / s, (z.y + y.z) / s, s / 4.f, (x.y - y.x) / s};
  }
  return q;
}

// Under g_arenaMutex.
void init_arena() {
  if (g_arenaInit)
    return;
  g_arenaInit = true;
  ArenaPose& d = g_defaultArena;
  if (const char* v = std::getenv("AURORA_XR_ARENA_POS"))
    std::sscanf(v, "%f,%f,%f", &d.pos.x, &d.pos.y, &d.pos.z);
  d.scale = env_float("AURORA_XR_ARENA_SCALE", 0.0035f);
  d.yaw = env_float("AURORA_XR_ARENA_YAW", 0.f) * 3.14159265f / 180.f; // degrees
  g_arena = d;
  g_arena.scale *= g_stageScale;
}

ArenaPose arena_pose() {
  std::lock_guard lock{g_arenaMutex};
  init_arena();
  return g_arena;
}

// Places the current stage only; it stays there for the rest of the session.
void set_arena_pose(const ArenaPose& a) {
  std::lock_guard lock{g_arenaMutex};
  g_arena = a;
  g_stageArenas[g_stage] = a;
}

// What the lasers can grab, in game units around the arena center (the game
// point placed at the arena position, aurora_xr_set_stage): by default
// wider and taller than any of Melee's main platforms, so pointing near it
// is enough; the game's own for the stage (aurora_xr_set_grab_box), which a
// hand inside it touches instead of pointing. Under g_arenaMutex.
constexpr XrVector3f kGrabBoxMin{-120.f, -80.f, -60.f};
constexpr XrVector3f kGrabBoxMax{120.f, 100.f, 60.f};
XrVector3f g_grabBoxMin = kGrabBoxMin, g_grabBoxMax = kGrabBoxMax;

std::pair<XrVector3f, XrVector3f> grab_box() {
  std::lock_guard lock{g_arenaMutex};
  return {g_grabBoxMin, g_grabBoxMax};
}
// Meters per game unit: Final Destination from about 25 cm to 8.5 m wide.
constexpr float kMinArenaScale = 0.0015f, kMaxArenaScale = 0.05f;

// A room point (meters) in game units from the arena center. Not from the
// world origin: on Corneria, centered on the Great Fox far above it, the box
// sat well below the ship.
XrVector3f to_game(const ArenaPose& a, XrVector3f p) { return rot_y(p - a.pos, -a.yaw) * (1.f / a.scale); }

bool in_grab_box(const ArenaPose& a, XrVector3f p) {
  const XrVector3f g = to_game(a, p);
  const auto [lo, hi] = grab_box();
  return g.x >= lo.x && g.x <= hi.x && g.y >= lo.y && g.y <= hi.y && g.z >= lo.z && g.z <= hi.z;
}

// Distance along the ray (meters) to the arena's grab box, or -1 for a miss.
float hit_arena(const ArenaPose& a, XrVector3f origin, XrVector3f dir) {
  const XrVector3f o = to_game(a, origin);
  const XrVector3f d = rot_y(dir, -a.yaw) * (1.f / a.scale);
  const float os[3] = {o.x, o.y, o.z}, ds[3] = {d.x, d.y, d.z};
  const auto [boxLo, boxHi] = grab_box();
  const float lo[3] = {boxLo.x, boxLo.y, boxLo.z};
  const float hi[3] = {boxHi.x, boxHi.y, boxHi.z};
  float t0 = -1e30f, t1 = 1e30f;
  for (int i = 0; i < 3; ++i) {
    if (std::abs(ds[i]) < 1e-9f) {
      if (os[i] < lo[i] || os[i] > hi[i])
        return -1.f;
      continue;
    }
    float ta = (lo[i] - os[i]) / ds[i], tb = (hi[i] - os[i]) / ds[i];
    if (ta > tb)
      std::swap(ta, tb);
    t0 = std::max(t0, ta);
    t1 = std::min(t1, tb);
  }
  if (t0 > t1 || t1 < 0.f)
    return -1.f;
  return std::max(t0, 0.f);
}

// The virtual screen, in the starting head space: a quad facing +Z in its
// own frame. XR thread only. Kept for the whole session.
struct ScreenPose {
  XrVector3f pos{0.f, 0.f, -0.85f};
  XrQuaternionf orientation{0.f, 0.f, 0.f, 1.f};
  float width = 0.7f; // meters: about 45 degrees across at 0.85 m
};
ScreenPose g_screen;
bool g_screenInit = false;
bool g_screenMoved = false; // the player has moved or sized it (a grab): the room survey leaves it be
// How wide the screen looks (its half width over its distance from the
// eyes): what room_fit sizes it to. The player's own from their last grab;
// 36 degrees across before one (what the user sized it to on 2026-10-18,
// 0.73 m at 1.12 m, from the first fit's 48). Written by the XR thread, read
// by the surface worker.
std::atomic<float> g_screenTan{0.325f};
// The room survey's screen (room_survey_apply), glided to over 0.8 s.
struct ScreenGlide {
  bool on = false;
  ScreenPose from, to;
  int64_t startNs = 0;
} g_screenGlide;
constexpr float kMinScreenWidth = 0.4f, kMaxScreenWidth = 8.f;
// One controller's reach along its laser while dragging the screen.
constexpr float kMinScreenDist = 0.3f, kMaxScreenDist = 10.f;

ScreenPose& screen_pose() {
  if (!g_screenInit) {
    g_screenInit = true;
    g_screen.pos = {0.f, env_float("AURORA_XR_SCREEN_Y", 0.f), -env_float("AURORA_XR_SCREEN_DISTANCE", 0.85f)};
    g_screen.width = env_float("AURORA_XR_SCREEN_WIDTH", 0.7f);
  }
  return g_screen;
}

// Screen height over width, from its swapchain.
float screen_aspect() {
  const auto& st = g_streams[kScreen];
  return st.width > 0 ? static_cast<float>(st.height) / static_cast<float>(st.width) : 9.f / 16.f;
}

XrQuaternionf qconj(const XrQuaternionf& q) { return {-q.x, -q.y, -q.z, q.w}; }

// The screen turned to face `eye`, upright.
XrQuaternionf facing(XrVector3f pos, XrVector3f eye) {
  const XrVector3f z = vnorm(eye - pos);
  XrVector3f x = vcross({0.f, 1.f, 0.f}, z);
  if (vlen(x) < 1e-3f)
    return {0.f, 0.f, 0.f, 1.f}; // straight above or below: keep it level
  x = vnorm(x);
  return quat_from_axes(x, vcross(z, x), z);
}

bool back_shown();

// The Back button's radius (meters), for the screen's size.
float back_radius(const ScreenPose& s) { return std::clamp(s.width * 0.018f, 0.012f, 0.02f); }

// The grab bar under the screen, in the screen's frame (meters): its
// center's height and half extents. It grows with the screen, within a
// hand's reach of sizes. While the Back button shows beside it, both sit
// lower if need be, centered on one line, so the button stays 1.2 cm clear
// of the picture.
struct BarRect {
  float y, hw, hh;
};
BarRect bar_rect(const ScreenPose& s) {
  const float hw = std::clamp(s.width * 0.1f, 0.05f, 0.2f);
  const float hh = std::clamp(s.width * 0.008f, 0.005f, 0.012f);
  const float bottom = -s.width * screen_aspect() * 0.5f;
  float y = bottom - hh * 3.f;
  if (back_shown())
    y = std::min(y, bottom - 0.012f - back_radius(s));
  return {y, hw, hh};
}

// A point in the screen's frame on (or near) the bar: 2 cm of slack around
// it, since it is thin.
bool on_bar(const ScreenPose& s, XrVector3f local) {
  const BarRect b = bar_rect(s);
  return std::abs(local.x) <= b.hw + 0.02f && std::abs(local.y - b.y) <= b.hh + 0.02f;
}

// The Back button left of the bar, for tracked hands (which have no B):
// its center and radius, in the screen's frame, level with the bar.
struct BackRect {
  float x, y, r;
};
BackRect back_rect(const ScreenPose& s) {
  const BarRect b = bar_rect(s);
  const float r = back_radius(s);
  return {-b.hw - r * 2.f, b.y, r};
}

// A point in the screen's frame on (or near) the Back button, with a
// centimeter of slack.
bool on_back(const ScreenPose& s, XrVector3f local) {
  const BackRect k = back_rect(s);
  return std::hypot(local.x - k.x, local.y - k.y) <= k.r + 0.01f;
}

// Touching the bar: on it and within a few centimeters of its surface.
bool in_bar_box(const ScreenPose& s, XrVector3f p) {
  const XrVector3f l = qrot(qconj(s.orientation), p - s.pos);
  return on_bar(s, l) && std::abs(l.z) <= 0.05f;
}

// Where the ray meets the screen's plane, from either side: meters along
// it and the point in the screen's frame. False: parallel, or behind.
bool hit_screen_plane(const ScreenPose& s, XrVector3f origin, XrVector3f dir, float& t, XrVector3f& local) {
  const XrQuaternionf inv = qconj(s.orientation);
  const XrVector3f o = qrot(inv, origin - s.pos), d = qrot(inv, dir);
  if (std::abs(d.z) < 1e-6f)
    return false;
  t = -o.z / d.z;
  if (t < 0.f)
    return false;
  local = o + d * t;
  return true;
}

// What the hands place: the arena while a fight is paused, the screen
// otherwise.
enum class Target { Arena, Screen };

struct Hand {
  bool valid = false;
  bool tracked = false;               // an articulated hand (pinches) rather than a controller (grip)
  XrVector3f origin{};                // the laser: from the controller, or from the hand's index knuckle
  XrVector3f dir{0.f, 0.f, -1.f};
  XrVector3f touch{};                 // what reaches into the arena: the pinch, or the controller's tip
  bool held = false, wasHeld = false; // grip or pinch closed, with hysteresis
  bool sel = false, wasSel = false;   // a controller's trigger pulled (a pinch is `held`), with hysteresis
  bool palmIn = false;                // a tracked hand's palm faces the head: not pointing
  bool inside = false;                // `touch` is in the arena's grab box (or on the screen's bar): a dot instead of a laser
  bool engaged = false;               // holding the arena
  bool direct = false;                // engaged from inside the box: drags by `touch`, not along the laser
  float hit = -1.f;                   // outside the box: meters along the laser to the arena, -1 = none
  // The screen only: what the laser is on, where on the picture (0..1,
  // x right, y down), and whether this hand is clicking the picture.
  enum class On { None, Picture, Bar, Back } on = On::None;
  float u = 0.f, v = 0.f;
  bool clicking = false;
  bool backing = false; // pressing the Back button
  // The room grade panel (paused fights): the laser is on it, or this hand
  // drags one of its sliders; then the arena isn't grabbed. Meters along the
  // laser to the panel.
  bool onPanel = false;
  float panelHit = -1.f;
};

struct Grab {
  std::array<Hand, 2> hands;
  Target target = Target::Arena;
  int oneHand = -1; // the hand dragging the target alone
  bool twoHands = false;
  ArenaPose start;
  ScreenPose screenStart;
  float dist = 0.f;  // one controller: the grabbed point's distance along its laser
  XrVector3f offset; // one hand: target position minus the grabbed point
  XrVector3f mid0;   // two hands: midpoint, span and heading at the start
  float span0 = 1.f, heading0 = 0.f;
};
Grab G;

// The Back button shows while a hand, rather than a controller, is seen:
// controllers have B.
bool back_shown() {
  return std::any_of(G.hands.begin(), G.hands.end(), [](const Hand& h) { return h.tracked; });
}

// Thumb and index tips closer than kPinchClose (meters) close a pinch;
// farther than kPinchOpen open it.
constexpr float kPinchClose = 0.02f, kPinchOpen = 0.035f;

// An articulated hand's pinch and laser, while hand tracking sees the hand
// (on Quest, whenever that controller is put down). False: use the
// controller. The laser runs from an estimated shoulder through the index
// knuckle, which stays put while the fingers pinch.
bool locate_pinch(int h, XrTime time, Hand& hand) {
  if (!B.handTrackers[h])
    return false;
  std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
  XrHandJointLocationsEXT locs{XR_TYPE_HAND_JOINT_LOCATIONS_EXT};
  locs.jointCount = static_cast<uint32_t>(joints.size());
  locs.jointLocations = joints.data();
  XrHandJointsLocateInfoEXT li{XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT};
  li.baseSpace = B.space;
  li.time = time;
  if (XR_FAILED(B.locateHandJoints(B.handTrackers[h], &li, &locs)) || !locs.isActive)
    return false;
  hand.tracked = true;
  const auto& thumb = joints[XR_HAND_JOINT_THUMB_TIP_EXT];
  const auto& index = joints[XR_HAND_JOINT_INDEX_TIP_EXT];
  const auto& knuckle = joints[XR_HAND_JOINT_INDEX_PROXIMAL_EXT];
  if (!(thumb.locationFlags & index.locationFlags & knuckle.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT))
    return true; // seen but lost this frame: not valid, so any grab lets go
  hand.valid = true;
  hand.touch = (thumb.pose.position + index.pose.position) * 0.5f;
  const XrVector3f fwd = qrot(B.headOrientation, {0.f, 0.f, -1.f});
  const XrVector3f right = vnorm({-fwd.z, 0.f, fwd.x});
  const XrVector3f shoulder = B.head + XrVector3f{0.f, -0.15f, 0.f} + right * (h == 0 ? -0.18f : 0.18f);
  hand.origin = knuckle.pose.position;
  hand.dir = vnorm(hand.origin - shoulder);
  const float d = vlen(index.pose.position - thumb.pose.position);
  hand.held = hand.held ? d < kPinchOpen : d < kPinchClose;
  // A palm turned toward the face isn't pointing (as on the Quest's own
  // menus): no laser, and its pinches do nothing, until it turns away. The
  // palm faces along its joint's -Y. A hand already holding or clicking
  // keeps going.
  const auto& palm = joints[XR_HAND_JOINT_PALM_EXT];
  if (palm.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) {
    const XrVector3f normal = qrot(palm.pose.orientation, {0.f, -1.f, 0.f});
    const float facing = vdot(normal, vnorm(B.head - palm.pose.position));
    hand.palmIn = hand.palmIn ? facing > 0.2f : facing > 0.45f;
  }
  if (hand.palmIn && !hand.engaged && !hand.clicking)
    hand.valid = false;
  return true;
}

void locate_hands(XrTime time) {
  for (int h = 0; h < 2; ++h) {
    Hand& hand = G.hands[h];
    const bool wasTracked = hand.tracked;
    hand.wasHeld = hand.held;
    hand.wasSel = hand.sel;
    hand.valid = false;
    if (!B.focused) {
      hand.held = false;
      hand.sel = false;
      continue;
    }
    if (!locate_pinch(h, time, hand)) {
      hand.tracked = false;
      if (wasTracked)
        hand.held = false; // picked up the controller: its grip starts open
      if (!B.aimSpace[h]) {
        hand.held = false;
        continue;
      }
      XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
      constexpr XrSpaceLocationFlags kValid =
          XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
      if (XR_SUCCEEDED(xrLocateSpace(B.aimSpace[h], B.space, time, &loc)) && (loc.locationFlags & kValid) == kValid) {
        hand.valid = true;
        hand.origin = loc.pose.position;
        hand.dir = vnorm(qrot(loc.pose.orientation, {0.f, 0.f, -1.f}));
        hand.touch = hand.origin;
      }
      const float grip = action_float(B.grab[h]);
      hand.held = hand.held ? grip > 0.35f : grip > 0.65f;
      const float trigger = action_float(h == 0 ? B.trigL : B.trigR);
      hand.sel = hand.sel ? trigger > 0.35f : trigger > 0.65f;
    } else if (!wasTracked) {
      hand.held = false; // put the controller down: the pinch starts open
    }
    if (hand.tracked || !hand.valid)
      hand.sel = false;
  }
}

void end_grab() {
  if (G.oneHand < 0 && !G.twoHands)
    return;
  G.oneHand = -1;
  G.twoHands = false;
  if (G.target == Target::Screen) {
    const ScreenPose& s = screen_pose();
    g_screenMoved = true;
    g_screenGlide.on = false;
    const float d = std::hypot(s.pos.x - B.head.x, s.pos.z - B.head.z);
    if (d > 0.2f)
      g_screenTan = std::clamp(0.5f * s.width / d, 0.1f, 1.f);
    Log.info("Screen placed at {:.2f},{:.2f},{:.2f}, width {:.2f} m ({:.0f} degrees across)", s.pos.x, s.pos.y, s.pos.z,
             s.width, 2.f * std::atan(g_screenTan.load()) * 57.2958f);
    return;
  }
  const ArenaPose a = arena_pose();
  int stage;
  float stageScale;
  {
    std::lock_guard lock{g_arenaMutex};
    stage = g_stage;
    stageScale = g_stageScale;
  }
  // The scale without the stage's own, as AURORA_XR_ARENA_SCALE takes it.
  Log.info("Arena for stage {} placed at {:.2f},{:.2f},{:.2f}, yaw {:.0f} deg, scale {:.4f} (stage's own x{:.2f})",
           stage, a.pos.x, a.pos.y, a.pos.z, a.yaw * 57.2958f, a.scale / stageScale, stageScale);
}

// What a hand drags alone: its pinch or controller tip, or the grabbed
// point on its laser.
XrVector3f drag_point(const Hand& hand) { return hand.direct ? hand.touch : hand.origin + hand.dir * G.dist; }

void begin_one_hand(int h, XrVector3f pos) {
  const Hand& hand = G.hands[h];
  G.oneHand = h;
  G.twoHands = false;
  G.start = arena_pose();
  G.screenStart = screen_pose();
  // The laser's hit, or (back from two hands with the laser off the target)
  // the point on the laser nearest the target.
  if (!hand.direct)
    G.dist = hand.hit >= 0.f ? hand.hit : std::max(vdot(pos - hand.origin, hand.dir), 0.1f);
  G.offset = pos - drag_point(hand);
}

void begin_two_hands() {
  const XrVector3f l = G.hands[0].touch, r = G.hands[1].touch;
  const XrVector3f span = r - l;
  G.oneHand = -1;
  G.twoHands = true;
  G.start = arena_pose();
  G.screenStart = screen_pose();
  G.mid0 = (l + r) * 0.5f;
  G.span0 = std::max(vlen(span), 0.05f);
  G.heading0 = std::atan2(-span.z, span.x);
}

// The screen while one hand drags it, or two resize it. It keeps facing
// the head.
void place_screen() {
  ScreenPose& s = screen_pose();
  if (G.oneHand >= 0) {
    const Hand& hand = G.hands[G.oneHand];
    if (!hand.direct && !hand.tracked) {
      // The stick reels it along the laser, about doubling the reach per
      // second at full tilt.
      static auto last = std::chrono::steady_clock::now();
      const auto now = std::chrono::steady_clock::now();
      const float dt = std::min(std::chrono::duration<float>(now - last).count(), 0.05f);
      last = now;
      const float y = B.stickY[G.oneHand];
      if (std::abs(y) > 0.2f)
        G.dist = std::clamp(G.dist * std::exp(y * dt * 0.7f), kMinScreenDist, kMaxScreenDist);
    }
    s.pos = drag_point(hand) + G.offset;
  } else {
    const XrVector3f lp = G.hands[0].touch, rp = G.hands[1].touch;
    const float ratio = vlen(rp - lp) / G.span0;
    s.width = std::clamp(G.screenStart.width * ratio, kMinScreenWidth, kMaxScreenWidth);
    s.pos = G.screenStart.pos + ((lp + rp) * 0.5f - G.mid0);
  }
  s.orientation = facing(s.pos, B.head);
}

// Engaging and letting go, the same for both targets: one engaged hand
// drags alone, two scale (and turn the arena). False: nothing held.
bool update_holds(XrVector3f targetPos) {
  const bool l = G.hands[0].engaged, r = G.hands[1].engaged;
  if (l && r) {
    if (!G.twoHands)
      begin_two_hands();
  } else if (l || r) {
    // Letting go of one of two hands carries on with the other alone.
    if (G.oneHand != (l ? 0 : 1))
      begin_one_hand(l ? 0 : 1, targetPos);
  } else {
    end_grab();
    return false;
  }
  return true;
}

void engage(Hand& hand) {
  hand.engaged = true;
  hand.direct = hand.inside;
}

// The picture's point in the game's frame (0..1), off its letterbox; false
// when it is in the bars.
bool picture_to_frame(float u, float v, float& x, float& y) {
  const auto& st = g_streams[kScreen];
  const uint32_t cw = g_contentW, ch = g_contentH;
  if (st.width == 0 || st.height == 0 || cw == 0 || ch == 0) {
    x = u;
    y = v;
    return true;
  }
  const auto vp = webgpu::calculate_present_viewport(st.width, st.height, cw, ch);
  x = (u * static_cast<float>(st.width) - vp.left) / vp.width;
  y = (v * static_cast<float>(st.height) - vp.top) / vp.height;
  return x >= 0.f && x <= 1.f && y >= 0.f && y <= 1.f;
}

// The virtual screen, per display frame. A laser on the picture points at
// it for the game and clicks with the trigger or a pinch; the bar under it
// (pointed at or touched) takes hold of it with a grip, trigger or pinch,
// and while one hand holds it the other joins by pressing with its laser
// anywhere on the screen, to resize it.
void update_screen(bool active) {
  const ScreenPose& screen = screen_pose();
  const float hw = screen.width * 0.5f, hh = screen.width * screen_aspect() * 0.5f;
  const bool back = back_shown();
  for (Hand& hand : G.hands) {
    hand.inside = hand.valid && in_bar_box(screen, hand.touch);
    hand.hit = -1.f;
    hand.on = Hand::On::None;
    float t;
    XrVector3f l;
    // A laser on the dev tools panel is on it, not on the screen behind.
    if (hand.valid && !hand.inside && !hand.onPanel && hit_screen_plane(screen, hand.origin, hand.dir, t, l)) {
      if (std::abs(l.x) <= hw && std::abs(l.y) <= hh) {
        hand.on = Hand::On::Picture;
        hand.u = l.x / (2.f * hw) + 0.5f;
        hand.v = 0.5f - l.y / (2.f * hh);
      } else if (on_bar(screen, l)) {
        hand.on = Hand::On::Bar;
      } else if (back && on_back(screen, l)) {
        hand.on = Hand::On::Back;
      }
      if (hand.on != Hand::On::None)
        hand.hit = t;
    }
    if (hand.inside)
      hand.on = Hand::On::Bar;
  }
  B.reelHand = -1;
  const auto down = [](const Hand& h) { return h.valid && (h.held || h.sel); };
  const auto pressed = [](const Hand& h) { return h.valid && ((h.held && !h.wasHeld) || (h.sel && !h.wasSel)); };
  // Clicks are the trigger's or a pinch's; a controller's grip only takes
  // hold of the screen.
  const auto clickDown = [](const Hand& h) { return h.valid && (h.tracked ? h.held : h.sel); };
  const auto clickPressed = [](const Hand& h) {
    return h.valid && (h.tracked ? h.held && !h.wasHeld : h.sel && !h.wasSel);
  };
  for (Hand& hand : G.hands) {
    hand.engaged &= active && down(hand);
    hand.clicking &= active && clickDown(hand);
    hand.backing &= active && back && clickDown(hand);
  }
  if (active) {
    for (Hand& hand : G.hands)
      if (pressed(hand) && hand.on == Hand::On::Bar)
        engage(hand);
    for (int h = 0; h < 2; ++h) {
      Hand& hand = G.hands[h];
      if (!pressed(hand) || hand.engaged)
        continue;
      if (G.hands[1 - h].engaged && hand.on != Hand::On::None) {
        hand.clicking = false;
        engage(hand);
      } else if (hand.on == Hand::On::Picture && clickPressed(hand)) {
        hand.clicking = true;
      } else if (hand.on == Hand::On::Back && clickPressed(hand)) {
        hand.backing = true;
      }
    }
  }
  // The hand pointing for the game: a clicking one, or else the one that
  // already was while it stays on the picture, or else either on it.
  static int pointer = -1;
  {
    const auto pointing = [](const Hand& h) { return h.on == Hand::On::Picture && !h.engaged; };
    int next = -1;
    for (int h = 0; h < 2; ++h)
      if (G.hands[h].clicking && (next < 0 || h == pointer))
        next = h;
    if (next < 0 && pointer >= 0 && pointing(G.hands[pointer]))
      next = pointer;
    for (int h = 0; h < 2 && next < 0; ++h)
      if (pointing(G.hands[h]))
        next = h;
    pointer = next;
  }
  {
    ScreenPointer p;
    if (active && pointer >= 0 && G.oneHand < 0 && !G.twoHands) {
      const Hand& hand = G.hands[pointer];
      float x, y;
      // A click keeps the pointer even past the picture's edge, held there.
      p.valid = picture_to_frame(hand.u, hand.v, x, y) || hand.clicking;
      p.x = std::clamp(x, 0.f, 1.f);
      p.y = std::clamp(y, 0.f, 1.f);
      p.pressed = hand.clicking;
    }
    std::lock_guard lock{g_padMutex};
    g_screenPointer = p;
    g_screenBack = std::any_of(G.hands.begin(), G.hands.end(), [](const Hand& h) { return h.backing; });
  }
  if (!active) {
    end_grab();
    return;
  }
  if (!update_holds(screen.pos))
    return;
  place_screen();
  if (G.oneHand >= 0 && !G.hands[G.oneHand].direct && !G.hands[G.oneHand].tracked)
    B.reelHand = G.oneHand;
}

// Per display frame. `active`: the target shows on the headset and can be
// placed (a paused fight's arena, or the screen).
void update_grab(Target target, bool active) {
  if (target != G.target) {
    for (Hand& hand : G.hands) {
      hand.engaged = false;
      hand.clicking = false;
      hand.backing = false;
      hand.on = Hand::On::None;
    }
    end_grab();
    G.target = target;
  }
  if (target == Target::Screen) {
    update_screen(active);
    return;
  }
  {
    std::lock_guard lock{g_padMutex};
    g_screenPointer = {};
    g_screenBack = false;
  }
  ArenaPose a = arena_pose();
  for (Hand& hand : G.hands) {
    hand.inside = hand.valid && in_grab_box(a, hand.touch);
    hand.hit = hand.valid && !hand.inside ? hit_arena(a, hand.origin, hand.dir) : -1.f;
  }
  B.reelHand = -1;
  if (!active) {
    for (Hand& hand : G.hands)
      hand.engaged = false;
    end_grab();
    return;
  }
  const auto pressed = [](const Hand& h) { return h.valid && h.held && !h.wasHeld && !h.onPanel; };
  for (Hand& hand : G.hands)
    hand.engaged &= hand.valid && hand.held;
  // A press inside the box or with the laser on the arena takes hold of it;
  // with one hand already holding, a press of the other joins it anywhere
  // (also when both press together). Inside the box the hand drags by its
  // touch point, outside along its laser, until it lets go.
  for (Hand& hand : G.hands)
    if (pressed(hand) && (hand.inside || hand.hit >= 0.f))
      engage(hand);
  for (int h = 0; h < 2; ++h)
    if (pressed(G.hands[h]) && !G.hands[h].engaged && G.hands[1 - h].engaged)
      engage(G.hands[h]);
  if (!update_holds(a.pos))
    return;
  if (G.oneHand >= 0) {
    a.pos = drag_point(G.hands[G.oneHand]) + G.offset;
  } else {
    const XrVector3f lp = G.hands[0].touch, rp = G.hands[1].touch;
    const XrVector3f span = rp - lp;
    a.scale = std::clamp(G.start.scale * vlen(span) / G.span0, kMinArenaScale, kMaxArenaScale);
    const float ratio = a.scale / G.start.scale;
    const float turn = std::atan2(-span.z, span.x) - G.heading0;
    a.yaw = G.start.yaw + turn;
    a.pos = (lp + rp) * 0.5f + rot_y((G.start.pos - G.mid0) * ratio, turn);
  }
  set_arena_pose(a);
}

// ---------------------------------------------------------------- XR thread: lasers

// Where the HUD quad goes, in the room. It moves, turns and scales with the
// arena.
void hud_placement(XrPosef& pose, XrExtent2Df& size) {
  const auto& hud = g_streams[kHud];
  const ArenaPose arena = arena_pose();
  // The HUD follows the player's resizes, not the stage's own scale.
  float k;
  {
    std::lock_guard lock{g_arenaMutex};
    k = arena.scale / (g_defaultArena.scale * g_stageScale);
  }
  const float width = env_float("AURORA_XR_HUD_WIDTH", 0.5f) * k;
  size = {width, hud.width > 0 ? width * static_cast<float>(hud.height) / static_cast<float>(hud.width) : width};
  // Above the arena's back edge, like a scoreboard. Quad layers draw over
  // the 3D view, so the HUD's bottom edge (where the damage meters are)
  // clears the stage's highest floor by about a fighter's height; on low
  // stages it keeps AURORA_XR_HUD_HEIGHT (its center above the arena).
  float height = env_float("AURORA_XR_HUD_HEIGHT", 0.3f) * k;
  float top, centerY;
  {
    std::lock_guard lock{g_arenaMutex};
    top = g_stageTop;
    centerY = g_arenaCenter[1];
  }
  if (!std::isnan(top) && !std::getenv("AURORA_XR_HUD_HEIGHT")) {
    const float clear = env_float("AURORA_XR_HUD_CLEARANCE", 40.f); // game units
    height = std::max(height, (top + clear - centerY) * arena.scale + size.height * 0.5f);
  }
  pose.orientation = yaw_quat(arena.yaw);
  pose.position = arena.pos + rot_y({0.f, height, -0.15f * k}, arena.yaw);
}

// AURORA_XR_DUMP: the left eye's view and the HUD quad, so a dump of the 3D
// image can be composited with the HUD offline (one line each:
// position xyz, orientation xyzw, then fov angles or quad size).
void write_dump_layout(const std::array<XrView, 2>& views) {
  XrPosef hud{};
  XrExtent2Df size{};
  hud_placement(hud, size);
  if (FILE* f = std::fopen((B.dumpDir + "/xr_layout.txt").c_str(), "w")) {
    const XrPosef& e = views[0].pose;
    const XrFovf& fov = views[0].fov;
    std::fprintf(f, "eye %f %f %f %f %f %f %f %f %f %f %f\n", e.position.x, e.position.y, e.position.z,
                 e.orientation.x, e.orientation.y, e.orientation.z, e.orientation.w, fov.angleLeft, fov.angleRight,
                 fov.angleUp, fov.angleDown);
    std::fprintf(f, "hud %f %f %f %f %f %f %f %f %f\n", hud.position.x, hud.position.y, hud.position.z,
                 hud.orientation.x, hud.orientation.y, hud.orientation.z, hud.orientation.w, size.width, size.height);
    std::fclose(f);
  }
}

// Static swapchain uploads whose copies have finished: free what they used.
// `all`: wait for the rest too (teardown, after the queue is idle).
void free_finished_uploads(bool all = false) {
  std::erase_if(B.uploads, [all](const auto& u) {
    if (!all && vkGetFenceStatus(B.dev, u.fence) != VK_SUCCESS)
      return false;
    vkDestroyFence(B.dev, u.fence, nullptr);
    vkFreeCommandBuffers(B.dev, B.pool, 1, &u.cmd);
    vkDestroyBuffer(B.dev, u.buf, nullptr);
    vkFreeMemory(B.dev, u.mem, nullptr);
    return true;
  });
}

// Fill a one-image swapchain once; the runtime keeps showing it. Without
// pixels (rgba empty) it is cleared to zero instead, in `format` with
// `layers` layers (else the swapchains' own format, one layer).
bool create_static_swapchain(XrSwapchain& out, uint32_t w, uint32_t h, const std::vector<uint8_t>& rgba,
                             int64_t format = 0, uint32_t layers = 1) {
  XrSwapchainCreateInfo ci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
  ci.createFlags = XR_SWAPCHAIN_CREATE_STATIC_IMAGE_BIT;
  // Color attachment: released in that layout, as OpenXR expects of color images.
  ci.usageFlags =
      XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
  ci.format = format != 0 ? format : B.swapFormat;
  ci.sampleCount = 1;
  ci.width = w;
  ci.height = h;
  ci.faceCount = 1;
  ci.arraySize = layers;
  ci.mipCount = 1;
  if (XR_FAILED(xrCreateSwapchain(B.session, &ci, &out))) {
    ci.createFlags = 0;
    XR_TRY(xrCreateSwapchain(B.session, &ci, &out));
  }
  uint32_t n = 0;
  XR_TRY(xrEnumerateSwapchainImages(out, 0, &n, nullptr));
  std::vector<XrSwapchainImageVulkan2KHR> imgs(n, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR});
  XR_TRY(xrEnumerateSwapchainImages(out, n, &n, reinterpret_cast<XrSwapchainImageBaseHeader*>(imgs.data())));

  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  const auto upload = [&]() -> bool {
    if (!rgba.empty()) {
      VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      bci.size = rgba.size();
      bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      VK_TRY(vkCreateBuffer(B.dev, &bci, nullptr, &buf));
      VkMemoryRequirements req;
      vkGetBufferMemoryRequirements(B.dev, buf, &req);
      VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      mai.allocationSize = req.size;
      mai.memoryTypeIndex =
          find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      if (mai.memoryTypeIndex == UINT32_MAX)
        return false;
      VK_TRY(vkAllocateMemory(B.dev, &mai, nullptr, &mem));
      VK_TRY(vkBindBufferMemory(B.dev, buf, mem, 0));
      void* p = nullptr;
      VK_TRY(vkMapMemory(B.dev, mem, 0, rgba.size(), 0, &p));
      std::memcpy(p, rgba.data(), rgba.size());
      vkUnmapMemory(B.dev, mem);
    }

    uint32_t index = 0;
    XrSwapchainImageAcquireInfo ai{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    XR_TRY(xrAcquireSwapchainImage(out, &ai, &index));
    XrSwapchainImageWaitInfo wi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wi.timeout = XR_INFINITE_DURATION;
    XR_TRY(xrWaitSwapchainImage(out, &wi));

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = B.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_TRY(vkAllocateCommandBuffers(B.dev, &cai, &cmd));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_TRY(vkBeginCommandBuffer(cmd, &cbi));
    VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    bar.srcQueueFamilyIndex = bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image = imgs[index].image;
    bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &bar);
    if (rgba.empty()) {
      const VkClearColorValue zero{};
      vkCmdClearColorImage(cmd, imgs[index].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1,
                           &bar.subresourceRange);
    } else {
      VkBufferImageCopy region{};
      region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.imageExtent = {w, h, 1};
      vkCmdCopyBufferToImage(cmd, buf, imgs[index].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }
    // The layout OpenXR expects a released color swapchain image in.
    bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar.dstAccessMask = 0;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &bar);
    VK_TRY(vkEndCommandBuffer(cmd));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_TRY(vkCreateFence(B.dev, &fci, nullptr, &fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    // Not waited on: this queue also carries the stream releases, which
    // wait on the render worker, which can wait on this thread. The staging
    // buffer goes once the fence signals.
    VK_TRY(vkQueueSubmit(B.queue, 1, &si, fence));
    XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    XR_TRY(xrReleaseSwapchainImage(out, &ri));
    B.uploads.push_back({buf, mem, cmd, fence});
    buf = VK_NULL_HANDLE;
    mem = VK_NULL_HANDLE;
    cmd = VK_NULL_HANDLE;
    fence = VK_NULL_HANDLE;
    return true;
  };
  const bool ok = upload();
  if (fence)
    vkDestroyFence(B.dev, fence, nullptr);
  if (cmd)
    vkFreeCommandBuffers(B.dev, B.pool, 1, &cmd);
  if (buf)
    vkDestroyBuffer(B.dev, buf, nullptr);
  if (mem)
    vkFreeMemory(B.dev, mem, nullptr);
  return ok;
}

constexpr uint32_t kBeamW = 16, kBeamH = 4, kDotSize = 64, kBarW = 256, kBarH = 32, kBackSize = 64;

// White, premultiplied, soft-edged: a beam (alpha across its width) and a
// round dot. The lasers tint them per frame (color scale/bias layers).
bool create_pointer_textures() {
  const bool srgb = B.swapFormat == VK_FORMAT_R8G8B8A8_SRGB;
  const auto texel = [srgb](std::vector<uint8_t>& px, size_t i, float alpha) {
    alpha = std::clamp(alpha, 0.f, 1.f);
    const float c = srgb ? (alpha <= 0.0031308f ? alpha * 12.92f : 1.055f * std::pow(alpha, 1.f / 2.4f) - 0.055f)
                         : alpha;
    px[i * 4 + 0] = px[i * 4 + 1] = px[i * 4 + 2] = static_cast<uint8_t>(std::lround(c * 255.f));
    px[i * 4 + 3] = static_cast<uint8_t>(std::lround(alpha * 255.f));
  };
  std::vector<uint8_t> beam(kBeamW * kBeamH * 4);
  for (uint32_t y = 0; y < kBeamH; ++y)
    for (uint32_t x = 0; x < kBeamW; ++x) {
      const float u = std::abs((x + 0.5f) / kBeamW * 2.f - 1.f);
      texel(beam, y * kBeamW + x, 1.f - u * u);
    }
  std::vector<uint8_t> dot(kDotSize * kDotSize * 4);
  for (uint32_t y = 0; y < kDotSize; ++y)
    for (uint32_t x = 0; x < kDotSize; ++x) {
      const float dx = (x + 0.5f) / kDotSize * 2.f - 1.f, dy = (y + 0.5f) / kDotSize * 2.f - 1.f;
      const float r = std::sqrt(dx * dx + dy * dy);
      texel(dot, y * kDotSize + x, std::clamp((1.f - r) / 0.2f, 0.f, 1.f));
    }
  // The bar: a pill, round at the ends.
  std::vector<uint8_t> bar(kBarW * kBarH * 4);
  for (uint32_t y = 0; y < kBarH; ++y)
    for (uint32_t x = 0; x < kBarW; ++x) {
      const float r = kBarH * 0.5f;
      const float px = std::clamp(x + 0.5f, r, kBarW - r), py = r;
      const float d = std::hypot(x + 0.5f - px, y + 0.5f - py);
      texel(bar, y * kBarW + x, std::clamp(r - d, 0.f, 1.5f) / 1.5f);
    }
  // Back: a see-through disc with a solid left chevron on it.
  std::vector<uint8_t> back(kBackSize * kBackSize * 4);
  for (uint32_t y = 0; y < kBackSize; ++y)
    for (uint32_t x = 0; x < kBackSize; ++x) {
      const float dx = (x + 0.5f) / kBackSize * 2.f - 1.f, dy = (y + 0.5f) / kBackSize * 2.f - 1.f;
      const float disc = std::clamp((1.f - std::hypot(dx, dy)) / 0.06f, 0.f, 1.f);
      // Two strokes from the tip (-0.3, 0) out to (0.15, +-0.4).
      float d = 1e9f;
      for (const float sy : {-1.f, 1.f}) {
        const float ax = -0.3f, ay = 0.f, bx = 0.15f, by = 0.4f * sy;
        const float t = std::clamp(((dx - ax) * (bx - ax) + (dy - ay) * (by - ay)) /
                                       ((bx - ax) * (bx - ax) + (by - ay) * (by - ay)),
                                   0.f, 1.f);
        d = std::min(d, std::hypot(dx - (ax + t * (bx - ax)), dy - (ay + t * (by - ay))));
      }
      const float arrow = std::clamp((0.11f - d) / 0.04f, 0.f, 1.f);
      texel(back, y * kBackSize + x, std::max(disc * 0.18f, arrow));
    }
  return create_static_swapchain(B.beamSwapchain, kBeamW, kBeamH, beam) &&
         create_static_swapchain(B.dotSwapchain, kDotSize, kDotSize, dot) &&
         create_static_swapchain(B.barSwapchain, kBarW, kBarH, bar) &&
         create_static_swapchain(B.backSwapchain, kBackSize, kBackSize, back);
}

// a beam and a dot per hand, the screen's bar and Back button, the dev tools
// panel's bar
constexpr size_t kMaxPointerLayers = 7;

struct PointerLayers {
  std::array<XrCompositionLayerQuad, kMaxPointerLayers> quads;
  std::array<XrCompositionLayerColorScaleBiasKHR, kMaxPointerLayers> tints;
  uint32_t count = 0;
};

void add_pointer_quad(PointerLayers& out, XrSwapchain swapchain, uint32_t w, uint32_t h, XrVector3f center,
                      XrQuaternionf orientation, XrExtent2Df size, XrColor4f color) {
  auto& q = out.quads[out.count];
  q = {XR_TYPE_COMPOSITION_LAYER_QUAD};
  q.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
  q.space = B.space;
  q.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
  q.subImage.swapchain = swapchain;
  q.subImage.imageRect = {{0, 0}, {static_cast<int32_t>(w), static_cast<int32_t>(h)}};
  q.pose = {orientation, center};
  q.size = size;
  if (B.hasColorScaleBias) {
    auto& t = out.tints[out.count];
    t = {XR_TYPE_COMPOSITION_LAYER_COLOR_SCALE_BIAS_KHR};
    // Premultiplied: the color scales with the alpha.
    t.colorScale = {color.r * color.a, color.g * color.a, color.b * color.a, color.a};
    q.next = &t;
  }
  ++out.count;
}

// The screen's grab bar: faint, brighter while a laser or hand is on it,
// tinted while the screen is held.
void add_screen_bar(PointerLayers& out) {
  if (!B.barSwapchain)
    return;
  const ScreenPose& s = screen_pose();
  const BarRect b = bar_rect(s);
  const bool held = G.target == Target::Screen && (G.oneHand >= 0 || G.twoHands);
  const bool hover = std::any_of(G.hands.begin(), G.hands.end(), [](const Hand& h) { return h.on == Hand::On::Bar; });
  const XrColor4f color = held    ? XrColor4f{0.45f, 0.85f, 1.f, 1.f}
                          : hover ? XrColor4f{1.f, 1.f, 1.f, 0.95f}
                                  : XrColor4f{0.85f, 0.88f, 0.92f, 0.55f};
  add_pointer_quad(out, B.barSwapchain, kBarW, kBarH, s.pos + qrot(s.orientation, {0.f, b.y, 0.f}), s.orientation,
                   {b.hw * 2.f, b.hh * 2.f}, color);
  if (!B.backSwapchain || !back_shown())
    return;
  const BackRect k = back_rect(s);
  const bool pressing = std::any_of(G.hands.begin(), G.hands.end(), [](const Hand& h) { return h.backing; });
  const bool over = std::any_of(G.hands.begin(), G.hands.end(), [](const Hand& h) { return h.on == Hand::On::Back; });
  const XrColor4f backColor = pressing ? XrColor4f{0.45f, 0.85f, 1.f, 1.f}
                              : over   ? XrColor4f{1.f, 1.f, 1.f, 0.9f}
                                       : XrColor4f{0.85f, 0.88f, 0.92f, 0.4f};
  add_pointer_quad(out, B.backSwapchain, kBackSize, kBackSize, s.pos + qrot(s.orientation, {k.x, k.y, 0.f}),
                   s.orientation, {k.r * 2.f, k.r * 2.f}, backColor);
}

// `onTargetOnly`: skip the hands that neither point at the target nor hold
// it; `panelOnly`: skip all but those on the dev tools panel. Adds to what
// `out` has.
void build_pointer_layers(PointerLayers& out, bool onTargetOnly = false, bool panelOnly = false) {
  if (!B.beamSwapchain || !B.dotSwapchain)
    return;
  for (int h = 0; h < 2; ++h) {
    const Hand& hand = G.hands[h];
    if (!hand.valid || (onTargetOnly && !hand.engaged && !hand.clicking && !hand.inside && hand.hit < 0.f) ||
        (panelOnly && !hand.onPanel))
      continue;
    const XrColor4f color = hand.engaged                     ? XrColor4f{1.f, 0.8f, 0.3f, 1.f}
                            : hand.clicking                  ? XrColor4f{1.f, 1.f, 1.f, 1.f}
                            : hand.inside || hand.hit >= 0.f ? XrColor4f{0.45f, 0.85f, 1.f, 1.f}
                                                             : XrColor4f{0.85f, 0.9f, 1.f, 0.5f};
    // A laser on the panel shows, though the hand is in the arena's box.
    if (!hand.onPanel && (hand.engaged ? hand.direct : hand.inside)) {
      // In the box: a dot at the touch point, facing the head.
      const XrVector3f dz = vnorm(B.head - hand.touch);
      const XrVector3f dx = vnorm(vcross({0.f, 1.f, 0.f}, dz));
      if (vlen(dx) > 0.5f)
        add_pointer_quad(out, B.dotSwapchain, kDotSize, kDotSize, hand.touch, quat_from_axes(dx, vcross(dz, dx), dz),
                         {0.012f, 0.012f}, color);
      continue;
    }
    const bool dragging = G.oneHand == h;
    const float length = dragging ? G.dist : hand.hit >= 0.f ? hand.hit : 1.f;
    // The beam: a thin quad along the laser, turned about it to face the head.
    const XrVector3f end = hand.origin + hand.dir * length;
    const XrVector3f center = hand.origin + hand.dir * (length * 0.5f);
    const XrVector3f toHead = B.head - center;
    const XrVector3f z = vnorm(toHead - hand.dir * vdot(toHead, hand.dir));
    if (vlen(z) > 0.5f)
      add_pointer_quad(out, B.beamSwapchain, kBeamW, kBeamH, center,
                       quat_from_axes(vcross(hand.dir, z), hand.dir, z), {0.004f, length}, color);
    if (hand.hit < 0.f && !dragging)
      continue;
    // The dot where the laser meets the target, facing the head; on the
    // screen's picture a smaller one, smaller still while it clicks.
    const XrVector3f dz = vnorm(B.head - end);
    const XrVector3f dx = vnorm(vcross({0.f, 1.f, 0.f}, dz));
    const float size = hand.onPanel ? 0.01f : hand.on != Hand::On::Picture ? 0.02f : hand.clicking ? 0.01f : 0.014f;
    if (vlen(dx) > 0.5f)
      add_pointer_quad(out, B.dotSwapchain, kDotSize, kDotSize, end, quat_from_axes(dx, vcross(dz, dx), dz),
                       {size, size}, color);
  }
}

// ---------------------------------------------------------------- XR worker: the surface under the stage
//
// In mixed reality the stage casts a soft shadow on the real surface under it
// (docs/xr-raycast-surface-plan.md): a quad layer between passthrough and the
// 3D view, so it costs the eye pass nothing, the compositor keeps it on the
// table at display rate, and the stage's own pixels cover it where it sits.
// Where that surface is comes from XR_META_environment_raycast (Quest 3 and
// 3S, with the spatial data permission, USE_SCENE): nine rays straight down
// through the stage's footprint against the headset's live depth, filtered
// for a level surface, for the player's own hands, and for agreement across
// rays and over time. A worker thread owns the raycaster and casts only while
// the stage is being placed (the hold before a fight, a grab in a pause, the
// Place Stage scene), for half a second after it's let go, and for a second
// when a stage appears without a hold (netplay, a stage change, a recenter);
// then the estimate stays frozen for the fight. Nothing waits on it: the XR
// thread only copies its snapshot as it builds the layers. Without the
// raycaster (Quest 2 and Pro, the permission denied), the floor from
// XR_EXT_local_floor stands in, and without either there is no shadow: one at
// a guessed height would say the stage floats where it doesn't.
//
// The shadow's strength is a disc's ambient occlusion, R²/(R²+h²) for a
// footprint of effective radius R at a gap h over the surface: dark when the
// stage rests on the desk, soft a hand over it, gone a meter over the floor.
// It fades with how much of the footprint is over the surface (a stage
// hanging off a table's edge), and widens with the gap.
//
// AURORA_XR_RAYCAST=0 leaves the raycaster off and AURORA_XR_LOCAL_FLOOR=0
// the floor; AURORA_XR_SURFACE_SHADOW=0 hides the shadow while the surface is
// still measured (blind A/B); AURORA_XR_SHADOW_ALPHA (0.7) is its opacity on
// contact, AURORA_XR_SHADOW_SPREAD (0.5) how fast it widens with the gap.
// AURORA_XR_SURFACE_LOG=1 logs the casts once a second while they run (ray
// statuses, the consensus height, its spread and coverage, the shadow it
// makes) and the rays' timing every 10 s.

constexpr int64_t kMsNs = 1000000;

struct SurfaceSettings {
  bool shadow = true, log = false;
  float alpha = 0.7f, spread = 0.5f;
};

const SurfaceSettings& surface_settings() {
  static const SurfaceSettings s = [] {
    SurfaceSettings s;
    s.shadow = env_flag("AURORA_XR_SURFACE_SHADOW", !g_config.xrNoStageShadow); // the launcher's Stage shadow
    s.log = env_flag("AURORA_XR_SURFACE_LOG", false);
    s.alpha = std::clamp(env_float("AURORA_XR_SHADOW_ALPHA", 0.7f), 0.f, 1.f);
    s.spread = std::max(env_float("AURORA_XR_SHADOW_SPREAD", 0.5f), 0.f);
    return s;
  }();
  return s;
}

// What the worker found, for the XR thread.
struct SurfaceEstimate {
  enum class State { Unknown, Known, Frozen };
  enum class Source { Raycast, Floor };
  State state = State::Unknown; // Known: measured now; Frozen: held since the last casts
  Source source = Source::Raycast;
  float y = 0.f;        // the surface's height (meters, LOCAL)
  float coverage = 0.f; // the share of the footprint's rays that found it
  float sigma = 0.f;    // its spread over the casts that settled it (meters)
};

const char* surface_state_name(SurfaceEstimate::State s) {
  return s == SurfaceEstimate::State::Unknown ? "unknown" : s == SurfaceEstimate::State::Known ? "known" : "frozen";
}

// What the room survey found (room_survey_*): a desk or table in front of
// and below the eyes, the floor, and how far the room is clear ahead.
// Heights in LOCAL; distances level from the head, meters.
struct RoomSurvey {
  enum class Kind { Unknown, Desk, Couch, Room };
  bool done = false;
  Kind kind = Kind::Unknown;
  bool desk = false, floor = false;
  float deskY = 0.f, deskNear = 0.f, deskFar = 0.f; // its height, and its near and far reach
  float floorY = 0.f;
  float clearance = -1.f; // to the nearest wall or obstacle ahead; -1: unknown; kRoomReach: open
  float far = -1.f;       // to the farthest wall ahead (kRoomReach: open; -1: unknown)
  float headY = 0.f, yaw = 0.f;
  XrVector3f head{};
  int rays = 0, informed = 0, points = 0;
  // Where the stage and the screen fit (room_fit), when the room was seen.
  bool fitted = false;
  float stageDist = 0.f, stageY = 0.f, stageScale = 0.f; // ahead of the head along LOCAL -z
  bool screenFitted = false;
  float screenDist = 0.f, screenWidth = 0.f; // ahead along the head's yaw
};
const char* room_kind_name(RoomSurvey::Kind k) {
  return k == RoomSurvey::Kind::Desk ? "desk" : k == RoomSurvey::Kind::Couch ? "couch" : k == RoomSurvey::Kind::Room ? "room" : "unknown";
}

// Between the XR thread (and the game's setters) and the worker.
struct SurfaceShared {
  std::mutex mutex;
  std::condition_variable cv;
  SurfaceEstimate estimate;         // the worker's latest
  bool surveyWanted = false;        // a room survey: cast from the head until it's done
  bool surveyFresh = false;         // start it over: what it found so far is gone (room_resurvey)
  XrPosef head{{0.f, 0.f, 0.f, 1.f}, {}};
  bool headValid = false;
  RoomSurvey survey;                // the worker's, once done
  bool placing = false;             // the stage is being placed: cast all along
  int64_t burstUntilNs = 0;         // cast until then (monotonic_ns)
  bool reset = false;               // forget the estimate first (another stage, a recenter)
  bool stop = false;
  XrTime displayTime = 0;           // the latest predicted display time, for the rays
  std::array<XrVector3f, 2> hands{}; // each hand's or controller's touch point
  std::array<bool, 2> handValid{};
  // Placing the board on a surface (board_place_frame): each hand's laser,
  // cast against the room about 45 times a second, and what it hit.
  bool pointing = false;
  std::array<XrVector3f, 2> rayOrigin{}, rayDir{};
  std::array<bool, 2> rayValid{};
  struct PointHit {
    bool hit = false, level = false; // something within reach; a level surface
    XrVector3f p{};
    int64_t ns = 0; // monotonic_ns; 0: never cast
  };
  std::array<PointHit, 2> pointHits{};
  bool casterReady = false, casterGone = false; // the raycaster is up; it never will be
};
SurfaceShared g_surface;
std::thread g_surfaceThread;

// Any thread: cast for `burstMs` (0: no burst), and with `reset` forget what
// was found first.
void surface_request(int burstMs, bool reset) {
  std::lock_guard lock{g_surface.mutex};
  g_surface.reset |= reset;
  if (burstMs > 0)
    g_surface.burstUntilNs = std::max(g_surface.burstUntilNs, monotonic_ns() + burstMs * kMsNs);
  g_surface.cv.notify_one();
}

// The arena and the stage's footprint, as one consistent copy.
struct ArenaSnapshot {
  ArenaPose pose;
  std::array<float, 3> center{};
  float zoom = 1.f;
  Footprint footprint;
  int stage = -1;
};

ArenaSnapshot arena_snapshot() {
  std::lock_guard lock{g_arenaMutex};
  init_arena();
  return {g_arena, g_arenaCenter, g_stageZoom, g_footprint, g_stage};
}

// A game point (game units) in the room.
XrVector3f arena_point(const ArenaSnapshot& a, float x, float y, float z) {
  const XrVector3f g{x - a.center[0], y - a.center[1], z - a.center[2]};
  return a.pose.pos + rot_y(g * (a.pose.scale * a.zoom), a.pose.yaw);
}

float smoothstep(float e0, float e1, float x) {
  const float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
  return t * t * (3.f - 2.f * t);
}

// The footprint's shadow on a surface at height y (LOCAL) with `coverage` of
// it over that surface: where its middle is, its size there (meters), the
// gap from the stage's underside down to it, the footprint's effective
// radius and the shadow's opacity (0: none to show).
struct ShadowShape {
  XrVector3f center{};
  float yaw = 0.f, width = 0.f, depth = 0.f, gap = 0.f, radius = 0.f, alpha = 0.f;
};

ShadowShape shadow_shape(const ArenaSnapshot& a, float y, float coverage) {
  const Footprint& f = a.footprint;
  const float s = a.pose.scale * a.zoom;
  const XrVector3f under = arena_point(a, (f.xmin + f.xmax) * 0.5f, f.ybottom, (f.zmin + f.zmax) * 0.5f);
  ShadowShape sh;
  sh.center = {under.x, y, under.z};
  sh.yaw = a.pose.yaw;
  sh.width = (f.xmax - f.xmin) * s;
  sh.depth = (f.zmax - f.zmin) * s;
  sh.gap = std::max(under.y - y, 0.f); // sunk into the table: as on contact
  sh.radius = std::sqrt(sh.width * sh.depth / 3.14159265f);
  const float r2 = sh.radius * sh.radius;
  // Coverage: on the headset a stage on a desk read 0.44-0.65 of its rays
  // on it (overhanging the desk); faded from 0.2 to 0.6, not 0.55 to 0.9,
  // which hid most shadows.
  sh.alpha = r2 > 0.f ? surface_settings().alpha * smoothstep(0.2f, 0.6f, coverage) * r2 / (r2 + sh.gap * sh.gap)
                      : 0.f;
  if (sh.alpha < 0.04f) // too faint to see
    sh.alpha = 0.f;
  return sh;
}

// The floor's height in LOCAL (XR_EXT_local_floor), false if unknown.
bool locate_floor(XrTime time, float& y) {
  if (!B.floorSpace || time == 0)
    return false;
  XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
  if (XR_FAILED(xrLocateSpace(B.floorSpace, B.space, time, &loc)) ||
      !(loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT))
    return false;
  y = loc.pose.position.y;
  return true;
}

// The worker's own.
struct SurfaceWorker {
  enum class Caster { Off, Creating, Ready, Declined, Unsupported, Failed };
  Caster caster = Caster::Off;
  XrEnvironmentRaycasterMETA raycaster = XR_NULL_HANDLE;
  XrFutureEXT future = XR_NULL_FUTURE_EXT;
  int64_t createNs = 0, lastCastNs = 0;
  bool active = false, placing = false; // casting; for placing (else a burst)
  SurfaceEstimate est;
  float emaY = 0.f; // followed, the estimate moving only by 5 mm or more
  struct Candidate {
    int64_t ns;
    float y, coverage;
  };
  std::deque<Candidate> recent; // the last 2 s
  int64_t lastCandidateNs = 0;
  // Logged (AURORA_XR_SURFACE_LOG): since the last line, casts by outcome
  // and rays by status (0: the call failed), and each ray's time.
  uint32_t casts = 0, blind = 0, none = 0;
  std::array<uint32_t, 7> statuses{};
  float lastY = 0.f, lastCoverage = 0.f;
  std::vector<float> rayUs;
  int64_t lineNs = 0, timingNs = 0;
};

using Caster = SurfaceWorker::Caster;

void raycaster_failed(SurfaceWorker& w, XrResult r) {
  w.future = XR_NULL_FUTURE_EXT;
  if (r == XR_ERROR_PERMISSION_INSUFFICIENT) {
    w.caster = Caster::Declined;
    Log.info("Environment raycast not allowed (the spatial data permission is off): no surface measured, the "
             "floor stands in");
  } else if (r == XR_ERROR_FEATURE_UNSUPPORTED) {
    w.caster = Caster::Unsupported;
    Log.info("Environment raycast unsupported here: the floor stands in");
  } else {
    w.caster = Caster::Failed;
    Log.warn("Environment raycaster creation failed: XrResult {}; the floor stands in", static_cast<int>(r));
  }
}

// Creates the raycaster without waiting: starts it, then polls its future.
void raycaster_step(SurfaceWorker& w) {
  if (w.caster == Caster::Off) {
    XrEnvironmentRaycasterCreateInfoMETA ci{XR_TYPE_ENVIRONMENT_RAYCASTER_CREATE_INFO_META};
    const XrResult r = B.createRaycasterAsync(B.session, &ci, &w.future);
    if (XR_FAILED(r)) {
      raycaster_failed(w, r);
      return;
    }
    w.caster = Caster::Creating;
    w.createNs = monotonic_ns();
  }
  if (w.caster != Caster::Creating)
    return;
  XrFuturePollInfoEXT pi{XR_TYPE_FUTURE_POLL_INFO_EXT};
  pi.future = w.future;
  XrFuturePollResultEXT pr{XR_TYPE_FUTURE_POLL_RESULT_EXT};
  if (const XrResult r = B.pollFuture(B.instance, &pi, &pr); XR_FAILED(r)) {
    raycaster_failed(w, r);
    return;
  }
  if (pr.state != XR_FUTURE_STATE_READY_EXT)
    return;
  XrEnvironmentRaycasterCreateCompletionMETA done{XR_TYPE_ENVIRONMENT_RAYCASTER_CREATE_COMPLETION_META};
  XrResult r = B.createRaycasterComplete(B.session, w.future, &done);
  w.future = XR_NULL_FUTURE_EXT;
  if (XR_SUCCEEDED(r))
    r = done.futureResult;
  if (XR_FAILED(r)) {
    raycaster_failed(w, r);
    return;
  }
  w.raycaster = done.environmentRaycaster;
  w.caster = Caster::Ready;
  w.lastCastNs = monotonic_ns();
  Log.info("Environment raycaster ready after {} ms", (w.lastCastNs - w.createNs) / kMsNs);
}

void raycaster_destroy(SurfaceWorker& w) {
  if (w.future != XR_NULL_FUTURE_EXT && B.cancelFuture) {
    XrFutureCancelInfoEXT ci{XR_TYPE_FUTURE_CANCEL_INFO_EXT};
    ci.future = w.future;
    B.cancelFuture(B.instance, &ci);
  }
  w.future = XR_NULL_FUTURE_EXT;
  if (w.raycaster)
    B.destroyRaycaster(w.raycaster);
  w.raycaster = XR_NULL_HANDLE;
  if (w.caster == Caster::Creating || w.caster == Caster::Ready)
    w.caster = Caster::Off;
}

// One cast: a grid of rays straight down, from just above the stage's
// underside, across most of its footprint.
struct SurfaceCast {
  bool blind = false; // half the rays or more saw nothing either way: no information
  bool found = false; // a surface under enough of the footprint, and under its middle
  float y = 0.f, coverage = 0.f;
};

constexpr int kSurfaceGrid = 3;            // rays across and deep
constexpr float kSurfaceGridReach = 0.8f;  // of the footprint's half extents
constexpr float kSurfaceRayLift = 0.1f;    // meters over the underside: a stage resting on the desk still sees it
constexpr float kSurfaceRayReach = 2.5f;   // meters: a low stage's floor, not the storey below

SurfaceCast cast_footprint(SurfaceWorker& w, const ArenaSnapshot& a, XrTime time,
                           const std::array<XrVector3f, 2>& hands, const std::array<bool, 2>& handValid) {
  SurfaceCast out;
  const Footprint& f = a.footprint;
  const float cx = (f.xmin + f.xmax) * 0.5f, cz = (f.zmin + f.zmax) * 0.5f;
  const float hx = (f.xmax - f.xmin) * 0.5f * kSurfaceGridReach, hz = (f.zmax - f.zmin) * 0.5f * kSurfaceGridReach;
  XrEnvironmentRaycastFilterDistanceMETA reach{XR_TYPE_ENVIRONMENT_RAYCAST_FILTER_DISTANCE_META};
  reach.maxDistance = kSurfaceRayReach;
  const XrEnvironmentRaycastFilterBaseHeaderMETA* filters[] = {
      reinterpret_cast<const XrEnvironmentRaycastFilterBaseHeaderMETA*>(&reach)};
  struct Height {
    float y;
    int row;
    bool weak; // the hit point is behind something: its pose may be off
  };
  constexpr int kRays = kSurfaceGrid * kSurfaceGrid;
  std::array<Height, kRays> heights{};
  int n = 0, noInfo = 0;
  for (int row = 0; row < kSurfaceGrid; ++row)
    for (int col = 0; col < kSurfaceGrid; ++col) {
      XrVector3f origin = arena_point(a, cx + hx * static_cast<float>(col - 1), f.ybottom,
                                      cz + hz * static_cast<float>(row - 1));
      origin.y += kSurfaceRayLift;
      XrEnvironmentRaycastHitGetInfoMETA gi{XR_TYPE_ENVIRONMENT_RAYCAST_HIT_GET_INFO_META};
      gi.baseSpace = B.space;
      gi.time = time;
      gi.origin = origin;
      gi.direction = {0.f, -1.f, 0.f};
      gi.filterCount = 1;
      gi.filters = filters;
      XrEnvironmentRaycastHitMETA hit{XR_TYPE_ENVIRONMENT_RAYCAST_HIT_META};
      const int64_t t0 = monotonic_ns();
      const XrResult r = B.performRaycast(w.raycaster, &gi, &hit);
      if (w.rayUs.size() < 8192)
        w.rayUs.push_back(static_cast<float>(monotonic_ns() - t0) / 1e3f);
      const int status = XR_SUCCEEDED(r) ? static_cast<int>(hit.status) : 0;
      ++w.statuses[status >= 1 && status <= 6 ? status : 0];
      if (status == 0 || hit.status == XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_POINT_OUTSIDE_OF_FOV_META ||
          hit.status == XR_ENVIRONMENT_RAYCAST_HIT_STATUS_RAY_OCCLUDED_META) {
        ++noInfo; // out of the depth cameras' view, or a hand in the way of it
        continue;
      }
      if (hit.status != XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_META &&
          hit.status != XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_POINT_OCCLUDED_META &&
          hit.status != XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_INVALID_ORIENTATION_META)
        continue; // nothing within reach
      const XrVector3f p = hit.pose.position;
      // Level within 15 degrees (the normal is +Z turned by the pose):
      // cushions, laps and the sides of things are no surface to rest on.
      if (hit.status != XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_INVALID_ORIENTATION_META &&
          qrot(hit.pose.orientation, {0.f, 0.f, 1.f}).y < 0.966f)
        continue;
      // The player's own hand or controller under the stage, between the
      // ray's start and what it hit.
      bool hand = false;
      for (int h = 0; h < 2; ++h)
        hand |= handValid[h] && std::hypot(hands[h].x - origin.x, hands[h].z - origin.z) < 0.12f &&
                hands[h].y > p.y && hands[h].y < origin.y;
      if (!hand)
        heights[n++] = {p.y, row, hit.status == XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_POINT_OCCLUDED_META};
    }
  if (noInfo * 2 >= kRays) {
    out.blind = true;
    return out;
  }
  // Weak hits can keep an estimate alive, never start one.
  if (w.est.state == SurfaceEstimate::State::Unknown)
    n = static_cast<int>(std::remove_if(heights.begin(), heights.begin() + n, [](const Height& h) { return h.weak; }) -
                         heights.begin());
  std::sort(heights.begin(), heights.begin() + n, [](const Height& l, const Height& r) { return l.y < r.y; });
  // The largest cluster within +-1.5 cm. On a tie the higher one: a lower
  // one only when it covers more rays (a box under one corner of the stage
  // shouldn't lift its whole shadow).
  int first = 0, count = 0;
  for (int i = 0, j = 0; i < n; ++i) {
    while (heights[i].y - heights[j].y > 0.03f)
      ++j;
    if (i - j + 1 >= count) {
      count = i - j + 1;
      first = j;
    }
  }
  bool middle = false;
  for (int i = first; i < first + count; ++i)
    middle |= heights[i].row == kSurfaceGrid / 2;
  if (count < 4 || !middle)
    return out;
  out.found = true;
  out.y = heights[first + count / 2].y;
  out.coverage = static_cast<float>(count) / static_cast<float>(kRays);
  return out;
}

// The newest candidates that agree with the newest one (within a
// centimeter), back to the first that doesn't: settled once they span `span`
// with a spread of 5 mm at most. A table holds still to the millimeter; a
// hand held under the stage, or a passer-by, doesn't.
bool surface_settled(const std::deque<SurfaceWorker::Candidate>& recent, int64_t span, float& y, float& sigma,
                     float& coverage) {
  if (recent.empty())
    return false;
  const auto& last = recent.back();
  std::vector<float> ys;
  float cov = 0.f;
  int64_t since = last.ns;
  for (auto it = recent.rbegin(); it != recent.rend() && std::abs(it->y - last.y) <= 0.01f; ++it) {
    ys.push_back(it->y);
    cov += it->coverage;
    since = it->ns;
  }
  if (ys.size() < 3 || last.ns - since < span)
    return false;
  double mean = 0.0, var = 0.0;
  for (float v : ys)
    mean += v;
  mean /= static_cast<double>(ys.size());
  for (float v : ys)
    var += (v - mean) * (v - mean);
  sigma = static_cast<float>(std::sqrt(var / static_cast<double>(ys.size())));
  if (sigma > 0.005f)
    return false;
  std::nth_element(ys.begin(), ys.begin() + static_cast<std::ptrdiff_t>(ys.size() / 2), ys.end());
  y = ys[ys.size() / 2];
  coverage = cov / static_cast<float>(ys.size());
  return true;
}

void surface_note(SurfaceWorker& w, const SurfaceCast& c, int64_t now, const ArenaSnapshot& a) {
  using State = SurfaceEstimate::State;
  ++w.casts;
  if (c.blind) {
    ++w.blind; // no information: nothing changes
    return;
  }
  if (!c.found) {
    ++w.none;
    if (w.est.state == State::Known && now - w.lastCandidateNs > 1000 * kMsNs) {
      w.est.state = State::Unknown;
      w.recent.clear();
      Log.info("Surface under stage {}: lost", a.stage);
    }
    return;
  }
  const int64_t since = w.lastCandidateNs != 0 ? now - w.lastCandidateNs : 0;
  w.lastCandidateNs = now;
  w.lastY = c.y;
  w.lastCoverage = c.coverage;
  w.recent.push_back({now, c.y, c.coverage});
  while (now - w.recent.front().ns > 2000 * kMsNs)
    w.recent.pop_front();
  float y = 0.f, sigma = 0.f, coverage = 0.f;
  const auto settle = [&](const char* how) {
    w.est = {State::Known, SurfaceEstimate::Source::Raycast, y, coverage, sigma};
    w.emaY = y;
    const ShadowShape sh = shadow_shape(a, y, coverage);
    Log.info("Surface under stage {}: {} at y {:.3f} (sigma {:.1f} mm, coverage {:.2f}), {:.2f} m under the stage; "
             "shadow {:.2f}",
             a.stage, how, y, sigma * 1000.f, coverage, sh.gap, sh.alpha);
  };
  if (w.est.state == State::Unknown) {
    // Placing casts all along; a burst gets a little longer to agree.
    if (surface_settled(w.recent, (w.placing ? 400 : 600) * kMsNs, y, sigma, coverage))
      settle("known");
    return;
  }
  if (std::abs(c.y - w.est.y) <= 0.03f) {
    // The same surface: followed smoothly, ignoring millimeters so the
    // shadow doesn't shimmer.
    const float k = 1.f - std::exp(-static_cast<float>(std::min<int64_t>(since, 500 * kMsNs)) / (250.f * kMsNs));
    w.emaY += (c.y - w.emaY) * k;
    if (std::abs(w.emaY - w.est.y) >= 0.005f)
      w.est.y = w.emaY;
    w.est.coverage += (c.coverage - w.est.coverage) * k;
  } else if (surface_settled(w.recent, 500 * kMsNs, y, sigma, coverage)) {
    settle("moved"); // over another surface, consistently
  }
}

void surface_publish(const SurfaceEstimate& e) {
  std::lock_guard lock{g_surface.mutex};
  g_surface.estimate = e;
}

void surface_log(SurfaceWorker& w, int64_t now, const ArenaSnapshot& a) {
  if (now - w.lineNs >= 1000 * kMsNs) {
    if (w.casts > 0) {
      const bool known = w.est.state != SurfaceEstimate::State::Unknown;
      const ShadowShape sh = shadow_shape(a, known ? w.est.y : w.lastY, known ? w.est.coverage : w.lastCoverage);
      Log.info("Surface casts: {} in {:.1f} s ({} blind, {} without a surface); rays hit/no hit/hit occluded/out of "
               "view/ray occluded/no normal/failed {}/{}/{}/{}/{}/{}/{}; last candidate y {:.3f} coverage {:.2f}; "
               "estimate {} y {:.3f} sigma {:.1f} mm coverage {:.2f}; gap {:.2f} m, R {:.2f} m, shadow {:.2f}",
               w.casts, static_cast<double>(now - w.lineNs) / 1e9, w.blind, w.none, w.statuses[1], w.statuses[2],
               w.statuses[3], w.statuses[4], w.statuses[5], w.statuses[6], w.statuses[0], w.lastY, w.lastCoverage,
               surface_state_name(w.est.state), w.est.y, w.est.sigma * 1000.f, w.est.coverage, sh.gap, sh.radius,
               sh.alpha);
    }
    w.lineNs = now;
    w.casts = w.blind = w.none = 0;
    w.statuses = {};
  }
  if (now - w.timingNs >= 10000 * kMsNs) {
    if (!w.rayUs.empty()) {
      std::sort(w.rayUs.begin(), w.rayUs.end());
      const auto at = [&](double q) { return w.rayUs[static_cast<size_t>(q * static_cast<double>(w.rayUs.size() - 1))]; };
      Log.info("Surface rays: {} in {:.0f} s, {:.0f}/{:.0f}/{:.0f} us each (median/99th percentile/most)",
               w.rayUs.size(), static_cast<double>(now - w.timingNs) / 1e9, at(0.5), at(0.99), w.rayUs.back());
    }
    w.timingNs = now;
    w.rayUs.clear();
  }
}

// ---------------------------------------------------------------- the room survey
//
// Where a stage and the screen start depends on how much room there is: at
// a desk facing a monitor the stage is small and near, on a couch in a
// living room it's farther out and larger. Once a launch, as soon as the
// head is tracked, a fan of rays from the head (5 across, +-40 degrees, 6
// from 10 degrees over level to 50 under) against the headset's depth, for
// up to 6 s (the player looking about the menus fills it in). Every hit is
// kept: room_fit puts the stage halfway to the farthest wall ahead, as large
// as fills the view (at most about 74 degrees across), clear of everything
// the rays found, and the screen farther out. Level hits a short way below
// the eyes and within reach are a desk or table, the lowest level surface
// (or XR_EXT_local_floor's floor) says seated or standing: the log's kind,
// and the defaults when the rays saw too little to fit. The result seeds the
// defaults the player hasn't set (room_survey_apply); every placement is
// still theirs. AURORA_XR_ROOM_SURVEY=0: none.
constexpr float kRoomReach = 5.f;
struct RoomSamples {
  struct Level {
    float y, d;
  };
  std::vector<Level> level;
  std::vector<float> walls;
  struct Wall {
    XrVector3f p;
  };
  std::vector<Wall> farWalls; // the forward rays' walls, where (the farthest is judged ahead of LOCAL -z)
  std::vector<XrVector3f> points; // every hit (LOCAL), what the stage must stay clear of
  int rays = 0, informed = 0, open = 0, passes = 0;
  int64_t startNs = 0;
};

void room_cast(SurfaceWorker& w, RoomSamples& rs, const XrPosef& head, XrTime time) {
  const XrVector3f fwd = qrot(head.orientation, {0.f, 0.f, -1.f});
  const float yaw = std::atan2(-fwd.x, -fwd.z);
  XrEnvironmentRaycastFilterDistanceMETA reach{XR_TYPE_ENVIRONMENT_RAYCAST_FILTER_DISTANCE_META};
  reach.maxDistance = kRoomReach;
  const XrEnvironmentRaycastFilterBaseHeaderMETA* filters[] = {
      reinterpret_cast<const XrEnvironmentRaycastFilterBaseHeaderMETA*>(&reach)};
  constexpr float kDeg = 3.14159265f / 180.f;
  for (const float across : {-40.f, -20.f, 0.f, 20.f, 40.f})
    for (const float down : {-10.f, 0.f, 10.f, 20.f, 35.f, 50.f}) {
      const XrVector3f dir = rot_y({0.f, -std::sin(down * kDeg), -std::cos(down * kDeg)}, yaw + across * kDeg);
      XrEnvironmentRaycastHitGetInfoMETA gi{XR_TYPE_ENVIRONMENT_RAYCAST_HIT_GET_INFO_META};
      gi.baseSpace = B.space;
      gi.time = time;
      gi.origin = head.position;
      gi.direction = dir;
      gi.filterCount = 1;
      gi.filters = filters;
      XrEnvironmentRaycastHitMETA hit{XR_TYPE_ENVIRONMENT_RAYCAST_HIT_META};
      const XrResult r = B.performRaycast(w.raycaster, &gi, &hit);
      ++rs.rays;
      if (XR_FAILED(r) || hit.status == XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_POINT_OUTSIDE_OF_FOV_META ||
          hit.status == XR_ENVIRONMENT_RAYCAST_HIT_STATUS_RAY_OCCLUDED_META)
        continue; // out of the depth cameras' view, or a hand in the way
      ++rs.informed;
      const bool forward = down <= 20.f;
      if (hit.status != XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_META &&
          hit.status != XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_POINT_OCCLUDED_META &&
          hit.status != XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_INVALID_ORIENTATION_META) {
        rs.open += forward ? 1 : 0; // nothing within reach
        continue;
      }
      const XrVector3f p = hit.pose.position;
      if (rs.points.size() < 6000)
        rs.points.push_back(p);
      if (hit.status == XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_INVALID_ORIENTATION_META)
        continue; // where, but not which way it faces
      const float d = std::hypot(p.x - head.position.x, p.z - head.position.z);
      const float up = qrot(hit.pose.orientation, {0.f, 0.f, 1.f}).y;
      if (up >= 0.94f && down > 0.f)
        rs.level.push_back({p.y, d});
      else if (up < 0.5f && forward) {
        rs.walls.push_back(d);
        rs.farWalls.push_back({p});
      }
    }
  ++rs.passes;
}

// A box ahead of the eyes, in a frame turned to `yaw` (forward -z at 0):
// `dist` to its center, `hx` and `hz` its half width and depth, `y0`-`y1`
// its heights (LOCAL). Clear when no point the rays found is in it, or
// between the eyes and it from `view0` up (what would cover what's watched
// of it: a desk's edge may hide a stage's underside), with `margin` around.
struct FitBox {
  float dist, hx, hz, y0, y1, view0;
};
bool room_clear(const std::vector<XrVector3f>& points, const XrVector3f& eye, float yaw, const FitBox& b,
                float margin) {
  const XrVector3f fwd{-std::sin(yaw), 0.f, -std::cos(yaw)}, right{std::cos(yaw), 0.f, -std::sin(yaw)};
  const float front = b.dist - b.hz;
  for (const XrVector3f& p : points) {
    const XrVector3f d = p - eye;
    const float z = d.x * fwd.x + d.z * fwd.z, x = d.x * right.x + d.z * right.z;
    if (z < 0.15f || z > b.dist + b.hz + margin)
      continue; // the player's own hands and lap, or past it
    if (z >= front - margin) {
      if (std::abs(x) < b.hx + margin && p.y > b.y0 - margin && p.y < b.y1 + margin)
        return false; // in it
    } else {
      // Between: inside the lines from the eyes to its front's edges.
      const float k = z / front;
      if (std::abs(x) < (b.hx + margin) * k && p.y > eye.y + (b.view0 - margin - eye.y) * k &&
          p.y < eye.y + (b.y1 + margin - eye.y) * k)
        return false;
    }
  }
  return true;
}

// The stage's and the screen's defaults from what the rays found
// (room_survey_apply's), in the user's terms: the stage halfway to the
// farthest wall ahead (at most 2.4 m, at least 40 cm), as large as fills
// the view (its playfield, 300 units across, at most about 74 degrees) and
// no larger than 0.012 m a unit nor smaller than 0.0018, 13 degrees under
// the eyes (where players put it: 0.36 m under at 1.6 m) and above whatever
// it stands over (the box 50 units under its floor to 130 over, 70 deep);
// brought in and shrunk until it and the view of it are clear of every hit
// with 6 cm to spare, preferring the farthest spot that shows it within 10%
// as large as the best. The screen: a little past halfway to that wall
// (0.45-3 m; the user pulled the first fit's, 85% of the way, in to about
// half), as wide to the eye as the player last made it (36 degrees before
// that, g_screenTan), brought in until clear (in front of a monitor, then).
void room_fit(const RoomSamples& rs, RoomSurvey& out) {
  constexpr float kDeg = 3.14159265f / 180.f;
  // The farthest wall ahead of LOCAL -z (within 40 degrees of it), the
  // 90th percentile to leave out a stray far hit; open when the forward
  // rays mostly reached nothing.
  std::vector<float> ahead;
  for (const auto& w : rs.farWalls) {
    const float dx = w.p.x - out.head.x, dz = w.p.z - out.head.z;
    if (-dz > 0.f && std::abs(std::atan2(dx, -dz)) < 40.f * kDeg)
      ahead.push_back(-dz); // how far ahead, not along the ray
  }
  std::sort(ahead.begin(), ahead.end());
  if (!ahead.empty() && rs.open * 4 < static_cast<int>(ahead.size()))
    out.far = ahead[ahead.size() * 9 / 10];
  else if (rs.open > 0)
    out.far = kRoomReach;
  out.points = static_cast<int>(rs.points.size());
  if (out.far < 0.f || rs.points.size() < 40)
    return;
  // Level surfaces under a footprint: the stage stands clear above them.
  const auto top_under = [&](const XrVector3f& c, float hx, float hz) {
    float top = out.floor ? out.floorY : -1e9f;
    for (const auto& l : rs.points) {
      if (std::abs(l.x - c.x) < hx && std::abs(l.z - c.z) < hz && l.y < out.headY - 0.1f)
        top = std::max(top, l.y);
    }
    return top;
  };
  const XrVector3f eye = out.head;
  constexpr float kHalfW = 150.f, kBelow = 50.f, kAbove = 130.f, kHalfD = 35.f, kMargin = 0.06f;
  const float tanView = std::tan(37.f * kDeg), tanDown = std::tan(13.f * kDeg);
  const float ideal = std::clamp(out.far * 0.5f, 0.4f, 2.4f);
  struct Cand {
    float dist, y, scale;
  };
  std::vector<Cand> cands;
  for (float dist = ideal; dist >= 0.4f - 1e-3f; dist -= 0.05f) {
    for (float sc = std::min(dist * tanView / kHalfW, 0.012f); sc >= 0.0018f; sc *= 0.94f) {
      const XrVector3f c{eye.x, 0.f, eye.z - dist};
      float y = eye.y - dist * tanDown;
      y = std::max(y, top_under(c, kHalfW * sc + kMargin, kHalfD * sc + kMargin) + kBelow * sc + kMargin + 0.01f);
      if (y > eye.y - 0.05f)
        continue; // over the eyes: smaller
      const FitBox box{dist, kHalfW * sc, kHalfD * sc, y - kBelow * sc, y + kAbove * sc, y - 15.f * sc};
      if (room_clear(rs.points, eye, 0.f, box, kMargin)) {
        cands.push_back({dist, y, sc});
        break;
      }
    }
  }
  if (!cands.empty()) {
    float best = 0.f;
    for (const Cand& c : cands)
      best = std::max(best, c.scale / c.dist);
    for (const Cand& c : cands) // farthest first
      if (c.scale / c.dist >= best * 0.9f) {
        out.fitted = true;
        out.stageDist = c.dist;
        out.stageY = c.y;
        out.stageScale = c.scale;
        break;
      }
  }
  // The screen, along the head's yaw, at its own height (about the eyes').
  const float tanScreen = g_screenTan.load();
  for (float d = std::clamp(out.far * 0.55f, 0.45f, 3.f); d >= 0.45f - 1e-3f; d -= 0.05f) {
    const float w = 2.f * d * tanScreen, h = w * 0.75f;
    const FitBox box{d, w * 0.5f, 0.03f, eye.y - h * 0.5f, eye.y + h * 0.5f, eye.y - h * 0.5f};
    if (room_clear(rs.points, eye, out.yaw, box, 0.05f)) {
      out.screenFitted = true;
      out.screenDist = d;
      out.screenWidth = w;
      break;
    }
  }
}

RoomSurvey room_finish(const RoomSamples& rs, const XrPosef& head, XrTime time) {
  RoomSurvey out;
  out.done = true;
  out.rays = rs.rays;
  out.informed = rs.informed;
  out.head = head.position;
  out.headY = head.position.y;
  const XrVector3f fwd = qrot(head.orientation, {0.f, 0.f, -1.f});
  out.yaw = std::atan2(-fwd.x, -fwd.z);
  // Level hits by height, in clusters within +-3 cm.
  std::vector<RoomSamples::Level> lv = rs.level;
  std::sort(lv.begin(), lv.end(), [](const auto& a, const auto& b) { return a.y < b.y; });
  struct Cluster {
    float y, near, far;
    int n;
  };
  std::vector<Cluster> clusters;
  for (size_t i = 0; i < lv.size();) {
    size_t j = i;
    float near = lv[i].d, far = lv[i].d;
    while (j + 1 < lv.size() && lv[j + 1].y - lv[i].y < 0.06f) {
      ++j;
      near = std::min(near, lv[j].d);
      far = std::max(far, lv[j].d);
    }
    clusters.push_back({lv[(i + j) / 2].y, near, far, static_cast<int>(j - i + 1)});
    i = j + 1;
  }
  if (float y; locate_floor(time, y)) {
    out.floor = true;
    out.floorY = y;
  } else {
    for (const Cluster& c : clusters) // the lowest well-seen level surface a seat's height or more down
      if (c.n >= 3 && c.y < out.headY - 0.8f) {
        out.floor = true;
        out.floorY = c.y;
        break;
      }
  }
  // The desk: the best-seen level surface 10-80 cm under the eyes, within
  // reach, and clear of the floor.
  const Cluster* desk = nullptr;
  for (const Cluster& c : clusters) {
    const float dy = c.y - out.headY;
    if (c.n >= 4 && dy > -0.8f && dy < -0.1f && c.near < 1.4f && (!out.floor || c.y > out.floorY + 0.3f) &&
        (!desk || c.n > desk->n))
      desk = &c;
  }
  if (desk) {
    out.desk = true;
    out.deskY = desk->y;
    out.deskNear = desk->near;
    out.deskFar = desk->far;
  }
  // Room ahead: the nearer walls the forward rays found, or open when they
  // mostly found nothing within reach.
  if (!rs.walls.empty()) {
    std::vector<float> w = rs.walls;
    std::sort(w.begin(), w.end());
    out.clearance = w[w.size() / 5];
  } else if (rs.open > 0) {
    out.clearance = kRoomReach;
  }
  const bool standing = out.floor && out.headY - out.floorY >= 1.4f;
  out.kind = out.desk && !standing ? RoomSurvey::Kind::Desk
           : standing              ? RoomSurvey::Kind::Room
           : out.floor             ? RoomSurvey::Kind::Couch
                                   : RoomSurvey::Kind::Unknown;
  room_fit(rs, out);
  return out;
}

// One laser against the room's depth: what it hits within 6 m, and whether
// that is a level surface (within about 20 degrees). False: no information
// (out of the depth cameras' view, or the hand in the way): keep the last.
bool cast_pointer(SurfaceWorker& w, XrVector3f origin, XrVector3f dir, XrTime time, SurfaceShared::PointHit& out) {
  XrEnvironmentRaycastFilterDistanceMETA reach{XR_TYPE_ENVIRONMENT_RAYCAST_FILTER_DISTANCE_META};
  reach.maxDistance = 6.f;
  const XrEnvironmentRaycastFilterBaseHeaderMETA* filters[] = {
      reinterpret_cast<const XrEnvironmentRaycastFilterBaseHeaderMETA*>(&reach)};
  XrEnvironmentRaycastHitGetInfoMETA gi{XR_TYPE_ENVIRONMENT_RAYCAST_HIT_GET_INFO_META};
  gi.baseSpace = B.space;
  gi.time = time;
  gi.origin = origin;
  gi.direction = dir;
  gi.filterCount = 1;
  gi.filters = filters;
  XrEnvironmentRaycastHitMETA hit{XR_TYPE_ENVIRONMENT_RAYCAST_HIT_META};
  const XrResult r = B.performRaycast(w.raycaster, &gi, &hit);
  if (XR_FAILED(r) || hit.status == XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_POINT_OUTSIDE_OF_FOV_META ||
      hit.status == XR_ENVIRONMENT_RAYCAST_HIT_STATUS_RAY_OCCLUDED_META)
    return false;
  out = {};
  out.ns = monotonic_ns();
  if (hit.status != XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_META &&
      hit.status != XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_POINT_OCCLUDED_META &&
      hit.status != XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_INVALID_ORIENTATION_META)
    return true; // nothing within reach
  out.hit = true;
  out.p = hit.pose.position;
  // The normal is +Z turned by the pose.
  out.level = hit.status != XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_INVALID_ORIENTATION_META &&
              qrot(hit.pose.orientation, {0.f, 0.f, 1.f}).y > 0.94f;
  return true;
}

void surface_main() {
  using State = SurfaceEstimate::State;
  SurfaceWorker w;
  RoomSamples rs; // the room survey's
  bool freshSurvey = false;
  w.lineNs = w.timingNs = monotonic_ns();
  for (;;) {
    bool placing, reset, surveying, headValid;
    XrPosef head;
    int64_t burstUntil;
    XrTime time;
    std::array<XrVector3f, 2> hands;
    std::array<bool, 2> handValid;
    bool pointing;
    std::array<XrVector3f, 2> rayOrigin, rayDir;
    std::array<bool, 2> rayValid;
    {
      std::unique_lock lock{g_surface.mutex};
      // 15 casts a second while casting, the raycaster's future polled every
      // 50 ms while it's created; otherwise asleep until asked (waking each
      // second to let an idle raycaster go).
      const bool surveyDue = g_surface.surveyWanted && !g_surface.survey.done;
      if (!g_surface.stop && !g_surface.reset)
        g_surface.cv.wait_for(lock, g_surface.pointing                   ? std::chrono::milliseconds(22)
                                    : w.caster == Caster::Creating ? std::chrono::milliseconds(50)
                                    : w.active || surveyDue      ? std::chrono::milliseconds(66)
                                                                 : std::chrono::milliseconds(1000));
      if (g_surface.stop)
        break;
      surveying = g_surface.surveyWanted && !g_surface.survey.done;
      freshSurvey |= std::exchange(g_surface.surveyFresh, false);
      head = g_surface.head;
      headValid = g_surface.headValid;
      placing = g_surface.placing;
      reset = std::exchange(g_surface.reset, false);
      burstUntil = g_surface.burstUntilNs;
      time = g_surface.displayTime;
      hands = g_surface.hands;
      handValid = g_surface.handValid;
      pointing = g_surface.pointing;
      rayOrigin = g_surface.rayOrigin;
      rayDir = g_surface.rayDir;
      rayValid = g_surface.rayValid;
    }
    const int64_t now = monotonic_ns();
    if (pointing && time != 0 && B.envRaycast &&
        (w.caster == Caster::Off || w.caster == Caster::Creating || w.caster == Caster::Ready)) {
      raycaster_step(w);
      if (w.caster == Caster::Ready) {
        std::array<SurfaceShared::PointHit, 2> got{};
        std::array<bool, 2> cast{};
        for (int h = 0; h < 2; ++h)
          if (rayValid[h])
            cast[h] = cast_pointer(w, rayOrigin[h], rayDir[h], time, got[h]);
        w.lastCastNs = now;
        std::lock_guard lock{g_surface.mutex};
        for (int h = 0; h < 2; ++h)
          if (cast[h])
            g_surface.pointHits[h] = got[h];
      }
    }
    {
      std::lock_guard lock{g_surface.mutex};
      g_surface.casterReady = w.caster == Caster::Ready;
      g_surface.casterGone = !B.envRaycast || w.caster == Caster::Declined || w.caster == Caster::Unsupported ||
                             w.caster == Caster::Failed;
    }
    if (reset) {
      w.est = {};
      w.recent.clear();
      surface_publish(w.est);
    }
    if (surveying && headValid && time != 0) {
      if (std::exchange(freshSurvey, false))
        rs = {};
      if (rs.startNs == 0)
        rs.startNs = now;
      bool canCast = false;
      if (B.envRaycast && (w.caster == Caster::Off || w.caster == Caster::Creating || w.caster == Caster::Ready)) {
        raycaster_step(w);
        canCast = w.caster == Caster::Ready;
        if (canCast) {
          room_cast(w, rs, head, time);
          w.lastCastNs = now;
        }
      }
      const bool creating = w.caster == Caster::Creating && now - rs.startNs < 3000 * kMsNs;
      if ((!canCast && !creating) || (rs.passes >= 10 && rs.informed >= 60) || now - rs.startNs > 6000 * kMsNs) {
        const RoomSurvey r = room_finish(rs, head, time);
        std::lock_guard lock{g_surface.mutex};
        g_surface.survey = r;
      }
    }
    const bool active = placing || now < burstUntil;
    if (active && !w.active) {
      // Measured afresh, keeping what was known in the meantime.
      w.recent.clear();
      w.lastCandidateNs = now;
      if (w.est.state == State::Frozen)
        w.est.state = State::Known;
    } else if (!active && w.active && w.est.state == State::Known) {
      // Placed: held for the fight.
      w.est.state = State::Frozen;
      surface_publish(w.est);
      if (surface_settings().log)
        Log.info("Surface frozen at y {:.3f}, coverage {:.2f}", w.est.y, w.est.coverage);
    }
    w.active = active;
    w.placing = placing;
    if (!active) {
      // An idle raycaster goes after 2 s: whatever it keeps running for
      // depth isn't paid for during fights.
      if (w.caster == Caster::Ready && now - w.lastCastNs > 2000 * kMsNs)
        raycaster_destroy(w);
      else if (w.caster == Caster::Creating)
        raycaster_step(w);
      continue;
    }
    const ArenaSnapshot a = arena_snapshot();
    if (!a.footprint.valid || time == 0)
      continue;
    if (B.envRaycast && (w.caster == Caster::Off || w.caster == Caster::Creating || w.caster == Caster::Ready)) {
      raycaster_step(w);
      if (w.caster != Caster::Ready)
        continue;
      surface_note(w, cast_footprint(w, a, time, hands, handValid), now, a);
      w.lastCastNs = now;
      surface_publish(w.est);
      if (surface_settings().log)
        surface_log(w, now, a);
    } else if (float y; locate_floor(time, y)) {
      if (w.est.state == State::Unknown)
        Log.info("Surface under stage {}: the floor, at y {:.3f}", a.stage, y);
      w.est = {State::Known, SurfaceEstimate::Source::Floor, y, 1.f, 0.f};
      surface_publish(w.est);
    }
  }
  raycaster_destroy(w);
}

void start_surface_worker() {
  if (g_surfaceThread.joinable() || (!B.envRaycast && !B.floorSpace))
    return;
  {
    std::lock_guard lock{g_surface.mutex};
    g_surface.stop = false;
  }
  g_surfaceThread = std::thread(surface_main);
}

// Before the session goes: the raycaster is the session's child.
void stop_surface_worker() {
  if (!g_surfaceThread.joinable())
    return;
  {
    std::lock_guard lock{g_surface.mutex};
    g_surface.stop = true;
  }
  g_surface.cv.notify_one();
  g_surfaceThread.join();
}

// XR thread, every frame: what the worker casts for. While the stage is
// being placed it casts all along; once it's let go, half a second more to
// confirm; when a fight's stage appears without a hold (netplay), a second.
void surface_frame(bool fight, bool placing, XrTime time) {
  if (!g_surfaceThread.joinable())
    return;
  static bool wasFight = false, wasPlacing = false;
  int burstMs = 0;
  if (fight && !wasFight && !placing)
    burstMs = 1000;
  if (wasPlacing && !placing)
    burstMs = 500;
  const bool changed = placing != wasPlacing || burstMs > 0;
  wasFight = fight;
  wasPlacing = placing;
  bool viewsValid;
  {
    std::lock_guard lock{g_viewMutex};
    viewsValid = g_viewsValid;
  }
  static const bool survey = env_flag("AURORA_XR_ROOM_SURVEY", true);
  std::lock_guard lock{g_surface.mutex};
  g_surface.displayTime = time;
  g_surface.placing = placing;
  g_surface.head = {B.headOrientation, B.head};
  g_surface.headValid = viewsValid;
  if (survey && viewsValid && B.focused && !g_surface.surveyWanted) {
    g_surface.surveyWanted = true;
    g_surface.cv.notify_one();
  }
  for (int h = 0; h < 2; ++h) {
    g_surface.hands[h] = G.hands[h].touch;
    g_surface.handValid[h] = G.hands[h].valid;
  }
  if (burstMs > 0)
    g_surface.burstUntilNs = std::max(g_surface.burstUntilNs, monotonic_ns() + burstMs * kMsNs);
  if (changed)
    g_surface.cv.notify_one();
}

// XR thread, every frame: once the room survey is done, the defaults it
// suggests for what the player hasn't placed yet, and the screen's glide:
// where room_fit found the stage and the screen fit; when the rays saw too
// little for that, by the kind of place:
//   desk:  the stage on the desk, within reach (today's desk scale, 0.0035);
//          the screen as it is (0.7 m at 0.85 m);
//   couch: seated with no desk ahead: the stage 0.007 at 1.4 m, 35 cm under
//          the eyes; the screen 1.3 m wide at 1.5 m;
//   room:  standing: the stage 0.011 at 2 m, 60 cm under; the screen 1.7 m
//          at 2 m;
// each brought in, and shrunk with it (the same size to the eye), to stay
// clear of a wall ahead. Unknown (no raycast and no floor): as it is. Not
// over an adopted spot, a stage already placed or up, a screen the player
// moved, or AURORA_XR_ARENA_POS/_SCALE and AURORA_XR_SCREEN_DISTANCE/_WIDTH.
// A recenter (room_resurvey): when the survey starts over (XR thread).
struct {
  int64_t startNs = 0;  // monotonic_ns; 0: none due
  bool waiting = false; // started; the screen (and the stage) go where they fit once it's done
  bool stage = false;   // the stage was being placed
} g_resurvey;

// XR thread, at a recenter: a fresh survey from where the player is now,
// and the screen put where it fits ahead of them, as wide to the eye as
// they last made it; while the stage is being placed, the stage too, rather
// than kept where it was in the room. A moment later, so the rays are all
// cast in the new space.
void room_resurvey() {
  static const bool survey = env_flag("AURORA_XR_ROOM_SURVEY", true);
  if (!survey || !g_surfaceThread.joinable())
    return;
  g_resurvey.startNs = monotonic_ns() + 300 * kMsNs;
  g_resurvey.stage = g_placing.load() != 0;
}

void room_survey_apply() {
  static bool applied = false;
  if (g_screenGlide.on && G.target == Target::Screen && (G.oneHand >= 0 || G.twoHands)) {
    g_screenGlide.on = false; // the player took it first
    g_screenMoved = true;
  }
  if (g_screenGlide.on) {
    ScreenPose& s = screen_pose();
    const float t = smoothstep(0.f, 1.f, static_cast<float>(monotonic_ns() - g_screenGlide.startNs) / (800.f * kMsNs));
    const ScreenPose &a = g_screenGlide.from, &b = g_screenGlide.to;
    s.pos = a.pos + (b.pos - a.pos) * t;
    s.width = a.width + (b.width - a.width) * t;
    XrQuaternionf q{a.orientation.x + (b.orientation.x - a.orientation.x) * t,
                    a.orientation.y + (b.orientation.y - a.orientation.y) * t,
                    a.orientation.z + (b.orientation.z - a.orientation.z) * t,
                    a.orientation.w + (b.orientation.w - a.orientation.w) * t};
    const float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (n > 1e-6f)
      s.orientation = {q.x / n, q.y / n, q.z / n, q.w / n};
    if (t >= 1.f)
      g_screenGlide.on = false;
  }
  // A recenter while placing (room_resurvey): the survey over from here,
  // once the new space has settled, and the stage put where it fits ahead.
  if (g_resurvey.startNs != 0 && monotonic_ns() >= g_resurvey.startNs) {
    g_resurvey.startNs = 0;
    g_resurvey.waiting = true;
    std::lock_guard lock{g_surface.mutex};
    g_surface.survey = {};
    g_surface.surveyFresh = true;
    g_surface.surveyWanted = true;
    g_surface.cv.notify_one();
    return;
  }
  bool resurveyed = false;
  if (g_resurvey.waiting) {
    std::lock_guard lock{g_surface.mutex};
    resurveyed = g_surface.survey.done;
  }
  if (applied && !resurveyed)
    return;
  RoomSurvey r;
  {
    std::lock_guard lock{g_surface.mutex};
    if (!g_surface.survey.done)
      return;
    r = g_surface.survey;
  }
  using Kind = RoomSurvey::Kind;
  float scale = 0.0035f, dist = 0.7f, y = r.headY - 0.25f, screenW = 0.7f, screenD = 0.85f;
  switch (r.kind) {
  case Kind::Desk:
    dist = std::clamp(r.deskNear + 0.3f, 0.55f, 0.95f);
    y = std::max(r.headY - 0.25f, r.deskY + 0.15f);
    break;
  case Kind::Couch:
    scale = 0.007f, dist = 1.4f, y = r.headY - 0.35f, screenW = 1.3f, screenD = 1.5f;
    break;
  case Kind::Room:
    scale = 0.011f, dist = 2.f, y = r.headY - 0.6f, screenW = 1.7f, screenD = 2.f;
    break;
  case Kind::Unknown:
    break;
  }
  if (r.fitted) {
    scale = r.stageScale, dist = r.stageDist, y = r.stageY;
  } else if (r.clearance > 0.f) {
    const float maxDist = r.clearance - (0.2f + 60.f * scale); // the stage's depth this side of it
    if (dist > maxDist) {
      const float f = std::max(maxDist, 0.5f) / dist;
      dist *= f;
      scale = std::max(scale * f, 0.0035f);
    }
  }
  if (r.screenFitted) {
    screenD = r.screenDist, screenW = r.screenWidth;
  } else if (r.clearance > 0.f) {
    const float maxScreen = r.clearance - 0.25f;
    if (screenD > maxScreen && screenD > 0.85f) {
      const float f = std::max(maxScreen, 0.85f) / screenD;
      screenD *= f;
      screenW = std::max(screenW * f, 0.7f);
    }
  }
  const auto clearance = r.clearance < 0.f ? std::string{"unknown"}
                         : r.clearance >= kRoomReach ? fmt::format("open past {:.0f} m", kRoomReach)
                                                     : fmt::format("{:.2f} m", r.clearance);
  const auto far = r.far < 0.f ? std::string{"unknown"}
                   : r.far >= kRoomReach ? fmt::format("open past {:.0f} m", kRoomReach)
                                         : fmt::format("{:.2f} m", r.far);
  Log.info("Room survey: {}/{} rays seen, {} hits; desk {}; floor {}; clear ahead {}, the far wall {}: {}{}",
           r.informed, r.rays, r.points,
           r.desk ? fmt::format("{:.2f} m under the eyes, {:.2f}-{:.2f} m out", r.headY - r.deskY, r.deskNear, r.deskFar)
                  : std::string{"none"},
           r.floor ? fmt::format("{:.2f} m under the eyes", r.headY - r.floorY) : std::string{"unknown"}, clearance,
           far, room_kind_name(r.kind), r.fitted ? ", fitted" : "");
  const bool screenEnv = std::getenv("AURORA_XR_SCREEN_DISTANCE") || std::getenv("AURORA_XR_SCREEN_WIDTH");
  if (resurveyed) {
    g_resurvey.waiting = false;
    if (g_resurvey.stage) {
      // The stage being placed, ahead of the eyes (LOCAL -z, the
      // recenter's forward), at its own size; the defaults still those of
      // the launch.
      std::lock_guard lock{g_arenaMutex};
      init_arena();
      ArenaPose a = g_arena;
      a.pos = XrVector3f{r.head.x, y, r.head.z - dist};
      a.yaw = 0.f;
      a.scale = scale * g_stageScale;
      g_arena = a;
      if (g_stage >= 0)
        g_stageArenas[g_stage] = a;
      Log.info("Recentered while placing: the stage {:.2f} m ahead, {:.2f} m under the eyes, {:.4f} m a unit", dist,
               r.headY - y, scale);
    }
    // The screen, ahead, even if the player had moved it: a recenter asks
    // for things in front. At its height; its size to the eye theirs.
    if (!screenEnv && r.screenFitted) {
      g_screenGlide.from = screen_pose();
      g_screenGlide.to.pos =
          XrVector3f{r.head.x, g_screenGlide.from.pos.y, r.head.z} + rot_y({0.f, 0.f, -screenD}, r.yaw);
      g_screenGlide.to.orientation = yaw_quat(r.yaw);
      g_screenGlide.to.width = screenW;
      g_screenGlide.startNs = monotonic_ns();
      g_screenGlide.on = true;
      Log.info("Recentered: the screen to {:.2f} m wide at {:.2f} m", screenW, screenD);
    }
    if (applied)
      return;
  }
  applied = true;
  if (r.kind == Kind::Unknown && !r.fitted)
    return;
  const bool arenaEnv = std::getenv("AURORA_XR_ARENA_POS") || std::getenv("AURORA_XR_ARENA_SCALE");
  {
    std::lock_guard lock{g_arenaMutex};
    init_arena();
    if (!arenaEnv && !g_adoptedSet && g_stageArenas.empty() && g_stage < 0) {
      g_defaultArena.pos = r.fitted ? XrVector3f{r.head.x, y, r.head.z - dist} : XrVector3f{0.f, y, -dist};
      g_defaultArena.scale = scale;
      g_arena = g_defaultArena;
      g_arena.scale *= g_stageScale;
      Log.info("Room survey: stages start {:.3f} m a unit, {:.2f} m out, {:.2f} m under the eyes", scale, dist,
               r.headY - y);
    }
  }
  if (!screenEnv && !g_screenMoved && (screenD != 0.85f || screenW != 0.7f)) {
    g_screenGlide.from = screen_pose();
    g_screenGlide.to.pos = XrVector3f{r.head.x, g_screenGlide.from.pos.y, r.head.z} + rot_y({0.f, 0.f, -screenD}, r.yaw);
    g_screenGlide.to.orientation = yaw_quat(r.yaw);
    g_screenGlide.to.width = screenW;
    g_screenGlide.startNs = monotonic_ns();
    g_screenGlide.on = true;
    Log.info("Room survey: the screen to {:.2f} m wide at {:.2f} m", screenW, screenD);
  }
}

// ---------------------------------------------------------------- XR thread: the stage spot's anchor
//
// The stage spot (aurora_xr_adopt_stage: where every stage starts) is kept
// in the room with a spatial anchor (docs/xr-stage-placement-ux.md C.5):
// made where the spot is as it's adopted, saved to the headset
// (XR_META_spatial_entity_persistence), its id and the spot's size in
// stage-spot.txt beside launcher.cfg, most recently used first, up to eight
// rooms. At each launch the saved ones are looked for
// (XR_META_spatial_entity_discovery); the first the headset finds in the
// room becomes the spot, so the first fight mode needn't ask for it again.
// Each frame the anchor is located, and when it moves in LOCAL (a recenter,
// tracking settling) the stage, the spot and every stage placed this session
// move with it: the stage stays where the player put it in the room.
// Adopting a spot again where the old one is found replaces it; elsewhere
// both are kept. The launcher's Forget renames the file stage-spot.forget,
// and the next launch erases those anchors from the headset. Without anchors
// (USE_ANCHOR_API not granted, AURORA_XR_ANCHORS=0), a recenter's own
// report (XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING) keeps the stage
// in place instead, and the spot lasts for the launch.
struct SpotSaved {
  XrUuidEXT uuid{};
  float scale = 0.f;
  int64_t used = 0; // seconds since the epoch
};
struct StageSpotAnchor {
  bool started = false;
  std::vector<SpotSaved> saved; // most recently used first
  XrAsyncRequestIdFB discoverReq = 0, createReq = 0, statusReq = 0, saveReq = 0;
  struct Candidate {
    XrSpace space;
    XrUuidEXT uuid;
  };
  std::vector<Candidate> candidates; // found, being located
  XrSpace space = XR_NULL_HANDLE;   // the spot's
  XrUuidEXT uuid{};
  bool haveLast = false;
  XrVector3f lastPos{};
  float lastYaw = 0.f;
  std::atomic<bool> createWanted{false};
  XrSpace pending = XR_NULL_HANDLE; // made, being made storable and saved
  XrUuidEXT pendingUuid{};
  float pendingScale = 0.f;
} g_spot;

constexpr size_t kMaxSpots = 8;

std::string spot_file(const char* name) {
  return std::string(g_config.userPath ? g_config.userPath : ".") + "/" + name;
}

std::string uuid_hex(const XrUuidEXT& u) {
  std::string s;
  for (uint8_t b : u.data)
    s += fmt::format("{:02x}", b);
  return s;
}

bool uuid_parse(const std::string& s, XrUuidEXT& u) {
  if (s.size() != 32)
    return false;
  for (int i = 0; i < 16; ++i)
    u.data[i] = static_cast<uint8_t>(std::strtoul(s.substr(i * 2, 2).c_str(), nullptr, 16));
  return true;
}

bool uuid_equal(const XrUuidEXT& a, const XrUuidEXT& b) { return std::memcmp(a.data, b.data, 16) == 0; }

std::vector<SpotSaved> spot_read(const char* name) {
  std::vector<SpotSaved> out;
  FILE* f = std::fopen(spot_file(name).c_str(), "r");
  if (!f)
    return out;
  char hex[64];
  float scale;
  long long used;
  while (std::fscanf(f, "%63s %f %lld", hex, &scale, &used) == 3) {
    SpotSaved s;
    if (uuid_parse(hex, s.uuid) && scale > 0.f) {
      s.scale = scale;
      s.used = used;
      out.push_back(s);
    }
  }
  std::fclose(f);
  return out;
}

void spot_write() {
  std::sort(g_spot.saved.begin(), g_spot.saved.end(), [](const SpotSaved& a, const SpotSaved& b) { return a.used > b.used; });
  FILE* f = std::fopen(spot_file("stage-spot.txt").c_str(), "w");
  if (!f) {
    Log.warn("Stage spot: can't write {}", spot_file("stage-spot.txt"));
    return;
  }
  for (const SpotSaved& s : g_spot.saved)
    std::fprintf(f, "%s %.6f %lld\n", uuid_hex(s.uuid).c_str(), s.scale, static_cast<long long>(s.used));
  std::fclose(f);
}

int64_t now_seconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void spot_erase(std::vector<XrUuidEXT> uuids) {
  if (uuids.empty())
    return;
  XrSpacesEraseInfoMETA ei{XR_TYPE_SPACES_ERASE_INFO_META};
  ei.uuidCount = static_cast<uint32_t>(uuids.size());
  ei.uuids = uuids.data();
  XrAsyncRequestIdFB req = 0;
  const XrResult r = B.eraseSpaces(B.session, &ei, &req);
  Log.info("Stage spot: erasing {} anchor(s) from the headset: XrResult {}", uuids.size(), static_cast<int>(r));
}

// Turns a component on (or finds it on already). True when it's on now.
bool spot_component(XrSpace space, XrSpaceComponentTypeFB type, XrAsyncRequestIdFB* req) {
  XrSpaceComponentStatusFB st{XR_TYPE_SPACE_COMPONENT_STATUS_FB};
  if (XR_SUCCEEDED(B.getComponentStatus(space, type, &st)) && st.enabled && !st.changePending)
    return true;
  XrSpaceComponentStatusSetInfoFB si{XR_TYPE_SPACE_COMPONENT_STATUS_SET_INFO_FB};
  si.componentType = type;
  si.enabled = XR_TRUE;
  XrAsyncRequestIdFB r = 0;
  const XrResult res = B.setComponentStatus(space, &si, &r);
  if (req)
    *req = r;
  return res == XR_ERROR_SPACE_COMPONENT_STATUS_ALREADY_SET_FB;
}

void spot_save(XrSpace space) {
  XrSpacesSaveInfoMETA si{XR_TYPE_SPACES_SAVE_INFO_META};
  si.spaceCount = 1;
  si.spaces = &space;
  if (XR_FAILED(B.saveSpaces(B.session, &si, &g_spot.saveReq)))
    Log.warn("Stage spot: the anchor couldn't be saved");
}

float pose_yaw(const XrQuaternionf& q) {
  const XrVector3f fwd = qrot(q, {0.f, 0.f, -1.f});
  return std::atan2(-fwd.x, -fwd.z);
}

// Every arena pose moved by the same rigid turn about +Y and shift: (p0,
// y0) goes to (p1, y1). Under g_arenaMutex.
void move_arenas(XrVector3f p0, float y0, XrVector3f p1, float y1) {
  const float dy = y1 - y0;
  auto move = [&](ArenaPose& a) {
    a.pos = p1 + rot_y(a.pos - p0, dy);
    a.yaw += dy;
  };
  move(g_adoptedArena);
  move(g_arena);
  for (auto& [stage, a] : g_stageArenas)
    move(a);
}

// A recenter (LOCAL): the new origin, as a pose in the old space. The stage
// keeps its place in the room: everything placed moves the other way.
void spot_recentered(const XrEventDataReferenceSpaceChangePending& e) {
  if (e.referenceSpaceType != XR_REFERENCE_SPACE_TYPE_LOCAL || !e.poseValid)
    return;
  const XrVector3f t = e.poseInPreviousSpace.position;
  const float th = pose_yaw(e.poseInPreviousSpace.orientation);
  // p_new = R^-1 (p_old - t): the old origin goes to rot_y(-t, -th).
  std::lock_guard lock{g_arenaMutex};
  if (!g_arenaInit)
    return;
  move_arenas(t, th, {0.f, 0.f, 0.f}, 0.f);
  if (g_spot.haveLast) {
    g_spot.lastPos = rot_y(g_spot.lastPos - t, -th);
    g_spot.lastYaw -= th;
  }
  Log.info("Recentered: the stage kept its place in the room");
}

// XR thread: a spatial anchor event. True if it was one.
bool spot_event(const XrEventDataBuffer& ev) {
  switch (ev.type) {
  case XR_TYPE_EVENT_DATA_SPACE_DISCOVERY_RESULTS_AVAILABLE_META: {
    const auto& e = reinterpret_cast<const XrEventDataSpaceDiscoveryResultsAvailableMETA&>(ev);
    if (e.requestId != g_spot.discoverReq)
      return true;
    XrSpaceDiscoveryResultsMETA res{XR_TYPE_SPACE_DISCOVERY_RESULTS_META};
    if (XR_FAILED(B.retrieveDiscovery(B.session, e.requestId, &res)))
      return true;
    std::vector<XrSpaceDiscoveryResultMETA> found(res.resultCountOutput);
    res.resultCapacityInput = static_cast<uint32_t>(found.size());
    res.results = found.data();
    if (XR_FAILED(B.retrieveDiscovery(B.session, e.requestId, &res)))
      return true;
    for (uint32_t i = 0; i < res.resultCountOutput; ++i) {
      spot_component(found[i].space, XR_SPACE_COMPONENT_TYPE_LOCATABLE_FB, nullptr);
      g_spot.candidates.push_back({found[i].space, found[i].uuid});
    }
    Log.info("Stage spot: {} saved anchor(s) found, locating", res.resultCountOutput);
    return true;
  }
  case XR_TYPE_EVENT_DATA_SPACE_DISCOVERY_COMPLETE_META: {
    const auto& e = reinterpret_cast<const XrEventDataSpaceDiscoveryCompleteMETA&>(ev);
    if (e.requestId == g_spot.discoverReq && XR_FAILED(e.result))
      Log.info("Stage spot: discovery ended, XrResult {}", static_cast<int>(e.result));
    return true;
  }
  case XR_TYPE_EVENT_DATA_SPATIAL_ANCHOR_CREATE_COMPLETE_FB: {
    const auto& e = reinterpret_cast<const XrEventDataSpatialAnchorCreateCompleteFB&>(ev);
    if (e.requestId != g_spot.createReq)
      return true;
    if (XR_FAILED(e.result)) {
      Log.warn("Stage spot: the anchor wasn't made: XrResult {}", static_cast<int>(e.result));
      return true;
    }
    g_spot.pending = e.space;
    g_spot.pendingUuid = e.uuid;
    if (spot_component(e.space, XR_SPACE_COMPONENT_TYPE_STORABLE_FB, &g_spot.statusReq))
      spot_save(e.space);
    return true;
  }
  case XR_TYPE_EVENT_DATA_SPACE_SET_STATUS_COMPLETE_FB: {
    const auto& e = reinterpret_cast<const XrEventDataSpaceSetStatusCompleteFB&>(ev);
    if (e.requestId == g_spot.statusReq && e.space == g_spot.pending) {
      if (XR_SUCCEEDED(e.result) || e.result == XR_ERROR_SPACE_COMPONENT_STATUS_ALREADY_SET_FB)
        spot_save(e.space);
      else
        Log.warn("Stage spot: the anchor can't be stored: XrResult {}", static_cast<int>(e.result));
    }
    return true;
  }
  case XR_TYPE_EVENT_DATA_SPACES_SAVE_RESULT_META: {
    const auto& e = reinterpret_cast<const XrEventDataSpacesSaveResultMETA&>(ev);
    if (e.requestId != g_spot.saveReq || g_spot.pending == XR_NULL_HANDLE)
      return true;
    if (XR_FAILED(e.result)) {
      Log.warn("Stage spot: the anchor wasn't saved: XrResult {}", static_cast<int>(e.result));
      xrDestroySpace(g_spot.pending);
      g_spot.pending = XR_NULL_HANDLE;
      return true;
    }
    // The old spot's anchor found here: the same room, so this replaces it.
    std::vector<XrUuidEXT> erase;
    if (g_spot.space != XR_NULL_HANDLE && g_spot.haveLast) {
      erase.push_back(g_spot.uuid);
      std::erase_if(g_spot.saved, [&](const SpotSaved& s) { return uuid_equal(s.uuid, g_spot.uuid); });
    }
    if (g_spot.space != XR_NULL_HANDLE)
      xrDestroySpace(g_spot.space);
    g_spot.space = g_spot.pending;
    g_spot.uuid = g_spot.pendingUuid;
    g_spot.pending = XR_NULL_HANDLE;
    g_spot.haveLast = false;
    g_spot.saved.insert(g_spot.saved.begin(), {g_spot.uuid, g_spot.pendingScale, now_seconds()});
    while (g_spot.saved.size() > kMaxSpots) {
      erase.push_back(g_spot.saved.back().uuid);
      g_spot.saved.pop_back();
    }
    spot_write();
    spot_erase(erase);
    Log.info("Stage spot saved in the room (anchor {})", uuid_hex(g_spot.uuid));
    return true;
  }
  case XR_TYPE_EVENT_DATA_SPACES_ERASE_RESULT_META:
    return true;
  default:
    return false;
  }
}

// XR thread, every frame while the session runs.
void spot_frame(XrTime time) {
  if (!B.anchors || !B.running || time == 0)
    return;
  if (!g_spot.started) {
    g_spot.started = true;
    // Forgotten in the launcher: off the headset too.
    std::vector<XrUuidEXT> forget;
    for (const SpotSaved& s : spot_read("stage-spot.forget"))
      forget.push_back(s.uuid);
    if (!forget.empty()) {
      spot_erase(forget);
      std::remove(spot_file("stage-spot.forget").c_str());
    }
    g_spot.saved = spot_read("stage-spot.txt");
    if (!g_spot.saved.empty()) {
      std::vector<XrUuidEXT> uuids;
      for (const SpotSaved& s : g_spot.saved)
        uuids.push_back(s.uuid);
      XrSpaceFilterUuidMETA fu{XR_TYPE_SPACE_FILTER_UUID_META};
      fu.uuidCount = static_cast<uint32_t>(uuids.size());
      fu.uuids = uuids.data();
      const XrSpaceFilterBaseHeaderMETA* filters[] = {reinterpret_cast<const XrSpaceFilterBaseHeaderMETA*>(&fu)};
      XrSpaceDiscoveryInfoMETA di{XR_TYPE_SPACE_DISCOVERY_INFO_META};
      di.filterCount = 1;
      di.filters = filters;
      const XrResult r = B.discoverSpaces(B.session, &di, &g_spot.discoverReq);
      Log.info("Stage spot: looking for {} saved spot(s) in the room: XrResult {}", uuids.size(), static_cast<int>(r));
    }
  }
  // A spot just adopted: an anchor where it is.
  if (g_spot.createWanted.exchange(false)) {
    ArenaPose a;
    {
      std::lock_guard lock{g_arenaMutex};
      a = g_adoptedArena;
    }
    XrSpatialAnchorCreateInfoFB ci{XR_TYPE_SPATIAL_ANCHOR_CREATE_INFO_FB};
    ci.space = B.space;
    ci.poseInSpace = {yaw_quat(a.yaw), a.pos};
    ci.time = time;
    g_spot.pendingScale = a.scale;
    if (XR_FAILED(B.createSpatialAnchor(B.session, &ci, &g_spot.createReq)))
      Log.warn("Stage spot: no anchor could be made");
  }
  auto locate = [&](XrSpace space, XrVector3f& pos, float& yaw) {
    XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
    if (XR_FAILED(xrLocateSpace(space, B.space, time, &loc)))
      return false;
    constexpr XrSpaceLocationFlags kValid = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
    if ((loc.locationFlags & kValid) != kValid)
      return false;
    pos = loc.pose.position;
    yaw = pose_yaw(loc.pose.orientation);
    return true;
  };
  // Saved spots, until one is found here (or the player places one first).
  if (!g_spot.candidates.empty()) {
    bool adopted;
    {
      std::lock_guard lock{g_arenaMutex};
      adopted = g_adoptedSet;
    }
    for (size_t i = 0; i < g_spot.candidates.size() && !adopted; ++i) {
      XrVector3f pos;
      float yaw;
      if (!locate(g_spot.candidates[i].space, pos, yaw))
        continue;
      auto it = std::find_if(g_spot.saved.begin(), g_spot.saved.end(),
                             [&](const SpotSaved& s) { return uuid_equal(s.uuid, g_spot.candidates[i].uuid); });
      if (it == g_spot.saved.end())
        continue;
      {
        std::lock_guard lock{g_arenaMutex};
        init_arena();
        g_adoptedArena = {pos, yaw, it->scale};
        g_adoptedSet = true;
        g_stageArenas.clear();
      }
      it->used = now_seconds();
      g_spot.space = g_spot.candidates[i].space;
      g_spot.uuid = g_spot.candidates[i].uuid;
      g_spot.candidates.erase(g_spot.candidates.begin() + static_cast<long>(i));
      g_spot.haveLast = true;
      g_spot.lastPos = pos;
      g_spot.lastYaw = yaw;
      spot_write();
      adopted = true;
      Log.info("Stage spot found in the room (anchor {}): {:.2f},{:.2f},{:.2f}, yaw {:.0f} deg, scale {:.5f}",
               uuid_hex(g_spot.uuid), pos.x, pos.y, pos.z, yaw * 57.2958f, it->scale);
    }
    if (adopted) {
      for (auto& c : g_spot.candidates)
        xrDestroySpace(c.space);
      g_spot.candidates.clear();
    }
  }
  // The spot's anchor: where it is now; if it moved in LOCAL, everything
  // placed moves with it.
  if (g_spot.space != XR_NULL_HANDLE) {
    XrVector3f pos;
    float yaw;
    if (locate(g_spot.space, pos, yaw)) {
      if (g_spot.haveLast) {
        const XrVector3f d = pos - g_spot.lastPos;
        if (vlen(d) > 0.0005f || std::abs(yaw - g_spot.lastYaw) > 0.001f) {
          std::lock_guard lock{g_arenaMutex};
          move_arenas(g_spot.lastPos, g_spot.lastYaw, pos, yaw);
          if (vlen(d) > 0.02f)
            Log.info("Stage spot: the room moved {:.2f} m in tracking space; the stage kept its place", vlen(d));
        }
      }
      g_spot.haveLast = true;
      g_spot.lastPos = pos;
      g_spot.lastYaw = yaw;
    }
  }
}

void spot_destroy() {
  for (auto& c : g_spot.candidates)
    xrDestroySpace(c.space);
  g_spot.candidates.clear();
  if (g_spot.space != XR_NULL_HANDLE)
    xrDestroySpace(g_spot.space);
  if (g_spot.pending != XR_NULL_HANDLE)
    xrDestroySpace(g_spot.pending);
  g_spot.space = g_spot.pending = XR_NULL_HANDLE;
  g_spot.haveLast = false;
  g_spot.started = false;
}

constexpr uint32_t kShadowSize = 128;
// How far the shadow's texture reaches, against the footprint it's for (each
// way): room for the penumbra, which fades out before the texture's edge.
constexpr float kShadowMargin = 1.6f;

// Premultiplied black: a rounded rectangle (a superellipse), solid in the
// middle and soft from halfway out, half dark at the footprint's edge.
bool create_shadow_texture() {
  std::vector<uint8_t> px(kShadowSize * kShadowSize * 4, 0);
  for (uint32_t y = 0; y < kShadowSize; ++y)
    for (uint32_t x = 0; x < kShadowSize; ++x) {
      const float u = ((x + 0.5f) / kShadowSize * 2.f - 1.f) * kShadowMargin;
      const float v = ((y + 0.5f) / kShadowSize * 2.f - 1.f) * kShadowMargin;
      const float r = std::pow(std::pow(std::abs(u), 4.f) + std::pow(std::abs(v), 4.f), 0.25f);
      px[(y * kShadowSize + x) * 4 + 3] = static_cast<uint8_t>(std::lround((1.f - smoothstep(0.5f, 1.5f, r)) * 255.f));
    }
  return create_static_swapchain(B.shadowSwapchain, kShadowSize, kShadowSize, px);
}

// The rotation b, then a.
XrQuaternionf qmul(const XrQuaternionf& a, const XrQuaternionf& b) {
  return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
          a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

// XR thread, every frame: the stage's shadow on the surface under it, false
// when there's none to show (`shown`: a fight shows over passthrough). It
// fades in and out over about a quarter second, never popping.
bool build_shadow_layer(XrCompositionLayerQuad& q, XrCompositionLayerColorScaleBiasKHR& tint, bool shown) {
  static float alpha = 0.f; // as shown, easing toward the target
  static ShadowShape last;
  static auto lastTime = std::chrono::steady_clock::now();
  const auto now = std::chrono::steady_clock::now();
  const float dt = std::min(std::chrono::duration<float>(now - lastTime).count(), 0.1f);
  lastTime = now;
  if (!shown || !surface_settings().shadow || !B.shadowSwapchain || !B.hasColorScaleBias) {
    alpha = 0.f;
    return false;
  }
  SurfaceEstimate e;
  {
    std::lock_guard lock{g_surface.mutex};
    e = g_surface.estimate;
  }
  float target = 0.f;
  if (const ArenaSnapshot a = arena_snapshot(); e.state != SurfaceEstimate::State::Unknown && a.footprint.valid) {
    last = shadow_shape(a, e.y, e.coverage);
    target = last.alpha;
  }
  alpha += (target - alpha) * (1.f - std::exp(-dt / 0.08f));
  if (target == 0.f && alpha < 0.003f)
    alpha = 0.f;
  if (alpha <= 0.f)
    return false;
  const float grow = 1.f + surface_settings().spread * (last.radius > 0.f ? last.gap / last.radius : 0.f);
  q = {XR_TYPE_COMPOSITION_LAYER_QUAD};
  q.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
  q.space = B.space;
  q.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
  q.subImage.swapchain = B.shadowSwapchain;
  q.subImage.imageRect = {{0, 0}, {static_cast<int32_t>(kShadowSize), static_cast<int32_t>(kShadowSize)}};
  // Flat on the surface (a quad faces +Z: tipped back to face up), turned
  // with the stage, 2 mm over it.
  constexpr XrQuaternionf kFaceUp{-0.70710678f, 0.f, 0.f, 0.70710678f};
  q.pose = {qmul(yaw_quat(last.yaw), kFaceUp), {last.center.x, last.center.y + 0.002f, last.center.z}};
  q.size = {last.width * kShadowMargin * grow, last.depth * kShadowMargin * grow};
  tint = {XR_TYPE_COMPOSITION_LAYER_COLOR_SCALE_BIAS_KHR};
  tint.colorScale = {alpha, alpha, alpha, alpha}; // premultiplied black: only the alpha matters
  q.next = &tint;
  return true;
}

// A pose the runtime accepts: finite, its orientation a unit quaternion.
// Fixed in place when it's only off length; false when it can't be.
bool fix_pose(XrPosef& p, bool& fixed) {
  const XrVector3f& v = p.position;
  XrQuaternionf& q = p.orientation;
  if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
    return false;
  const float n2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  if (!std::isfinite(n2) || n2 < 0.25f)
    return false;
  if (std::abs(n2 - 1.f) > 1e-4f) {
    const float k = 1.f / std::sqrt(n2);
    q = {q.x * k, q.y * k, q.z * k, q.w * k};
    fixed = true;
  }
  return true;
}

// Before xrEndFrame: one bad pose fails the whole frame (XR_ERROR_POSE_INVALID),
// which used to end the XR thread and leave the headset black. Each layer's
// poses are checked; one only off length is normalized, a layer with one
// that can't be fixed is left out, and the first few are logged with their
// values to find where they came from. Returns the layers kept.
uint32_t sanitize_layers(const XrCompositionLayerBaseHeader** layers, uint32_t count) {
  static int logged = 0;
  const auto report = [](uint32_t i, const char* what, const XrPosef& p, bool kept) {
    if (logged++ >= 8)
      return;
    Log.warn("Composition layer {} ({}) had an invalid pose: position {},{},{}, orientation {},{},{},{}; {}", i, what,
             p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z,
             p.orientation.w, kept ? "normalized" : "left out");
  };
  uint32_t kept = 0;
  for (uint32_t i = 0; i < count; ++i) {
    auto* base = const_cast<XrCompositionLayerBaseHeader*>(layers[i]);
    bool ok = true;
    if (base->type == XR_TYPE_COMPOSITION_LAYER_QUAD) {
      auto& q = *reinterpret_cast<XrCompositionLayerQuad*>(base);
      const XrPosef before = q.pose;
      bool fixed = false;
      ok = fix_pose(q.pose, fixed);
      if (!ok || fixed)
        report(i, "quad", before, ok);
      // A quad of no size (a laser too short to see) shows nothing anyway:
      // left out quietly.
      ok = ok && std::isfinite(q.size.width) && std::isfinite(q.size.height) && q.size.width > 0.f &&
           q.size.height > 0.f;
    } else if (base->type == XR_TYPE_COMPOSITION_LAYER_PROJECTION) {
      auto& proj = *reinterpret_cast<XrCompositionLayerProjection*>(base);
      for (uint32_t v = 0; v < proj.viewCount && ok; ++v) {
        auto& view = const_cast<XrCompositionLayerProjectionView&>(proj.views[v]);
        const XrPosef before = view.pose;
        bool fixed = false;
        ok = fix_pose(view.pose, fixed);
        if (!ok || fixed)
          report(i, "3D view", before, ok);
        for (auto* n = reinterpret_cast<XrBaseOutStructure*>(const_cast<void*>(view.next)); n && ok; n = n->next)
          if (n->type == XR_TYPE_COMPOSITION_LAYER_SPACE_WARP_INFO_FB) {
            auto& sw = *reinterpret_cast<XrCompositionLayerSpaceWarpInfoFB*>(n);
            const XrPosef b2 = sw.appSpaceDeltaPose;
            bool f2 = false;
            ok = fix_pose(sw.appSpaceDeltaPose, f2);
            if (!ok || f2)
              report(i, "SpaceWarp delta", b2, ok);
          }
      }
    }
    if (ok)
      layers[kept++] = layers[i];
  }
  return kept;
}

// ---------------------------------------------------------------- XR worker: the room's grade
//
// During a fight in mixed reality the room takes on a little of the stage's
// palette (docs/xr-stage-grounding-report.md, T5; Resolution Games' colour
// tint on the video feed): Brinstar's warm red, Icicle Mountain's and Big
// Blue's cool, Fountain of Dreams' dusk violet, the house lights down for
// Battlefield and Final Destination. The game names each stage's grade
// (aurora_xr_set_room_grade, src/pc/xr_scene.c). It's a passthrough style,
// so the compositor applies it and the eye pass pays nothing. With
// XR_META_passthrough_color_lut it's a colour LUT built for the stage and an
// interpolated map from an identity LUT to it, whose weight fades in as the
// fight starts (or its hold is let go) and out as it ends; without the
// extension, brightness and saturation, which can't tint. A grade may also
// change mid-fight (Venom's caves, as its lights dim): the LUT is then
// rewritten in place, at most ten times a second. A worker thread
// builds the LUTs and sets the styles; the XR thread only says whether a
// fight shows over the room.
//
// Off by default (grade_wanted): AURORA_XR_GRADE=1 turns it on;
// AURORA_XR_GRADE_STRENGTH (1) scales every
// grade's departure from none; AURORA_XR_GRADE_FADE (1 s) is the fade's
// length; AURORA_XR_GRADE_LUT (16, up to 32 and the runtime's limit; 0:
// brightness and saturation instead) the LUT's size; AURORA_XR_GRADE_LOG=1
// logs the builds and the fades.

// Gains per channel (linear light) and a saturation; all 1 is none.
struct RoomGrade {
  float r = 1.f, g = 1.f, b = 1.f, saturation = 1.f;
  bool none() const { return r == 1.f && g == 1.f && b == 1.f && saturation == 1.f; }
  // Its departure from none scaled by k.
  RoomGrade scaled(float k) const {
    return {1.f + (r - 1.f) * k, 1.f + (g - 1.f) * k, 1.f + (b - 1.f) * k, 1.f + (saturation - 1.f) * k};
  }
};

struct GradeSettings {
  bool on = true, log = false;
  float strength = 1.f, fade = 1.f;
  uint32_t lut = 16; // 0: brightness and saturation
};

const GradeSettings& grade_settings() {
  static const GradeSettings s = [] {
    GradeSettings s;
    s.on = grade_wanted();
    s.log = env_flag("AURORA_XR_GRADE_LOG", false);
    s.strength = std::clamp(env_float("AURORA_XR_GRADE_STRENGTH", 1.f), 0.f, 3.f);
    s.fade = std::max(env_float("AURORA_XR_GRADE_FADE", 1.f), 0.05f);
    // A power of two (the extension requires it), within its limit.
    const uint32_t want = static_cast<uint32_t>(std::clamp(env_float("AURORA_XR_GRADE_LUT", 16.f), 0.f, 32.f));
    s.lut = want >= 2 && B.maxLutResolution >= 2 ? std::bit_floor(std::min(want, B.maxLutResolution)) : 0;
    return s;
  }();
  return s;
}

// Between the game, the XR thread and the worker.
struct GradeShared {
  std::mutex mutex;
  std::condition_variable cv;
  RoomGrade grade;   // the current stage's: the game's, or the panel's in its place
  uint64_t seq = 0;  // bumped with every new grade
  bool on = false;   // a fight shows over the room
  bool stop = false;
  RoomGrade game;    // the game's for the current stage (aurora_xr_set_room_grade)
  int stage = -1;    // aurora_xr_set_stage's
  std::unordered_map<int, RoomGrade> panel; // set on the grade panel, by stage
} g_grade;

// The grade shown from now on. g_grade.mutex held.
void set_grade_locked(const RoomGrade& grade) {
  if (grade.r == g_grade.grade.r && grade.g == g_grade.grade.g && grade.b == g_grade.grade.b &&
      grade.saturation == g_grade.grade.saturation)
    return;
  g_grade.grade = grade;
  ++g_grade.seq;
  g_grade.cv.notify_one();
}
std::thread g_gradeThread;
// The passthrough's start, pause and styles, from the XR thread and the worker.
std::mutex g_passthroughMutex;

float srgb_decode(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }
float srgb_encode(float l) {
  l = std::clamp(l, 0.f, 1.f);
  return l <= 0.0031308f ? l * 12.92f : 1.055f * std::pow(l, 1.f / 2.4f) - 0.055f;
}

// A res³ RGB LUT (sRGB in and out, red fastest): saturation about each
// colour's luminance, then the gains, in linear light.
std::vector<uint8_t> grade_lut(const RoomGrade& g, uint32_t res) {
  std::vector<uint8_t> data(static_cast<size_t>(res) * res * res * 3);
  std::vector<float> lin(res);
  for (uint32_t i = 0; i < res; ++i)
    lin[i] = srgb_decode(static_cast<float>(i) / static_cast<float>(res - 1));
  size_t o = 0;
  for (uint32_t bi = 0; bi < res; ++bi)
    for (uint32_t gi = 0; gi < res; ++gi)
      for (uint32_t ri = 0; ri < res; ++ri) {
        const float r = lin[ri], gg = lin[gi], b = lin[bi];
        const float y = 0.2126f * r + 0.7152f * gg + 0.0722f * b;
        const float out[3] = {(y + (r - y) * g.saturation) * g.r, (y + (gg - y) * g.saturation) * g.g,
                              (y + (b - y) * g.saturation) * g.b};
        for (float c : out)
          data[o++] = static_cast<uint8_t>(std::lround(srgb_encode(c) * 255.f));
      }
  return data;
}

XrPassthroughColorLutMETA create_lut(const RoomGrade& g, uint32_t res) {
  const std::vector<uint8_t> data = grade_lut(g, res);
  XrPassthroughColorLutCreateInfoMETA ci{XR_TYPE_PASSTHROUGH_COLOR_LUT_CREATE_INFO_META};
  ci.channels = XR_PASSTHROUGH_COLOR_LUT_CHANNELS_RGB_META;
  ci.resolution = res;
  ci.data = {static_cast<uint32_t>(data.size()), data.data()};
  XrPassthroughColorLutMETA lut = XR_NULL_HANDLE;
  if (const XrResult r = B.createColorLut(B.passthrough, &ci, &lut); XR_FAILED(r)) {
    Log.warn("Passthrough colour LUT ({}^3) not created: XrResult {}", res, static_cast<int>(r));
    return XR_NULL_HANDLE;
  }
  return lut;
}

// A grade that changes during a fight (a stage's own lighting, as Venom's
// caves) rewrites the LUT in place instead of making another.
bool update_lut(XrPassthroughColorLutMETA lut, const RoomGrade& g, uint32_t res) {
  if (!B.updateColorLut)
    return false;
  const std::vector<uint8_t> data = grade_lut(g, res);
  XrPassthroughColorLutUpdateInfoMETA ui{XR_TYPE_PASSTHROUGH_COLOR_LUT_UPDATE_INFO_META};
  ui.data = {static_cast<uint32_t>(data.size()), data.data()};
  return XR_SUCCEEDED(B.updateColorLut(lut, &ui));
}

void grade_main() {
  const GradeSettings& set = grade_settings();
  uint32_t lutRes = set.lut;
  XrPassthroughColorLutMETA identity = XR_NULL_HANDLE, target = XR_NULL_HANDLE;
  // The look the target LUT (or the fallback) shows, strength applied: the
  // last grade that wasn't none, kept so a grade going to none fades out.
  RoomGrade built;
  bool lookOn = false; // the current grade isn't none
  uint64_t builtSeq = UINT64_MAX;
  int64_t builtNs = 0, loggedNs = 0; // a new grade at most every 100 ms
  float weight = 0.f;            // faded toward 1 while a fight shows, 0 otherwise
  float appliedWeight = 0.f;     // in the style last set (0: a plain one)
  bool force = false;
  float loggedWant = 0.f;
  auto last = std::chrono::steady_clock::now();
  // A style with the grade at weight w, or plain at 0.
  const auto apply = [&](float w) {
    XrPassthroughStyleFB style{XR_TYPE_PASSTHROUGH_STYLE_FB};
    style.textureOpacityFactor = 1.f;
    style.edgeColor = {0.f, 0.f, 0.f, 0.f};
    XrPassthroughColorMapInterpolatedLutMETA interp{XR_TYPE_PASSTHROUGH_COLOR_MAP_INTERPOLATED_LUT_META};
    XrPassthroughBrightnessContrastSaturationFB bcs{XR_TYPE_PASSTHROUGH_BRIGHTNESS_CONTRAST_SATURATION_FB};
    if (identity && target) {
      // Weight 0 too: the style stays a LUT for the session, only its
      // weight moves (below).
      interp.sourceColorLut = identity;
      interp.targetColorLut = target;
      interp.weight = w;
      style.next = &interp;
    } else if (w > 0.f) {
      // Brightness is about L* (-100 to 100): the gains' luminance as a
      // change of lightness. Saturation as is. No tint.
      const RoomGrade g = built.scaled(w);
      const float lum = 0.2126f * g.r + 0.7152f * g.g + 0.0722f * g.b;
      bcs.brightness = std::clamp(100.f * (std::cbrt(lum) - 1.f), -100.f, 100.f);
      bcs.contrast = 1.f;
      bcs.saturation = std::max(g.saturation, 0.f);
      style.next = &bcs;
    }
    XrResult r;
    {
      std::lock_guard lock{g_passthroughMutex};
      r = B.setPassthroughStyle(B.passthroughLayer, &style);
    }
    static int warnings = 0;
    if (XR_FAILED(r) && warnings++ < 3)
      Log.warn("Passthrough style not set: XrResult {}", static_cast<int>(r));
    appliedWeight = w;
  };
  // The LUT style goes on once, now, at weight 0 (an identity LUT both
  // ways), and stays until the end: the compositor blacked passthrough out
  // for a moment each time a style with a LUT replaced a plain one or the
  // other way round, so a fight's grade fading in and out only moves the
  // weight.
  if (lutRes > 0) {
    identity = create_lut({}, lutRes);
    target = identity ? create_lut({}, lutRes) : XR_NULL_HANDLE;
    if (target)
      apply(0.f);
  }
  for (;;) {
    RoomGrade grade;
    uint64_t seq;
    bool on;
    {
      std::unique_lock lock{g_grade.mutex};
      const bool fading = weight != (g_grade.on && lookOn ? 1.f : 0.f);
      if (!g_grade.stop && (g_grade.seq == builtSeq || monotonic_ns() - builtNs < 100 * kMsNs))
        g_grade.cv.wait_for(lock, fading || g_grade.seq != builtSeq ? std::chrono::milliseconds(33)
                                                                    : std::chrono::milliseconds(1000));
      if (g_grade.stop)
        break;
      grade = g_grade.grade;
      seq = g_grade.seq;
      on = g_grade.on;
    }
    const auto now = std::chrono::steady_clock::now();
    const float dt = std::min(std::chrono::duration<float>(now - last).count(), 0.25f);
    last = now;
    if (const int64_t t0 = monotonic_ns(); seq != builtSeq && t0 - builtNs >= 100 * kMsNs) {
      builtSeq = seq;
      builtNs = t0;
      const RoomGrade g = grade.scaled(set.strength);
      lookOn = !g.none();
      if (lookOn) {
        built = g;
        const char* how = "updated";
        if (lutRes > 0) {
          if (!identity)
            identity = create_lut({}, lutRes);
          if (!target || !update_lut(target, built, lutRes)) {
            if (target)
              B.destroyColorLut(target); // kept by the runtime while a style still uses it
            target = identity ? create_lut(built, lutRes) : XR_NULL_HANDLE;
            how = "built";
          }
          if (!target) {
            lutRes = 0; // brightness and saturation from now on
            Log.warn("Room grade: no colour LUTs, brightness and saturation instead");
          }
        }
        if (set.log && t0 - loggedNs >= 1000 * kMsNs) {
          loggedNs = t0;
          Log.info("Room grade {:.2f},{:.2f},{:.2f} saturation {:.2f}: {} in {:.1f} ms", built.r, built.g, built.b,
                   built.saturation, lutRes > 0 ? fmt::format("{}^3 LUT {}", lutRes, how) : std::string{"no LUT"},
                   static_cast<double>(monotonic_ns() - t0) / 1e6);
        }
        force = true; // the style takes the new or rewritten LUT
      }
    }
    const float want = on && lookOn ? 1.f : 0.f;
    if (set.log && want != loggedWant && (want == 0.f || loggedWant == 0.f)) {
      Log.info("Room grade fading {} (to {:.2f})", want > weight ? "in" : "out", want);
      loggedWant = want;
    }
    if (weight != want) {
      const float step = dt / set.fade;
      weight = want > weight ? std::min(weight + step, want) : std::max(weight - step, want);
    }
    // Eased, so the fade starts and ends softly.
    const float shown = smoothstep(0.f, 1.f, weight);
    if (force || std::abs(shown - appliedWeight) >= 0.004f || (shown == 0.f) != (appliedWeight == 0.f)) {
      apply(shown);
      force = false;
    }
  }
  // Plain again before the LUTs go (the session is ending).
  if (target || appliedWeight > 0.f) {
    XrPassthroughStyleFB plain{XR_TYPE_PASSTHROUGH_STYLE_FB};
    plain.textureOpacityFactor = 1.f;
    std::lock_guard lock{g_passthroughMutex};
    B.setPassthroughStyle(B.passthroughLayer, &plain);
  }
  if (target)
    B.destroyColorLut(target);
  if (identity)
    B.destroyColorLut(identity);
}

void start_grade_worker() {
  if (g_gradeThread.joinable() || !grade_settings().on || !B.passthroughLayer || !B.setPassthroughStyle)
    return;
  {
    std::lock_guard lock{g_grade.mutex};
    g_grade.stop = false;
  }
  Log.info("Room grade: {}", grade_settings().lut > 0 ? fmt::format("colour LUTs, {}^3", grade_settings().lut)
                                                      : std::string{"brightness and saturation"});
  g_gradeThread = std::thread(grade_main);
}

// Before the passthrough layer goes.
void stop_grade_worker() {
  if (!g_gradeThread.joinable())
    return;
  {
    std::lock_guard lock{g_grade.mutex};
    g_grade.stop = true;
  }
  g_grade.cv.notify_one();
  g_gradeThread.join();
}

// XR thread, every frame: whether a fight (not held for placing) shows over
// the room.
void grade_frame(bool on) {
  static bool was = false;
  if (on == was || !g_gradeThread.joinable())
    return;
  was = on;
  std::lock_guard lock{g_grade.mutex};
  g_grade.on = on;
  g_grade.cv.notify_one();
}

// ---------------------------------------------------------------- XR thread: placing the stage
//
// While the game holds a fight before it starts (aurora_xr_set_placing), the
// how-to cards float above and behind where the arena starts, and the button
// legend just below its front edge, both turned to face the starting head
// position. They stay put while the player moves the arena.

// Premultiplied for the compositor, in the swapchain's encoding.
bool upload_placing_image(PlacingImage& img) {
  const bool srgb = B.swapFormat == VK_FORMAT_R8G8B8A8_SRGB;
  std::array<float, 256> decode;
  for (int i = 0; i < 256; ++i) {
    const float c = i / 255.f;
    decode[i] = c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
  }
  constexpr int kSteps = 4096;
  std::vector<uint8_t> encode(kSteps + 1);
  for (int i = 0; i <= kSteps; ++i) {
    const float l = static_cast<float>(i) / kSteps;
    const float c = !srgb ? l : l <= 0.0031308f ? l * 12.92f : 1.055f * std::pow(l, 1.f / 2.4f) - 0.055f;
    encode[i] = static_cast<uint8_t>(std::lround(c * 255.f));
  }
  std::vector<uint8_t> px(img.rgba.size());
  for (size_t i = 0; i + 3 < px.size(); i += 4) {
    const float a = img.rgba[i + 3] / 255.f;
    for (int c = 0; c < 3; ++c)
      px[i + c] = encode[static_cast<int>(std::lround(decode[img.rgba[i + c]] * a * kSteps))];
    px[i + 3] = img.rgba[i + 3];
  }
  return create_static_swapchain(img.swapchain, img.width, img.height, px);
}

constexpr size_t kMaxPlacingLayers = 2 + 3; // the cards, the legend, the clips

struct PlacingLayers {
  std::array<XrCompositionLayerQuad, kMaxPlacingLayers> quads;
  uint32_t count = 0;
};

void build_placing_layers(PlacingLayers& out, int mode) {
  out.count = 0;
  const ArenaPose d = [] {
    std::lock_guard lock{g_arenaMutex};
    init_arena();
    return g_defaultArena;
  }();
  // Where each sits against the arena's starting position, and its width
  // (meters). AURORA_XR_PLACE_CARDS / _LEGEND: "dx,dy,dz,width".
  struct Spot {
    const char* env;
    XrVector3f offset;
    float width;
  };
  const std::array<Spot, 2> spots{{{"AURORA_XR_PLACE_CARDS", {0.f, 0.42f, -0.3f}, 0.85f},
                                    {"AURORA_XR_PLACE_LEGEND", {0.f, -0.17f, 0.12f}, 0.34f}}};
  // Soft head lock, sideways only: the layout turns about where the head
  // stood when the hold began, lagging behind once the head has turned past
  // a dead zone, and settles when it catches up. Height and pitch stay put.
  static auto last = std::chrono::steady_clock::time_point{};
  static float yaw = 0.f;
  static bool following = false;
  static XrVector3f pivot{};
  const auto now = std::chrono::steady_clock::now();
  const float dt = std::chrono::duration<float>(now - last).count();
  last = now;
  const XrVector3f fwd = qrot(B.headOrientation, {0.f, 0.f, -1.f});
  const float headYaw = std::atan2(-fwd.x, -fwd.z); // rot_y's sense
  if (dt > 0.25f) { // a new hold: start where the head looks
    yaw = headYaw;
    following = false;
    pivot = {B.head.x, 0.f, B.head.z};
  } else {
    constexpr float kPi = 3.14159265f;
    float diff = headYaw - yaw;
    diff -= 2.f * kPi * std::floor((diff + kPi) / (2.f * kPi));
    if (std::fabs(diff) > 20.f * kPi / 180.f)
      following = true;
    if (following) {
      yaw += diff * std::min(1.f, dt * 4.f);
      if (std::fabs(diff) < 1.f * kPi / 180.f)
        following = false;
    }
  }
  // Hands, if one is tracked (a controller put down), else controllers.
  const bool hands = G.hands[0].tracked || G.hands[1].tracked;
  std::lock_guard lock{g_placingMutex};
  const auto ready = [](PlacingImage& img, const char* what) {
    if (img.width == 0)
      return false;
    if (!img.swapchain && !upload_placing_image(img)) {
      Log.warn("Placing picture {} unavailable", what);
      img.width = 0;
      return false;
    }
    return true;
  };
  for (int which = mode >= 2 ? 0 : 1; which < 2; ++which) {
    // The hands' version when there is one, else the controllers'.
    PlacingImage* img = &g_placingImages[which];
    if (hands && g_placingImages[which + 2].width > 0)
      img = &g_placingImages[which + 2];
    if (!ready(*img, which == 0 ? "cards" : "legend"))
      continue;
    Spot spot = spots[which];
    if (const char* v = std::getenv(spot.env))
      std::sscanf(v, "%f,%f,%f,%f", &spot.offset.x, &spot.offset.y, &spot.offset.z, &spot.width);
    const XrVector3f pos = pivot + rot_y(d.pos + spot.offset, yaw);
    const XrQuaternionf turn = facing(pos, pivot);
    const XrExtent2Df size{spot.width, spot.width * static_cast<float>(img->height) / static_cast<float>(img->width)};
    auto& q = out.quads[out.count++];
    q = {XR_TYPE_COMPOSITION_LAYER_QUAD};
    q.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
    q.space = B.space;
    q.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
    q.subImage.swapchain = img->swapchain;
    q.subImage.imageRect = {{0, 0}, {static_cast<int32_t>(img->width), static_cast<int32_t>(img->height)}};
    q.pose = {turn, pos};
    q.size = size;
    if (which != 0)
      continue;
    // The clips for these cards, each over its picture, a hair in front.
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    const size_t first = img == &g_placingImages[2] ? 3 : 0;
    for (size_t c = first; c < first + 3; ++c) {
      PlacingClip& clip = g_placingClips[c];
      if (clip.frames <= 0 || !ready(clip.atlas, "clip"))
        continue;
      const int period = std::max(1, 2 * (clip.frames - 1));
      const int step = static_cast<int>(std::fmod(t * clip.fps, static_cast<double>(period)));
      const int frame = step < clip.frames ? step : period - step;
      const float u = (clip.x + clip.w * 0.5f) / static_cast<float>(img->width) - 0.5f;
      const float v = 0.5f - (clip.y + clip.h * 0.5f) / static_cast<float>(img->height);
      auto& cq = out.quads[out.count++];
      cq = q;
      cq.layerFlags = 0; // opaque
      cq.subImage.swapchain = clip.atlas.swapchain;
      cq.subImage.imageRect = {{(frame % clip.cols) * clip.frameW, (frame / clip.cols) * clip.frameH},
                               {clip.frameW, clip.frameH}};
      cq.pose.position = pos + qrot(turn, {u * size.width, v * size.height, 0.002f});
      cq.size = {size.width * clip.w / static_cast<float>(img->width),
                 size.height * clip.h / static_cast<float>(img->height)};
    }
  }
}

// ---------------------------------------------------------------- XR thread: the dev tools panel
//
// For tuning and A/Bs on the headset (AURORA_XR_DEV_TOOLS=1, a debug aid): a
// panel floats beside the screen (or, if it first shows with a fight in 3D,
// right of where the head looked), with a page per tab, always there. A
// laser on it shows, in play too, and the trigger (a pinch for a tracked
// hand) picks a tab or a choice and drags a slider, instead of pressing L
// or R; Reset gives the page's settings back as
// they were without it. It moves as the screen does: a controller's grip
// with its laser on it, or the bar under it pointed at or touched and held
// with a grip, trigger or pinch; the other hand pressing on it as well
// resizes it. Once moved it stays where it was put, for the session. A page shows only with something on it: the
// game's options, the room's grade on, audio settings or a stage's mixer
// from the game.
//
// Game, the game's own options (aurora_xr_dev_option_add), a row of
// choices each, which the game reads (aurora_xr_dev_option).
//
// Tint, in mixed reality with the room's grade on (AURORA_XR_GRADE=1): the
// tint as a colour in HSL (hue, tint: its saturation, light: its lightness;
// that colour is the gains per channel) and the room's own saturation. The
// room follows within about 100 ms. What it sets replaces the game's grade
// for that stage until the app quits; each change is logged as a row for
// s_grades (src/pc/xr_scene.c).
//
// Audio, which the game applies (aurora_xr_dev_audio; src/pc/spatial_audio.c):
// spatial audio on, or bypassed for every sound to play plain stereo; the
// music's level, to hear the rest without it.
//
// Stage, which the game applies too (aurora_xr_dev_mix; src/pc/stage_mix.c):
// a mixer for the current stage's own sounds, a channel at a time picked
// from the row of its names: its level, and a three-band parametric EQ on a
// graph of the channel's response, each band a handle dragged across
// (frequency) and up and down (gain), the last one grabbed taking the Q
// slider. Reset puts the channel back. Each change is logged as a
// MELEE_STAGE_MIX value for the whole stage, which the game reads at launch.
//
// AURORA_XR_DEV_TOOLS_POS "dx,dy,dz,width" (0.3,-0.15,-0.55,0.36): where it
// floats against the head, in meters.

struct Hsl {
  float h = 0.f, s = 0.f, l = 1.f; // hue in degrees
};

// Gains of 0 to 1 as HSL. A grey keeps `hue`, so the hue slider stays put.
Hsl rgb_to_hsl(float r, float g, float b, float hue) {
  r = std::clamp(r, 0.f, 1.f);
  g = std::clamp(g, 0.f, 1.f);
  b = std::clamp(b, 0.f, 1.f);
  const float mx = std::max({r, g, b}), mn = std::min({r, g, b}), d = mx - mn;
  Hsl c{hue, 0.f, (mx + mn) * 0.5f};
  if (d < 1e-4f)
    return c;
  c.s = std::min(d / (1.f - std::fabs(2.f * c.l - 1.f)), 1.f);
  c.h = mx == r ? 60.f * std::fmod((g - b) / d + 6.f, 6.f) : mx == g ? 60.f * ((b - r) / d + 2.f)
                                                                   : 60.f * ((r - g) / d + 4.f);
  return c;
}

std::array<float, 3> hsl_to_rgb(const Hsl& c) {
  const float k = (1.f - std::fabs(2.f * c.l - 1.f)) * c.s;
  const float hp = std::fmod(std::max(c.h, 0.f), 360.f) / 60.f;
  const float x = k * (1.f - std::fabs(std::fmod(hp, 2.f) - 1.f));
  const float m = c.l - k * 0.5f;
  switch (static_cast<int>(hp)) {
  case 0:
    return {k + m, x + m, m};
  case 1:
    return {x + m, k + m, m};
  case 2:
    return {m, k + m, x + m};
  case 3:
    return {m, x + m, k + m};
  case 4:
    return {x + m, m, k + m};
  default:
    return {k + m, m, x + m};
  }
}

// A 5x7 pixel font for the panel's labels, rows top to bottom.
struct Glyph {
  char c;
  const char* rows;
};
constexpr Glyph kGlyphs[] = {
    {'0', ".###.#...##..###.#.###..##...#.###."}, {'1', "..#...##....#....#....#....#...###."},
    {'2', ".###.#...#....#...#...#...#...#####"}, {'3', "#####...#...#.....#....##...#.###."},
    {'4', "...#...##..#.#.#..#.#####...#....#."}, {'5', "######....####.....#....##...#.###."},
    {'6', "..##..#...#....####.#...##...#.###."}, {'7', "#####....#...#...#...#....#....#..."},
    {'8', ".###.#...##...#.###.#...##...#.###."}, {'9', ".###.#...##...#.####....#...#..##.."},
    {'A', ".###.#...##...#######...##...##...#"}, {'B', "####.#...##...#####.#...##...#####."},
    {'C', ".###.#...##....#....#....#...#.###."}, {'D', "###..#..#.#...##...##...##..#.###.."},
    {'E', "######....#....####.#....#....#####"}, {'F', "######....#....####.#....#....#...."},
    {'G', ".###.#...##....#.####...##...#.####"}, {'H', "#...##...##...#######...##...##...#"},
    {'I', ".###...#....#....#....#....#...###."}, {'J', "..###...#....#....#....#.#..#..##.."},
    {'K', "#...##..#.#.#..##...#.#..#..#.#...#"}, {'L', "#....#....#....#....#....#....#####"},
    {'M', "#...###.###.#.##.#.##...##...##...#"}, {'N', "#...##...###..##.#.##..###...##...#"},
    {'O', ".###.#...##...##...##...##...#.###."}, {'P', "####.#...##...#####.#....#....#...."},
    {'Q', ".###.#...##...##...##.#.##..#..##.#"}, {'R', "####.#...##...#####.#.#..#..#.#...#"},
    {'S', ".#####....#.....###.....#....#####."}, {'T', "#####..#....#....#....#....#....#.."},
    {'U', "#...##...##...##...##...##...#.###."}, {'V', "#...##...##...##...##...#.#.#...#.."},
    {'W', "#...##...##...##.#.##.#.##.#.#.#.#."}, {'X', "#...##...#.#.#...#...#.#.#...##...#"},
    {'Y', "#...##...#.#.#...#....#....#....#.."}, {'Z', "#####....#...#...#...#...#....#####"},
    {'.', ".........................##...##..."}, {'-', "...............#####..............."},
    {'x', "..........#...#.#.#...#...#.#.#...#"}, {':', "......##...##.......##...##........"},
    {'_', "..............................#####"}, {'=', "..........#####.....#####.........."},
    {'+', ".....""..#..""..#..""#####""..#..""..#.."".....",},
};

using Rgba = std::array<uint8_t, 4>;

struct Canvas {
  int w, h;
  std::vector<uint8_t> px; // straight alpha, sRGB

  Canvas(int width, int height) : w(width), h(height), px(static_cast<size_t>(width) * height * 4, 0) {}
  void fill(int x0, int y0, int x1, int y1, Rgba c) {
    for (int y = std::max(y0, 0); y < std::min(y1, h); ++y)
      for (int x = std::max(x0, 0); x < std::min(x1, w); ++x)
        std::memcpy(&px[(static_cast<size_t>(y) * w + x) * 4], c.data(), 4);
  }
  // Text at `scale` pixels per font pixel; returns where it ends.
  int text(int x, int y, int scale, std::string_view s, Rgba c) {
    for (char ch : s) {
      for (const Glyph& g : kGlyphs) {
        if (g.c != ch)
          continue;
        for (int r = 0; r < 7; ++r)
          for (int col = 0; col < 5; ++col)
            if (g.rows[r * 5 + col] == '#')
              fill(x + col * scale, y + r * scale, x + (col + 1) * scale, y + (r + 1) * scale, c);
      }
      x += (ch == '.' ? 3 : 6) * scale; // a narrow point
    }
    return x;
  }
};

Rgba linear_rgba(float r, float g, float b) {
  return {static_cast<uint8_t>(std::lround(srgb_encode(r) * 255.f)),
          static_cast<uint8_t>(std::lround(srgb_encode(g) * 255.f)),
          static_cast<uint8_t>(std::lround(srgb_encode(b) * 255.f)), 255};
}

constexpr int kPanelW = 768, kPanelH = 480;
constexpr int kTrackX0 = 170, kTrackX1 = 590, kTrackH = 30;
constexpr int kRowY0 = 76, kRowStep = 86;
constexpr int kResetX0 = 604, kResetX1 = 750, kResetY0 = 10, kResetY1 = 52;
enum { kHue, kTint, kLight, kSat, kSliders };
constexpr std::array<const char*, kSliders> kSliderNames{"HUE", "TINT", "LIGHT", "SAT"};
constexpr std::array<float, kSliders> kSliderMax{360.f, 1.f, 1.f, 1.5f};

// The pages, as tabs left of Reset.
enum { kPageGame, kPageTint, kPageAudio, kPageMix, kPages };
constexpr std::array<const char*, kPages> kPageNames{"GAME", "TINT", "AUDIO", "STAGE"};
constexpr int kTabX0 = 10, kTabW = 150, kTabGap = 10;

// The Audio page: choices, a row each.
enum { kSpatial, kMusic, kAudioRows };
constexpr int kAudioRowY0 = 80, kAudioRowStep = 70, kChoiceX1 = 750;
constexpr std::array<const char*, kAudioRows> kAudioNames{"SPATIAL", "MUSIC"};
constexpr std::array<const char*, 2> kSpatialNames{"ON", "OFF"};

// The Stage page: the channels' names in a row, the channel's level, the
// graph of its EQ (20 Hz to 16 kHz across, +-18 dB up and down) with a
// handle a band, and the Q of the band last grabbed.
constexpr int kMixChanY0 = 64, kMixChanY1 = 102, kMixLevelY = 122, kMixQY = 390;
constexpr int kGraphX0 = 64, kGraphX1 = 750, kGraphY0 = 172, kGraphY1 = 352;
constexpr float kGraphFMin = 20.f, kGraphFMax = 16000.f, kGraphDb = 18.f;
constexpr float kMixLevelMax = 18.f, kMixQMin = 0.3f, kMixQMax = 10.f;
constexpr float kMixRate = 32000.f; // the game's mixing rate (AX's), for the graph
constexpr std::array<const char*, AURORA_XR_MIX_BANDS> kBandNames{"LOW", "MID", "HIGH"};
constexpr std::array<Rgba, AURORA_XR_MIX_BANDS> kBandColors{Rgba{255, 170, 80, 255}, Rgba{120, 220, 140, 255},
                                                            Rgba{190, 150, 255, 255}};

// What's under a laser: a tint slider (kHue..kSat), an Audio row (kAudio +
// row), a tab (kTab + page), Reset, a Stage channel (kMixChan + channel),
// its level, its graph or its Q, a Game option (kOption + option), or -1.
constexpr int kAudio = 10, kTab = 20, kReset = 30, kMixChan = 40, kMixLevel = 50, kMixGraph = 51, kMixQ = 52,
              kOption = 60;

struct DevPanel {
  bool shown = false;
  XrPosef pose{{0.f, 0.f, 0.f, 1.f}, {}};
  float width = 0.36f;
  int page = -1; // a kPage; -1 until first opened
  bool tintUsable = false; // the grade is on and the room shows
  int stage = -1;
  std::array<float, kSliders> value{0.f, 0.f, 1.f, 1.f};
  int hover = -1;      // a target
  int active = -1;     // the slider being dragged (a target)
  int activeHand = -1; // the hand dragging it
  int mixChannel = 0;  // the Stage page's channel
  int mixBand = 1;     // its band last grabbed, which the Q slider sets
  float grabDx = 0.f, grabDy = 0.f; // from the laser to the grabbed handle, in pixels
  bool dirty = true;   // the picture needs drawing
  std::chrono::steady_clock::time_point drawn{};
  // Moving it: the hand moving it alone (from the bar by touch, or along its
  // laser at moveDist), or both hands resizing it from width0 and pos0 as
  // their touch points spread from span0 about mid0.
  bool placed = false; // moved by the player: it stays where they put it
  int moveHand = -1;
  bool moveDirect = false, sizing = false;
  float moveDist = 0.f, span0 = 1.f, width0 = 0.36f;
  XrVector3f moveOffset{}, mid0{}, pos0{};
  bool barHover = false;
  PlacingImage image;
  std::deque<std::pair<XrSwapchain, std::chrono::steady_clock::time_point>> retired;
} P;

// The Audio page's settings: the game's (aurora_xr_dev_audio_defaults) and
// the panel's, which the game reads (aurora_xr_dev_audio).
struct DevAudio {
  std::mutex mutex;
  bool have = false;
  AuroraXrDevAudio defaults{}, value{};
} g_devAudio;

// The Stage page's mixer (aurora_xr_dev_mix_stage): each stage's channels,
// the game's settings and the panel's, which the game reads
// (aurora_xr_dev_mix); `stage` the one playing.
struct DevMixStage {
  int count = 0;
  std::array<std::string, AURORA_XR_MIX_CHANNELS> names;
  std::array<AuroraXrMixChannel, AURORA_XR_MIX_CHANNELS> defaults{}, value{};
};
struct DevMix {
  std::mutex mutex;
  int stage = -1;
  std::map<int, DevMixStage> stages;
} g_devMix;
std::atomic<bool> g_devMixListening{false}; // the panel shows its Stage page

// The Game page's rows: options (aurora_xr_dev_option_add) and sliders
// (aurora_xr_dev_slider_add); each one's value (a choice's index, or the
// slider's), which the game reads (aurora_xr_dev_option, _slider), apart for
// a cheap read.
struct DevOption {
  std::string name;
  std::vector<std::string> choices; // none: a slider
  float initial = 0.f;
  float min = 0.f, max = 1.f, step = 0.f;
  bool slider() const { return choices.empty(); }
};
struct DevOptions {
  std::mutex mutex;
  std::vector<DevOption> options;
} g_devOptions;
std::array<std::atomic<float>, AURORA_XR_DEV_OPTIONS> g_devOptionValue{};
std::atomic<bool> g_devOptionsAdded{false}; // since the panel last looked: it draws again

bool dev_tools_wanted() {
  static const bool on = [] {
    // The panel's first name, which also turned the grade on: no longer read.
    if (std::getenv("AURORA_XR_GRADE_PANEL") != nullptr)
      Log.warn("AURORA_XR_GRADE_PANEL is no longer read: AURORA_XR_DEV_TOOLS=1 opens the dev tools panel, "
               "AURORA_XR_GRADE=1 turns the room grade on");
    return env_flag("AURORA_XR_DEV_TOOLS", false);
  }();
  return on;
}

RoomGrade panel_grade() {
  const auto rgb = hsl_to_rgb({P.value[kHue], P.value[kTint], P.value[kLight]});
  return {rgb[0], rgb[1], rgb[2], P.value[kSat]};
}

void panel_from_grade(const RoomGrade& g) {
  const Hsl c = rgb_to_hsl(g.r, g.g, g.b, P.value[kHue]);
  P.value = {c.h, c.s, c.l, std::clamp(g.saturation, 0.f, kSliderMax[kSat])};
  P.dirty = true;
}

void panel_log(const char* what) {
  const RoomGrade g = panel_grade();
  const float strength = grade_settings().strength;
  Log.info("Room grade panel: {} stage 0x{:02X}: {{0x{:02X}, {:.2f}f, {:.2f}f, {:.2f}f, {:.2f}f}} (hue {:.0f}, tint "
           "{:.2f}, light {:.2f}){}",
           what, P.stage, P.stage, g.r, g.g, g.b, g.saturation, P.value[kHue], P.value[kTint], P.value[kLight],
           strength != 1.f ? fmt::format(", shown at strength x{:.2f}", strength) : std::string{});
}

void audio_log() {
  AuroraXrDevAudio a;
  {
    std::lock_guard lock{g_devAudio.mutex};
    a = g_devAudio.value;
  }
  Log.info("Dev tools: audio: spatial {}, music {:.0f}%", a.spatial ? "on" : "off", a.music * 100.f);
}

// The pages with something on them, in tab order; the Game page when none
// has, to say so.
std::vector<int> shown_pages() {
  std::vector<int> pages;
  {
    std::lock_guard lock{g_devOptions.mutex};
    if (!g_devOptions.options.empty())
      pages.push_back(kPageGame);
  }
  if (P.tintUsable)
    pages.push_back(kPageTint);
  {
    std::lock_guard lock{g_devAudio.mutex};
    if (g_devAudio.have)
      pages.push_back(kPageAudio);
  }
  {
    std::lock_guard lock{g_devMix.mutex};
    if (!g_devMix.stages.empty())
      pages.push_back(kPageMix);
  }
  if (pages.empty())
    pages.push_back(kPageGame);
  return pages;
}

void draw_header(Canvas& c) {
  const Rgba text{232, 236, 242, 255}, dim{150, 156, 168, 255};
  const std::vector<int> pages = shown_pages();
  for (size_t i = 0; i < pages.size(); ++i) {
    const int page = pages[i];
    const int x0 = kTabX0 + static_cast<int>(i) * (kTabW + kTabGap);
    const bool on = P.page == page;
    c.fill(x0, kResetY0, x0 + kTabW, kResetY1,
           on ? Rgba{70, 110, 150, 255} : P.hover == kTab + page ? Rgba{60, 66, 80, 255} : Rgba{36, 40, 50, 255});
    const int w = static_cast<int>(std::strlen(kPageNames[page])) * 18 - 3;
    c.text(x0 + (kTabW - w) / 2, kResetY0 + 11, 3, kPageNames[page], on ? text : dim);
  }
  if (P.page == kPageTint && pages.size() <= 3)
    c.text(kTabX0 + static_cast<int>(pages.size()) * (kTabW + kTabGap) + 8, kResetY0 + 11, 3,
           fmt::format("0x{:02X}", P.stage), dim);
  c.fill(kResetX0, kResetY0, kResetX1, kResetY1, P.hover == kReset ? Rgba{70, 110, 150, 255} : Rgba{52, 56, 68, 255});
  c.text(kResetX0 + (kResetX1 - kResetX0 - 87) / 2, kResetY0 + 11, 3, "RESET", text);
}

void draw_tint(Canvas& c) {
  const Rgba text{232, 236, 242, 255}, dim{150, 156, 168, 255}, lit{120, 200, 255, 255};
  if (!P.tintUsable) {
    c.text(18, kRowY0 + 10, 3, "THE ROOM TINT IS OFF.", text);
    c.text(18, kRowY0 + 60, 3, "IT NEEDS MIXED REALITY AND", dim);
    c.text(18, kRowY0 + 100, 3, "AURORA_XR_GRADE=1 AT LAUNCH.", dim);
    return;
  }
  const Hsl tint{P.value[kHue], P.value[kTint], P.value[kLight]};
  for (int i = 0; i < kSliders; ++i) {
    const int y = kRowY0 + i * kRowStep;
    c.text(18, y + 4, 3, kSliderNames[i], i == P.active ? lit : i == P.hover ? text : dim);
    for (int x = kTrackX0; x < kTrackX1; ++x) {
      const float t = static_cast<float>(x - kTrackX0) / static_cast<float>(kTrackX1 - kTrackX0 - 1);
      std::array<float, 3> rgb;
      if (i == kHue) {
        rgb = hsl_to_rgb({t * 360.f, 1.f, 0.5f});
      } else if (i == kTint) {
        rgb = hsl_to_rgb({tint.h, t, 0.5f});
      } else if (i == kLight) {
        rgb = hsl_to_rgb({tint.h, tint.s, t});
      } else {
        // The tint's hue at full strength, desaturated toward its luminance.
        const auto full = hsl_to_rgb({tint.h, 1.f, 0.5f});
        const float y = 0.2126f * full[0] + 0.7152f * full[1] + 0.0722f * full[2];
        const float k = t * kSliderMax[kSat];
        for (int ch = 0; ch < 3; ++ch)
          rgb[ch] = std::clamp(y + (full[ch] - y) * k, 0.f, 1.f);
      }
      c.fill(x, y, x + 1, y + kTrackH, linear_rgba(rgb[0], rgb[1], rgb[2]));
    }
    const int kx = kTrackX0 + static_cast<int>(std::lround(P.value[i] / kSliderMax[i] * (kTrackX1 - kTrackX0)));
    c.fill(kx - 6, y - 9, kx + 6, y + kTrackH + 9, {10, 10, 12, 255});
    c.fill(kx - 3, y - 6, kx + 3, y + kTrackH + 6, i == P.active ? lit : Rgba{250, 250, 250, 255});
    c.text(kTrackX1 + 22, y + 4, 3, i == kHue ? fmt::format("{:.0f}", P.value[i]) : fmt::format("{:.2f}", P.value[i]),
           text);
  }
  const RoomGrade g = panel_grade();
  c.text(18, kPanelH - 44, 3, fmt::format("R {:.2f}  G {:.2f}  B {:.2f}", g.r, g.g, g.b), dim);
  c.fill(kTrackX1 + 16, kPanelH - 56, kResetX1, kPanelH - 14, linear_rgba(g.r, g.g, g.b));
}

void draw_audio(Canvas& c) {
  const Rgba text{232, 236, 242, 255}, dim{150, 156, 168, 255}, lit{120, 200, 255, 255};
  AuroraXrDevAudio a;
  bool have;
  {
    std::lock_guard lock{g_devAudio.mutex};
    a = g_devAudio.value;
    have = g_devAudio.have;
  }
  if (!have) {
    c.text(18, kAudioRowY0 + 10, 3, "NO AUDIO SETTINGS FROM THE GAME.", dim);
    return;
  }
  for (int row = 0; row < kAudioRows; ++row) {
    const int y = kAudioRowY0 + row * kAudioRowStep;
    const int target = kAudio + row;
    c.text(18, y + 4, 3, kAudioNames[row], target == P.active ? lit : target == P.hover ? text : dim);
    if (row == kMusic) {
      const int kx = kTrackX0 + static_cast<int>(std::lround(std::clamp(a.music, 0.f, 1.f) * (kTrackX1 - kTrackX0)));
      c.fill(kTrackX0, y + 10, kTrackX1, y + kTrackH - 10, {52, 56, 68, 255});
      c.fill(kTrackX0, y + 10, kx, y + kTrackH - 10, {70, 110, 150, 255});
      c.fill(kx - 6, y - 9, kx + 6, y + kTrackH + 9, {10, 10, 12, 255});
      c.fill(kx - 3, y - 6, kx + 3, y + kTrackH + 6, target == P.active ? lit : Rgba{250, 250, 250, 255});
      c.text(kTrackX1 + 22, y + 4, 3, fmt::format("{:.0f}", a.music * 100.f), text);
      continue;
    }
    const int n = 2, chosen = a.spatial ? 0 : 1;
    const int w = (kChoiceX1 - kTrackX0) / n;
    for (int i = 0; i < n; ++i) {
      const int x0 = kTrackX0 + i * w;
      const char* name = kSpatialNames[i];
      c.fill(x0 + 2, y - 4, x0 + w - 2, y + kTrackH + 4, i == chosen ? Rgba{70, 110, 150, 255} : Rgba{36, 40, 50, 255});
      const int tw = static_cast<int>(std::strlen(name)) * 18 - 3;
      c.text(x0 + (w - tw) / 2, y + 4, 3, name, i == chosen ? text : dim);
    }
  }
  c.text(18, kAudioRowY0 + kAudioRows * kAudioRowStep + 10, 3, "SPATIAL OFF: EVERY SOUND PLAIN STEREO.", dim);
  c.text(18, kAudioRowY0 + kAudioRows * kAudioRowStep + 50, 3, "MUSIC: PERCENT OF ITS USUAL LEVEL.", dim);
}

// A row of choices at y from kTrackX0 to kChoiceX1, `chosen` lit.
void draw_choices(Canvas& c, int y, const std::vector<std::string>& names, int chosen) {
  const Rgba text{232, 236, 242, 255}, dim{150, 156, 168, 255};
  const int n = std::max(static_cast<int>(names.size()), 1);
  const int w = (kChoiceX1 - kTrackX0) / n;
  for (int i = 0; i < static_cast<int>(names.size()); ++i) {
    const int x0 = kTrackX0 + i * w;
    c.fill(x0 + 2, y - 4, x0 + w - 2, y + kTrackH + 4, i == chosen ? Rgba{70, 110, 150, 255} : Rgba{36, 40, 50, 255});
    const int tw = static_cast<int>(names[i].size()) * 18 - 3;
    c.text(x0 + (w - tw) / 2, y + 4, 3, names[i], i == chosen ? text : dim);
  }
}

void draw_game(Canvas& c) {
  const Rgba text{232, 236, 242, 255}, dim{150, 156, 168, 255};
  std::vector<DevOption> options;
  {
    std::lock_guard lock{g_devOptions.mutex};
    options = g_devOptions.options;
  }
  if (options.empty()) {
    c.text(18, kAudioRowY0 + 10, 3, "NOTHING ON THE PANEL YET.", dim);
    return;
  }
  const Rgba lit{120, 200, 255, 255};
  for (int row = 0; row < static_cast<int>(options.size()); ++row) {
    const DevOption& o = options[row];
    const int y = kAudioRowY0 + row * kAudioRowStep;
    const int target = kOption + row;
    const float v = g_devOptionValue[row].load(std::memory_order_relaxed);
    c.text(18, y + 4, 3, o.name, target == P.active ? lit : target == P.hover ? text : dim);
    if (!o.slider()) {
      draw_choices(c, y, o.choices, static_cast<int>(v));
      continue;
    }
    const float t = std::clamp((v - o.min) / (o.max - o.min), 0.f, 1.f);
    const int kx = kTrackX0 + static_cast<int>(std::lround(t * (kTrackX1 - kTrackX0)));
    c.fill(kTrackX0, y + 10, kTrackX1, y + kTrackH - 10, {52, 56, 68, 255});
    c.fill(kTrackX0, y + 10, kx, y + kTrackH - 10, {70, 110, 150, 255});
    c.fill(kx - 6, y - 9, kx + 6, y + kTrackH + 9, {10, 10, 12, 255});
    c.fill(kx - 3, y - 6, kx + 3, y + kTrackH + 6, target == P.active ? lit : Rgba{250, 250, 250, 255});
    c.text(kTrackX1 + 22, y + 4, 3, fmt::format("{:.2f}", v), text);
  }
}

// The Stage page's scales.
float graph_freq(float px) {
  const float t = std::clamp((px - kGraphX0) / static_cast<float>(kGraphX1 - kGraphX0), 0.f, 1.f);
  return kGraphFMin * std::pow(kGraphFMax / kGraphFMin, t);
}
float graph_x(float freq) {
  return kGraphX0 + std::log(std::clamp(freq, kGraphFMin, kGraphFMax) / kGraphFMin) /
                        std::log(kGraphFMax / kGraphFMin) * (kGraphX1 - kGraphX0);
}
float graph_db(float py) {
  return (0.5f - (py - kGraphY0) / static_cast<float>(kGraphY1 - kGraphY0)) * 2.f * kGraphDb;
}
float graph_y(float db) { return kGraphY0 + (0.5f - db / (2.f * kGraphDb)) * (kGraphY1 - kGraphY0); }
float level_t(float db) { return (db - AURORA_XR_MIX_OFF) / (kMixLevelMax - AURORA_XR_MIX_OFF); }
float q_t(float q) { return std::log(q / kMixQMin) / std::log(kMixQMax / kMixQMin); }

std::string mix_freq(float f) { return f < 1000.f ? fmt::format("{:.0f}", f) : fmt::format("{:.1f}K", f / 1000.f); }
std::string mix_db(float db) { return fmt::format("{:+.1f}", db); }
std::string mix_level(float db) { return db <= AURORA_XR_MIX_OFF ? std::string{"OFF"} : mix_db(db); }

// A band's filter as the game makes it (RBJ's cookbook; src/pc/stage_mix.c)
// and its response in dB at `f`.
struct MixBiquad {
  double b0, b1, b2, a1, a2;
};
MixBiquad mix_biquad(int kind, const AuroraXrMixBand& m) {
  const double rate = kMixRate;
  const double f = std::clamp(static_cast<double>(m.freq), 10.0, rate * 0.45);
  const double q = std::max(static_cast<double>(m.q), 0.1);
  const double A = std::pow(10.0, m.gain / 40.0);
  const double w0 = 6.283185307179586 * f / rate;
  const double cw = std::cos(w0), alpha = std::sin(w0) / (2.0 * q);
  double b0, b1, b2, a0, a1, a2;
  if (kind == 1) {
    b0 = 1.0 + alpha * A, b1 = -2.0 * cw, b2 = 1.0 - alpha * A;
    a0 = 1.0 + alpha / A, a1 = -2.0 * cw, a2 = 1.0 - alpha / A;
  } else {
    const double sa = 2.0 * std::sqrt(A) * alpha, s = kind == 0 ? 1.0 : -1.0;
    b0 = A * ((A + 1.0) - s * (A - 1.0) * cw + sa);
    b1 = s * 2.0 * A * ((A - 1.0) - s * (A + 1.0) * cw);
    b2 = A * ((A + 1.0) - s * (A - 1.0) * cw - sa);
    a0 = (A + 1.0) + s * (A - 1.0) * cw + sa;
    a1 = -s * 2.0 * ((A - 1.0) + s * (A + 1.0) * cw);
    a2 = (A + 1.0) + s * (A - 1.0) * cw - sa;
  }
  return {b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0};
}
double mix_response_db(const MixBiquad& q, double f) {
  const std::complex<double> z1 = std::polar(1.0, -6.283185307179586 * f / kMixRate), z2 = z1 * z1;
  return 20.0 * std::log10(std::abs(q.b0 + q.b1 * z1 + q.b2 * z2) / std::abs(1.0 + q.a1 * z1 + q.a2 * z2));
}

// Each change, as the MELEE_STAGE_MIX value for the stage that keeps it.
void mix_log() {
  std::lock_guard lock{g_devMix.mutex};
  const auto it = g_devMix.stages.find(g_devMix.stage);
  if (it == g_devMix.stages.end() || it->second.count == 0)
    return;
  std::string env;
  for (int c = 0; c < it->second.count; ++c) {
    std::string name = it->second.names[c];
    std::replace(name.begin(), name.end(), ' ', '_');
    const AuroraXrMixChannel& m = it->second.value[c];
    env += fmt::format("{}{}={:.1f}", c > 0 ? ";" : "", name, std::max(m.level, AURORA_XR_MIX_OFF));
    for (const AuroraXrMixBand& b : m.band)
      env += fmt::format("/{:.0f},{:.1f},{:.2f}", b.freq, b.gain, b.q);
  }
  Log.info("Dev tools: stage 0x{:02X} mix: MELEE_STAGE_MIX={}", g_devMix.stage, env);
}

void draw_mix(Canvas& c) {
  const Rgba text{232, 236, 242, 255}, dim{150, 156, 168, 255}, lit{120, 200, 255, 255};
  int stage;
  DevMixStage st;
  bool have;
  {
    std::lock_guard lock{g_devMix.mutex};
    stage = g_devMix.stage;
    const auto it = g_devMix.stages.find(stage);
    have = it != g_devMix.stages.end();
    if (have)
      st = it->second;
  }
  if (!have) {
    c.text(18, kMixChanY0 + 10, 3, "NO STAGE FROM THE GAME YET.", dim);
    return;
  }
  if (st.count == 0) {
    c.text(18, kMixChanY0 + 10, 3, "NOTHING TO MIX ON THIS STAGE.", dim);
    return;
  }
  const int ch = std::clamp(P.mixChannel, 0, st.count - 1);
  const int band = std::clamp(P.mixBand, 0, AURORA_XR_MIX_BANDS - 1);
  const AuroraXrMixChannel& m = st.value[ch];

  // The channels.
  const int w = (kChoiceX1 - 18) / st.count;
  for (int i = 0; i < st.count; ++i) {
    const int x0 = 18 + i * w;
    const bool on = i == ch;
    c.fill(x0 + 2, kMixChanY0, x0 + w - 2, kMixChanY1,
           on ? Rgba{70, 110, 150, 255} : P.hover == kMixChan + i ? Rgba{60, 66, 80, 255} : Rgba{36, 40, 50, 255});
    const int tw = static_cast<int>(st.names[i].size()) * 18 - 3;
    c.text(x0 + (w - tw) / 2, kMixChanY0 + 9, 3, st.names[i], on ? text : dim);
  }

  // Its level.
  {
    const int y = kMixLevelY;
    c.text(18, y + 4, 3, "LEVEL", P.active == kMixLevel ? lit : P.hover == kMixLevel ? text : dim);
    const int kx = kTrackX0 + static_cast<int>(std::lround(std::clamp(level_t(m.level), 0.f, 1.f) *
                                                           (kTrackX1 - kTrackX0)));
    const int k0 = kTrackX0 + static_cast<int>(std::lround(level_t(0.f) * (kTrackX1 - kTrackX0)));
    c.fill(kTrackX0, y + 10, kTrackX1, y + kTrackH - 10, {52, 56, 68, 255});
    c.fill(kTrackX0, y + 10, kx, y + kTrackH - 10, {70, 110, 150, 255});
    c.fill(k0 - 1, y + 4, k0 + 1, y + kTrackH - 4, {150, 156, 168, 255}); // 0 dB
    c.fill(kx - 6, y - 9, kx + 6, y + kTrackH + 9, {10, 10, 12, 255});
    c.fill(kx - 3, y - 6, kx + 3, y + kTrackH + 6, P.active == kMixLevel ? lit : Rgba{250, 250, 250, 255});
    c.text(kTrackX1 + 22, y + 4, 3, mix_level(m.level), text);
  }

  // The graph: its grid, the response, the handles.
  c.fill(kGraphX0, kGraphY0, kGraphX1, kGraphY1, P.hover == kMixGraph ? Rgba{30, 34, 44, 255} : Rgba{26, 29, 37, 255});
  for (const float f : {50.f, 100.f, 200.f, 500.f, 1000.f, 2000.f, 5000.f, 10000.f}) {
    const int x = static_cast<int>(std::lround(graph_x(f)));
    const bool major = f == 100.f || f == 1000.f || f == 10000.f;
    c.fill(x, kGraphY0, x + 1, kGraphY1, major ? Rgba{62, 68, 82, 255} : Rgba{42, 46, 58, 255});
    if (major) {
      const std::string label = f == 100.f ? "100" : f == 1000.f ? "1K" : "10K";
      c.text(x - static_cast<int>(label.size()) * 6 + 1, kGraphY1 + 5, 2, label, dim);
    }
  }
  for (const float db : {-12.f, -6.f, 0.f, 6.f, 12.f}) {
    const int y = static_cast<int>(std::lround(graph_y(db)));
    c.fill(kGraphX0, y, kGraphX1, y + 1, db == 0.f ? Rgba{80, 86, 100, 255} : Rgba{42, 46, 58, 255});
    if (db == 0.f || db == 12.f || db == -12.f) {
      const std::string label = db == 0.f ? "0" : db > 0.f ? "+12" : "-12";
      c.text(kGraphX0 - 8 - static_cast<int>(label.size()) * 12, y - 7, 2, label, dim);
    }
  }
  std::array<MixBiquad, AURORA_XR_MIX_BANDS> bq;
  for (int b = 0; b < AURORA_XR_MIX_BANDS; ++b)
    bq[b] = mix_biquad(b, m.band[b]);
  const int y0 = static_cast<int>(std::lround(graph_y(0.f)));
  int prev = -1;
  for (int x = kGraphX0; x < kGraphX1; ++x) {
    double db = 0.0;
    for (const MixBiquad& q : bq)
      db += mix_response_db(q, graph_freq(x + 0.5f));
    const int y = std::clamp(static_cast<int>(std::lround(graph_y(static_cast<float>(db)))), kGraphY0, kGraphY1 - 1);
    c.fill(x, std::min(y, y0), x + 1, std::max(y, y0) + 1, {44, 70, 98, 255});
    const int a = prev < 0 ? y : prev;
    c.fill(x, std::min(a, y) - 1, x + 1, std::max(a, y) + 2, {140, 200, 255, 255});
    prev = y;
  }
  for (int b = 0; b < AURORA_XR_MIX_BANDS; ++b) {
    const int nx = static_cast<int>(std::lround(graph_x(m.band[b].freq)));
    const int ny = std::clamp(static_cast<int>(std::lround(graph_y(m.band[b].gain))), kGraphY0, kGraphY1);
    if (b == band)
      c.fill(nx - 11, ny - 11, nx + 11, ny + 11, P.active == kMixGraph ? lit : Rgba{250, 250, 250, 255});
    c.fill(nx - 8, ny - 8, nx + 8, ny + 8, {10, 10, 12, 255});
    c.fill(nx - 6, ny - 6, nx + 6, ny + 6, kBandColors[b]);
    const char letter[2] = {kBandNames[b][0], '\0'};
    c.text(nx - 5, ny < kGraphY0 + 34 ? ny + 16 : ny - 30, 2, letter, kBandColors[b]);
  }

  // The Q of the band last grabbed, and where that band is.
  {
    const int y = kMixQY;
    c.text(18, y + 4, 3, "Q", P.active == kMixQ ? lit : P.hover == kMixQ ? text : dim);
    c.text(42, y + 4, 3, kBandNames[band], kBandColors[band]);
    const int kx = kTrackX0 + static_cast<int>(std::lround(std::clamp(q_t(m.band[band].q), 0.f, 1.f) *
                                                           (kTrackX1 - kTrackX0)));
    c.fill(kTrackX0, y + 10, kTrackX1, y + kTrackH - 10, {52, 56, 68, 255});
    c.fill(kx - 6, y - 9, kx + 6, y + kTrackH + 9, {10, 10, 12, 255});
    c.fill(kx - 3, y - 6, kx + 3, y + kTrackH + 6, P.active == kMixQ ? lit : kBandColors[band]);
    c.text(kTrackX1 + 22, y + 4, 3, fmt::format("{:.2f}", m.band[band].q), text);
  }
  const char* shape = band == 0 ? "LOW SHELF" : band == 1 ? "PEAK" : "HIGH SHELF";
  c.text(18, kMixQY + 52, 3,
         fmt::format("{}  {} HZ  {} DB", shape, mix_freq(m.band[band].freq), mix_db(m.band[band].gain)), dim);
}

void draw_panel(Canvas& c) {
  c.fill(0, 0, kPanelW, kPanelH, {18, 20, 26, 225});
  c.fill(0, 0, kPanelW, 3, {90, 96, 110, 255});
  draw_header(c);
  if (P.page == kPageGame)
    draw_game(c);
  else if (P.page == kPageTint)
    draw_tint(c);
  else if (P.page == kPageAudio)
    draw_audio(c);
  else
    draw_mix(c);
}

// Where a laser meets the panel's plane: meters along it, and the point in
// the panel's pixels (outside 0..kPanelW, 0..kPanelH when it misses the
// panel itself). False when it points away from the panel.
bool panel_ray(const Hand& hand, float& t, float& px, float& py) {
  const XrQuaternionf& q = P.pose.orientation;
  const XrVector3f n = qrot(q, {0.f, 0.f, 1.f});
  const float facing = vdot(hand.dir, n);
  if (facing > -1e-3f)
    return false;
  t = vdot(P.pose.position - hand.origin, n) / facing;
  if (t <= 0.f)
    return false;
  const XrVector3f d = hand.origin + hand.dir * t - P.pose.position;
  const float height = P.width * kPanelH / kPanelW;
  px = (vdot(d, qrot(q, {1.f, 0.f, 0.f})) / P.width + 0.5f) * kPanelW;
  py = (0.5f - vdot(d, qrot(q, {0.f, 1.f, 0.f})) / height) * kPanelH;
  return true;
}

// What's at a point on the panel (a target), or -1.
int panel_target(float px, float py) {
  if (py >= kResetY0 && py < kResetY1) {
    if (px >= kResetX0 && px < kResetX1)
      return kReset;
    const std::vector<int> pages = shown_pages();
    for (size_t i = 0; i < pages.size(); ++i) {
      const float x0 = static_cast<float>(kTabX0 + static_cast<int>(i) * (kTabW + kTabGap));
      if (px >= x0 && px < x0 + kTabW)
        return kTab + pages[i];
    }
    return -1;
  }
  if (P.page == kPageGame) {
    int count;
    {
      std::lock_guard lock{g_devOptions.mutex};
      count = static_cast<int>(g_devOptions.options.size());
    }
    for (int row = 0; row < count; ++row) {
      const float y = static_cast<float>(kAudioRowY0 + row * kAudioRowStep);
      if (px >= kTrackX0 - 24 && px < kChoiceX1 && py >= y - 18 && py < y + kTrackH + 18)
        return kOption + row;
    }
    return -1;
  }
  if (P.page == kPageTint) {
    if (!P.tintUsable)
      return -1;
    for (int i = 0; i < kSliders; ++i) {
      const float y = static_cast<float>(kRowY0 + i * kRowStep);
      if (px >= kTrackX0 - 24 && px < kTrackX1 + 24 && py >= y - 22 && py < y + kTrackH + 22)
        return i;
    }
    return -1;
  }
  if (P.page == kPageMix) {
    int count = 0;
    {
      std::lock_guard lock{g_devMix.mutex};
      const auto it = g_devMix.stages.find(g_devMix.stage);
      count = it != g_devMix.stages.end() ? it->second.count : 0;
    }
    if (count == 0)
      return -1;
    if (py >= kMixChanY0 && py < kMixChanY1 && px >= 18 && px < kChoiceX1)
      return kMixChan + std::clamp(static_cast<int>((px - 18) / ((kChoiceX1 - 18) / count)), 0, count - 1);
    if (px >= kTrackX0 - 24 && px < kTrackX1 + 24 && py >= kMixLevelY - 14 && py < kMixLevelY + kTrackH + 14)
      return kMixLevel;
    if (px >= kGraphX0 - 14 && px < kGraphX1 + 14 && py >= kGraphY0 - 14 && py < kGraphY1 + 14)
      return kMixGraph;
    if (px >= kTrackX0 - 24 && px < kTrackX1 + 24 && py >= kMixQY - 18 && py < kMixQY + kTrackH + 18)
      return kMixQ;
    return -1;
  }
  for (int row = 0; row < kAudioRows; ++row) {
    const float y = static_cast<float>(kAudioRowY0 + row * kAudioRowStep);
    if (px >= kTrackX0 - 24 && px < kChoiceX1 && py >= y - 18 && py < y + kTrackH + 18)
      return kAudio + row;
  }
  return -1;
}

bool panel_holding(const Hand& h) { return h.valid && (h.tracked ? h.held : h.sel); }
bool panel_pressed(const Hand& h) { return h.valid && (h.tracked ? h.held && !h.wasHeld : h.sel && !h.wasSel); }

// A press on an Audio choice row at px: picks the segment under it.
void audio_choose(int row, float px) {
  const int n = 2, w = (kChoiceX1 - kTrackX0) / n;
  const int i = std::clamp(static_cast<int>((px - kTrackX0) / static_cast<float>(w)), 0, n - 1);
  {
    std::lock_guard lock{g_devAudio.mutex};
    if (row == kSpatial)
      g_devAudio.value.spatial = i == 0;
  }
  P.dirty = true;
  audio_log();
}

void option_log(int option) {
  std::lock_guard lock{g_devOptions.mutex};
  const DevOption& o = g_devOptions.options[option];
  const float v = g_devOptionValue[option].load(std::memory_order_relaxed);
  if (o.slider())
    Log.info("Dev tools: {} = {:.2f}", o.name, v);
  else
    Log.info("Dev tools: {} = {}", o.name, o.choices[static_cast<int>(v)]);
}

bool option_is_slider(int option) {
  std::lock_guard lock{g_devOptions.mutex};
  return option >= 0 && option < static_cast<int>(g_devOptions.options.size()) &&
         g_devOptions.options[option].slider();
}

// Dragging a Game slider to px.
void option_drag(int option, float px) {
  std::lock_guard lock{g_devOptions.mutex};
  if (option >= static_cast<int>(g_devOptions.options.size()))
    return;
  const DevOption& o = g_devOptions.options[option];
  const float t = std::clamp((px - kTrackX0) / static_cast<float>(kTrackX1 - kTrackX0), 0.f, 1.f);
  float v = o.min + t * (o.max - o.min);
  if (o.step > 0.f)
    v = std::clamp(o.min + std::round((v - o.min) / o.step) * o.step, o.min, o.max);
  if (v != g_devOptionValue[option].load(std::memory_order_relaxed)) {
    g_devOptionValue[option].store(v, std::memory_order_relaxed);
    P.dirty = true;
  }
}

// A press on a Game option's row at px: picks the choice under it.
void option_choose(int option, float px) {
  {
    std::lock_guard lock{g_devOptions.mutex};
    if (option >= static_cast<int>(g_devOptions.options.size()))
      return;
    const int n = static_cast<int>(g_devOptions.options[option].choices.size());
    const int w = (kChoiceX1 - kTrackX0) / n;
    g_devOptionValue[option].store(
        static_cast<float>(std::clamp(static_cast<int>((px - kTrackX0) / static_cast<float>(w)), 0, n - 1)),
        std::memory_order_relaxed);
  }
  P.dirty = true;
  option_log(option);
}

// A press on the Stage page's graph at px, py: grabs the nearest band's
// handle, to move with the laser from there.
void mix_grab(float px, float py) {
  std::lock_guard lock{g_devMix.mutex};
  const auto it = g_devMix.stages.find(g_devMix.stage);
  if (it == g_devMix.stages.end() || it->second.count == 0)
    return;
  const AuroraXrMixChannel& m = it->second.value[std::clamp(P.mixChannel, 0, it->second.count - 1)];
  float best = std::numeric_limits<float>::max();
  for (int b = 0; b < AURORA_XR_MIX_BANDS; ++b) {
    const float dx = graph_x(m.band[b].freq) - px, dy = graph_y(m.band[b].gain) - py;
    if (dx * dx + dy * dy < best) {
      best = dx * dx + dy * dy;
      P.mixBand = b;
      P.grabDx = dx;
      P.grabDy = dy;
    }
  }
}

// Dragging on the Stage page (a target) to px, py.
void mix_drag(int target, float px, float py) {
  std::lock_guard lock{g_devMix.mutex};
  const auto it = g_devMix.stages.find(g_devMix.stage);
  if (it == g_devMix.stages.end() || it->second.count == 0)
    return;
  AuroraXrMixChannel& m = it->second.value[std::clamp(P.mixChannel, 0, it->second.count - 1)];
  const AuroraXrMixChannel was = m;
  const float t = std::clamp((px - kTrackX0) / static_cast<float>(kTrackX1 - kTrackX0), 0.f, 1.f);
  if (target == kMixLevel) {
    const float v = std::round((AURORA_XR_MIX_OFF + t * (kMixLevelMax - AURORA_XR_MIX_OFF)) * 2.f) / 2.f;
    m.level = v <= AURORA_XR_MIX_OFF + 0.5f ? AURORA_XR_MIX_OFF : v;
  } else if (target == kMixGraph) {
    AuroraXrMixBand& b = m.band[std::clamp(P.mixBand, 0, AURORA_XR_MIX_BANDS - 1)];
    const float f = graph_freq(px + P.grabDx);
    const float step = f < 100.f ? 1.f : f < 1000.f ? 5.f : f < 5000.f ? 50.f : 100.f;
    b.freq = std::round(f / step) * step;
    b.gain = std::clamp(std::round(graph_db(py + P.grabDy) * 2.f) / 2.f, -kGraphDb, kGraphDb);
  } else if (target == kMixQ) {
    AuroraXrMixBand& b = m.band[std::clamp(P.mixBand, 0, AURORA_XR_MIX_BANDS - 1)];
    b.q = std::round(kMixQMin * std::pow(kMixQMax / kMixQMin, t) * 100.f) / 100.f;
  }
  if (std::memcmp(&was, &m, sizeof(m)) != 0)
    P.dirty = true;
}

// Dragging a slider (a target) to px.
void panel_drag(int target, float px, float py) {
  if (target == kMixLevel || target == kMixGraph || target == kMixQ) {
    mix_drag(target, px, py);
    return;
  }
  if (target >= kOption && target < kOption + AURORA_XR_DEV_OPTIONS) {
    option_drag(target - kOption, px);
    return;
  }
  const float t = std::clamp((px - kTrackX0) / static_cast<float>(kTrackX1 - kTrackX0), 0.f, 1.f);
  if (target == kAudio + kMusic) {
    const float v = std::round(t * 20.f) / 20.f; // steps of 5%
    std::lock_guard lock{g_devAudio.mutex};
    if (v != g_devAudio.value.music) {
      g_devAudio.value.music = v;
      P.dirty = true;
    }
    return;
  }
  if (target < 0 || target >= kSliders)
    return;
  const float v = t * kSliderMax[target];
  if (v != P.value[target]) {
    P.value[target] = v;
    P.dirty = true;
    std::lock_guard lock{g_grade.mutex};
    g_grade.panel[P.stage] = panel_grade();
    set_grade_locked(g_grade.panel[P.stage]);
  }
}

void panel_reset() {
  if (P.page == kPageGame) {
    std::lock_guard lock{g_devOptions.mutex};
    for (size_t i = 0; i < g_devOptions.options.size(); ++i)
      g_devOptionValue[i].store(g_devOptions.options[i].initial, std::memory_order_relaxed);
    P.dirty = true;
    Log.info("Dev tools: the game's options back as they were");
    return;
  }
  if (P.page == kPageMix) {
    {
      std::lock_guard lock{g_devMix.mutex};
      const auto it = g_devMix.stages.find(g_devMix.stage);
      if (it == g_devMix.stages.end() || it->second.count == 0)
        return;
      const int c = std::clamp(P.mixChannel, 0, it->second.count - 1);
      it->second.value[c] = it->second.defaults[c];
    }
    P.dirty = true;
    mix_log();
    return;
  }
  if (P.page == kPageTint) {
    std::lock_guard lock{g_grade.mutex};
    g_grade.panel.erase(P.stage);
    set_grade_locked(g_grade.game);
    panel_from_grade(g_grade.game);
    Log.info("Room grade panel: stage 0x{:02X} back to the game's grade", P.stage);
    return;
  }
  {
    std::lock_guard lock{g_devAudio.mutex};
    g_devAudio.value = g_devAudio.defaults;
  }
  P.dirty = true;
  audio_log();
}

void panel_released() {
  if (P.active >= 0 && P.active < kSliders)
    panel_log("set");
  else if (P.active == kAudio + kMusic)
    audio_log();
  else if (P.active == kMixLevel || P.active == kMixGraph || P.active == kMixQ)
    mix_log();
  else if (P.active >= kOption && P.active < kOption + AURORA_XR_DEV_OPTIONS)
    option_log(P.active - kOption);
}

// The bar under the panel, in its frame (meters), as the screen's.
BarRect panel_bar() {
  const float hw = std::clamp(P.width * 0.1f, 0.05f, 0.2f);
  const float hh = std::clamp(P.width * 0.008f, 0.005f, 0.012f);
  return {-P.width * kPanelH / kPanelW * 0.5f - hh * 3.f, hw, hh};
}

// A point in the panel's frame on (or within 2 cm of) its bar.
bool on_panel_bar(float x, float y) {
  const BarRect b = panel_bar();
  return std::abs(x) <= b.hw + 0.02f && std::abs(y - b.y) <= b.hh + 0.02f;
}

// A point in the room touching the bar: on it and within 5 cm of its plane.
bool in_panel_bar_box(XrVector3f p) {
  const XrVector3f l = qrot(qconj(P.pose.orientation), p - P.pose.position);
  return on_panel_bar(l.x, l.y) && std::abs(l.z) <= 0.05f;
}

// Opening where the player hasn't put it: right of the screen, at a reading
// distance along the head's line to there (beside the screen); or right of
// where the head looks (AURORA_XR_DEV_TOOLS_POS), in 3D.
void place_dev_panel(bool besideScreen) {
  P.width = 0.36f;
  if (besideScreen) {
    const ScreenPose& s = screen_pose();
    const XrVector3f edge = s.pos + qrot(s.orientation, {s.width * 0.5f, 0.f, 0.f});
    const float d = std::min(vlen(s.pos - B.head), 0.7f);
    // Turned right past the screen's edge by half the panel and a few cm.
    const XrVector3f toEdge = vnorm(edge - B.head);
    P.pose.position = B.head + rot_y(toEdge, -std::atan((P.width * 0.5f + 0.04f) / d)) * d;
  } else {
    const XrVector3f fwd = qrot(B.headOrientation, {0.f, 0.f, -1.f});
    XrVector3f offset{0.3f, -0.15f, -0.55f};
    if (const char* v = std::getenv("AURORA_XR_DEV_TOOLS_POS"))
      std::sscanf(v, "%f,%f,%f,%f", &offset.x, &offset.y, &offset.z, &P.width);
    P.pose.position = B.head + rot_y(offset, std::atan2(-fwd.x, -fwd.z));
  }
  P.pose.orientation = facing(P.pose.position, B.head);
}

void end_panel_move() {
  if (P.moveHand < 0 && !P.sizing)
    return;
  P.moveHand = -1;
  P.sizing = false;
  P.placed = true;
  Log.info("Dev tools panel placed at {:.2f},{:.2f},{:.2f}, width {:.2f} m", P.pose.position.x, P.pose.position.y,
           P.pose.position.z, P.width);
}

void begin_panel_move(int h, float hit) {
  const Hand& hand = G.hands[h];
  P.moveHand = h;
  P.sizing = false;
  P.moveDirect = in_panel_bar_box(hand.touch);
  P.moveDist = hit >= 0.f ? hit : std::max(vdot(P.pose.position - hand.origin, hand.dir), 0.1f);
  P.moveOffset = P.pose.position - (P.moveDirect ? hand.touch : hand.origin + hand.dir * P.moveDist);
}

// Moving the panel, each display frame it shows, before its presses: takes
// hold of it, moves or resizes it, lets go. A hand holding it is on it.
void move_dev_panel() {
  const auto down = [](const Hand& h) { return h.valid && (h.held || h.sel); };
  const auto pressed = [](const Hand& h) { return h.valid && ((h.held && !h.wasHeld) || (h.sel && !h.wasSel)); };
  const auto gripPressed = [](const Hand& h) { return h.valid && !h.tracked && h.held && !h.wasHeld; };
  // What each hand's laser is on: the panel, its bar (or a touch on it).
  std::array<float, 2> hit{-1.f, -1.f};
  std::array<bool, 2> onFace{}, onBar{};
  P.barHover = false;
  for (int h = 0; h < 2; ++h) {
    const Hand& hand = G.hands[h];
    if (!hand.valid || hand.engaged)
      continue;
    float t, px, py;
    if (in_panel_bar_box(hand.touch)) {
      onBar[h] = true;
      hit[h] = std::max(vdot(P.pose.position - hand.origin, hand.dir), 0.05f);
    } else if (panel_ray(hand, t, px, py)) {
      const float x = (px / kPanelW - 0.5f) * P.width, y = (0.5f - py / kPanelH) * P.width * kPanelH / kPanelW;
      onFace[h] = px >= 0.f && px < kPanelW && py >= 0.f && py < kPanelH;
      onBar[h] = on_panel_bar(x, y);
      if (onFace[h] || onBar[h])
        hit[h] = t;
    }
    P.barHover |= onBar[h];
  }
  // Letting go; with two, the other carries on alone.
  if (P.sizing) {
    const bool l = down(G.hands[0]), r = down(G.hands[1]);
    if (l && !r)
      begin_panel_move(0, -1.f);
    else if (r && !l)
      begin_panel_move(1, -1.f);
    else if (!l && !r)
      end_panel_move();
  } else if (P.moveHand >= 0 && !down(G.hands[P.moveHand])) {
    end_panel_move();
  }
  // Taking hold: the bar with anything, the panel with a controller's grip;
  // the other hand, while one holds, pressing on either, to resize.
  for (int h = 0; h < 2; ++h) {
    if (P.activeHand == h || P.moveHand == h || P.sizing)
      continue;
    const Hand& hand = G.hands[h];
    const bool take = (onBar[h] && pressed(hand)) || (onFace[h] && gripPressed(hand));
    if (P.moveHand >= 0 && (onBar[h] || onFace[h]) && pressed(hand)) {
      P.sizing = true;
      P.moveHand = -1;
      const XrVector3f l = G.hands[0].touch, r = G.hands[1].touch;
      P.span0 = std::max(vlen(r - l), 0.05f);
      P.mid0 = (l + r) * 0.5f;
      P.width0 = P.width;
      P.pos0 = P.pose.position;
    } else if (P.moveHand < 0 && P.activeHand < 0 && take) {
      begin_panel_move(h, hit[h]);
    }
  }
  if (P.sizing) {
    const XrVector3f l = G.hands[0].touch, r = G.hands[1].touch;
    P.width = std::clamp(P.width0 * vlen(r - l) / P.span0, 0.15f, 1.5f);
    P.pose.position = P.pos0 + ((l + r) * 0.5f - P.mid0);
  } else if (P.moveHand >= 0) {
    const Hand& hand = G.hands[P.moveHand];
    if (!P.moveDirect && !hand.tracked) {
      // The stick reels it along the laser, as the screen's.
      static auto last = std::chrono::steady_clock::now();
      const auto now = std::chrono::steady_clock::now();
      const float dt = std::min(std::chrono::duration<float>(now - last).count(), 0.05f);
      last = now;
      const float y = B.stickY[P.moveHand];
      if (std::abs(y) > 0.2f)
        P.moveDist = std::clamp(P.moveDist * std::exp(y * dt * 0.7f), 0.2f, 4.f);
    }
    P.pose.position = (P.moveDirect ? hand.touch : hand.origin + hand.dir * P.moveDist) + P.moveOffset;
  }
  if (P.sizing || P.moveHand >= 0)
    P.pose.orientation = facing(P.pose.position, B.head);
  for (int h = 0; h < 2; ++h) {
    Hand& hand = G.hands[h];
    const bool holding = P.sizing || P.moveHand == h;
    if (holding || onBar[h]) {
      hand.onPanel = true;
      hand.panelHit = holding && !P.sizing && !P.moveDirect ? P.moveDist : hit[h];
    }
  }
}

// The controller reeling the panel with its stick, or -1.
int dev_panel_reel_hand() {
  return P.shown && P.moveHand >= 0 && !P.moveDirect && !G.hands[P.moveHand].tracked ? P.moveHand : -1;
}

// The panel's bar: faint, brighter while a laser or hand is on it, tinted
// while the panel is held.
void add_panel_bar(PointerLayers& out) {
  if (!B.barSwapchain || !P.shown)
    return;
  const BarRect b = panel_bar();
  const bool held = P.moveHand >= 0 || P.sizing;
  const XrColor4f color = held         ? XrColor4f{0.45f, 0.85f, 1.f, 1.f}
                          : P.barHover ? XrColor4f{1.f, 1.f, 1.f, 0.95f}
                                       : XrColor4f{0.85f, 0.88f, 0.92f, 0.55f};
  add_pointer_quad(out, B.barSwapchain, kBarW, kBarH, P.pose.position + qrot(P.pose.orientation, {0.f, b.y, 0.f}),
                   P.pose.orientation, {b.hw * 2.f, b.hh * 2.f}, color);
}

// Each display frame, before the arena's grab: shows the panel while `show`,
// and moves it, points, presses and drags on it. Marks the hands on it
// (Hand::onPanel). `tintUsable`: the room's grade is on and the room shows.
// `besideScreen`: the screen shows, so it opens beside it.
void update_dev_panel(bool show, bool tintUsable, bool besideScreen) {
  for (Hand& hand : G.hands) {
    hand.onPanel = false;
    hand.panelHit = -1.f;
  }
  if (tintUsable != P.tintUsable) {
    P.tintUsable = tintUsable;
    P.dirty = true;
  }
  if (g_devOptionsAdded.exchange(false, std::memory_order_relaxed))
    P.dirty = true;
  if (show && !P.shown) {
    if (!P.placed) {
      place_dev_panel(besideScreen);
      const ScreenPose& sp = screen_pose();
      Log.info("Dev tools panel opened at {:.2f},{:.2f},{:.2f}, width {:.2f} m{}", P.pose.position.x,
               P.pose.position.y, P.pose.position.z, P.width,
               besideScreen ? fmt::format(", beside the screen at {:.2f},{:.2f},{:.2f}, width {:.2f} m", sp.pos.x,
                                          sp.pos.y, sp.pos.z, sp.width)
                            : std::string{});
    }
    P.dirty = true;
    std::lock_guard lock{g_grade.mutex};
    P.stage = g_grade.stage;
    panel_from_grade(g_grade.grade);
  }
  if (show) {
    // Kept while it has something on it; otherwise the first that has.
    const std::vector<int> pages = shown_pages();
    if (std::find(pages.begin(), pages.end(), P.page) == pages.end()) {
      P.page = pages.front();
      P.dirty = true;
    }
  }
  if (!show && P.active >= 0)
    panel_released();
  if (!show) {
    P.active = P.activeHand = -1;
    end_panel_move();
  }
  P.shown = show;
  g_devMixListening.store(show && P.page == kPageMix, std::memory_order_relaxed);
  if (!show)
    return;
  move_dev_panel();
  int hover = -1;
  for (int h = 0; h < 2; ++h) {
    Hand& hand = G.hands[h];
    if (P.sizing || P.moveHand == h)
      continue;
    float t, px, py;
    if (!panel_ray(hand, t, px, py))
      continue;
    const bool dragging = P.activeHand == h;
    const bool on = px >= 0.f && px < kPanelW && py >= 0.f && py < kPanelH;
    // A laser on the panel is on it, though the arena is nearer along it or
    // the hand is in the arena (the stage need not be moved aside first);
    // only a hand already holding the arena keeps it.
    if (!dragging && (!hand.valid || !on || hand.engaged))
      continue;
    hand.onPanel = true;
    hand.panelHit = t;
    const int target = dragging ? P.active : panel_target(px, py);
    hover = target;
    if (dragging) {
      panel_drag(P.active, px, py);
    } else if (panel_pressed(hand) && target == kReset) {
      panel_reset();
    } else if (panel_pressed(hand) && target >= kTab && target < kTab + kPages) {
      P.page = target - kTab;
      P.dirty = true;
    } else if (panel_pressed(hand) && target == kAudio + kSpatial) {
      audio_choose(target - kAudio, px);
    } else if (panel_pressed(hand) && target >= kOption && target < kOption + AURORA_XR_DEV_OPTIONS &&
               !option_is_slider(target - kOption)) {
      option_choose(target - kOption, px);
    } else if (panel_pressed(hand) && target >= kMixChan && target < kMixChan + AURORA_XR_MIX_CHANNELS) {
      P.mixChannel = target - kMixChan;
      P.dirty = true;
    } else if (panel_pressed(hand) &&
               (target == kAudio + kMusic || target == kMixLevel || target == kMixGraph || target == kMixQ ||
                (target >= 0 && target < kSliders) ||
                (target >= kOption && target < kOption + AURORA_XR_DEV_OPTIONS)) &&
               P.active < 0) {
      if (target == kMixGraph)
        mix_grab(px, py);
      P.active = target;
      P.activeHand = h;
      P.dirty = true;
      --h; // again, dragging from where it pressed
      continue;
    }
  }
  if (P.activeHand >= 0 && !panel_holding(G.hands[P.activeHand])) {
    panel_released();
    P.active = P.activeHand = -1;
    P.dirty = true;
  }
  if (hover != P.hover) {
    P.hover = hover;
    P.dirty = true;
  }
}

// The panel's layer, drawn again (a new static swapchain) when it changed,
// at most 20 times a second. Old swapchains go once their upload is surely
// done.
bool build_dev_panel_layer(XrCompositionLayerQuad& q) {
  const auto now = std::chrono::steady_clock::now();
  while (!P.retired.empty() && now - P.retired.front().second > std::chrono::milliseconds(500)) {
    xrDestroySwapchain(P.retired.front().first);
    P.retired.pop_front();
  }
  if (!P.shown)
    return false;
  if (P.dirty && now - P.drawn >= std::chrono::milliseconds(50)) {
    Canvas c{kPanelW, kPanelH};
    draw_panel(c);
    // AURORA_XR_DUMP: the panel as last drawn, over black.
    if (!B.dumpDir.empty()) {
      if (FILE* f = std::fopen((B.dumpDir + "/xr_panel.ppm").c_str(), "wb")) {
        std::fprintf(f, "P6\n%d %d\n255\n", kPanelW, kPanelH);
        for (size_t i = 0; i < c.px.size(); i += 4)
          for (int ch = 0; ch < 3; ++ch)
            std::fputc(c.px[i + ch] * c.px[i + 3] / 255, f);
        std::fclose(f);
      }
    }
    if (P.image.swapchain)
      P.retired.emplace_back(P.image.swapchain, now);
    P.image.swapchain = XR_NULL_HANDLE;
    P.image.rgba = std::move(c.px);
    P.image.width = kPanelW;
    P.image.height = kPanelH;
    if (!upload_placing_image(P.image)) {
      static bool warned = false;
      if (!warned)
        Log.warn("Dev tools panel: picture not uploaded");
      warned = true;
    }
    P.dirty = false;
    P.drawn = now;
  }
  if (!P.image.swapchain)
    return false;
  q = {XR_TYPE_COMPOSITION_LAYER_QUAD};
  q.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
  q.space = B.space;
  q.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
  q.subImage.swapchain = P.image.swapchain;
  q.subImage.imageRect = {{0, 0}, {kPanelW, kPanelH}};
  q.pose = P.pose;
  q.size = {P.width, P.width * kPanelH / kPanelW};
  return true;
}

void destroy_dev_panel() {
  for (auto& r : P.retired)
    xrDestroySwapchain(r.first);
  P.retired.clear();
  if (P.image.swapchain)
    xrDestroySwapchain(P.image.swapchain);
  P.image.swapchain = XR_NULL_HANDLE;
  P.shown = false;
  P.active = P.activeHand = -1;
}

// Full VR stops passthrough outright (the cameras stop, not just the layer),
// mixed reality starts it again.
void sync_passthrough() {
  const bool want = g_passthroughWanted;
  if (!B.passthrough || want == B.passthroughRunning)
    return;
  const auto fn = want ? B.startPassthrough : B.pausePassthrough;
  if (!fn)
    return;
  XrResult r;
  {
    std::lock_guard lock{g_passthroughMutex};
    r = fn(B.passthrough);
  }
  if (XR_FAILED(r)) {
    Log.warn("Passthrough {} failed: XrResult {}", want ? "start" : "pause", static_cast<int>(r));
    return;
  }
  B.passthroughRunning = want;
  Log.info("Passthrough {}", want ? "on" : "off");
}

// The Guardian boundary is hidden while passthrough shows the room (the
// runtime allows nothing else) and comes back in full VR.
// XR_META_boundary_visibility; AURORA_XR_BOUNDARY=1 keeps it.
void sync_boundary() {
#ifdef XR_META_boundary_visibility
  if (!B.requestBoundaryVisibility || !B.running)
    return;
  const bool want = B.passthroughRunning && g_passthroughWanted;
  // Refused (no passthrough layer shown yet): try again about once a second.
  static uint64_t retryAt = 0;
  if (want == B.boundarySuppressed || B.framesShown < retryAt)
    return;
  B.boundarySuppressed = want;
  const XrResult r = B.requestBoundaryVisibility(
      B.session, want ? XR_BOUNDARY_VISIBILITY_SUPPRESSED_META : XR_BOUNDARY_VISIBILITY_NOT_SUPPRESSED_META);
  if (r == XR_SUCCESS)
    Log.info("Boundary {} requested", want ? "suppression" : "restore");
  if (r == XR_BOUNDARY_VISIBILITY_SUPPRESSION_NOT_ALLOWED_META || XR_FAILED(r)) {
    static int warnings = 0;
    if (warnings++ < 3)
      Log.warn("Boundary {} refused: XrResult {}", want ? "suppression" : "restore", static_cast<int>(r));
    B.boundarySuppressed = !want;
    retryAt = B.framesShown + 72;
  }
#endif
}

void poll_tweaks() {
  static bool init = false;
  static const char* path = std::getenv("AURORA_XR_TWEAK");
  static int calls = 0;
  if (!init) {
    init = true;
    g_tweak.lead = env_flag("AURORA_XR_POSE_LEAD", true);
    g_tweak.set = env_flag("AURORA_XR_SET_HANDOVER", true);
    g_tweak.cutout = env_flag("AURORA_XR_APPSW_CUTOUT", true);
    g_tweak.trans = env_flag("AURORA_XR_APPSW_TRANSLUCENT", true);
    g_tweak.epoch = env_flag("AURORA_XR_APPSW_EPOCH", true);
    g_tweak.gap = env_flag("AURORA_XR_APPSW_GAP", true);
    g_tweak.late = env_flag("AURORA_XR_APPSW_LATE", false);
    g_tweak.eyesort = env_flag("AURORA_XR_EYE_SORT", true);
    gfx::set_xr_eye_sort(g_tweak.eyesort);
    g_tweak.bindless = env_flag("AURORA_XR_BINDLESS", true);
    gfx::set_xr_motion_classes((g_tweak.cutout ? 1u : 0u) | (g_tweak.trans ? 2u : 0u));
  }
  if (path == nullptr || ++calls % 30 != 0)
    return;
  FILE* f = std::fopen(path, "r");
  if (f == nullptr)
    return;
  char line[128] = {0};
  const bool got = std::fgets(line, sizeof line, f) != nullptr;
  std::fclose(f);
  if (!got)
    return;
  Tweaks t = g_tweak;
  for (const char* q = line; *q != '\0';) {
    int v = 0;
    if (std::sscanf(q, "lead=%d", &v) == 1)
      t.lead = v != 0;
    else if (std::sscanf(q, "set=%d", &v) == 1)
      t.set = v != 0;
    else if (std::sscanf(q, "cutout=%d", &v) == 1)
      t.cutout = v != 0;
    else if (std::sscanf(q, "trans=%d", &v) == 1)
      t.trans = v != 0;
    else if (std::sscanf(q, "epoch=%d", &v) == 1)
      t.epoch = v != 0;
    else if (std::sscanf(q, "gap=%d", &v) == 1)
      t.gap = v != 0;
    else if (std::sscanf(q, "late=%d", &v) == 1)
      t.late = v != 0;
    else if (std::sscanf(q, "eyesort=%d", &v) == 1)
      t.eyesort = v != 0;
    else if (std::sscanf(q, "bindless=%d", &v) == 1)
      t.bindless = v != 0;
    while (*q != '\0' && *q != ' ')
      ++q;
    while (*q == ' ')
      ++q;
  }
  if (t.lead != g_tweak.lead || t.set != g_tweak.set || t.cutout != g_tweak.cutout || t.trans != g_tweak.trans ||
      t.epoch != g_tweak.epoch || t.gap != g_tweak.gap || t.late != g_tweak.late || t.eyesort != g_tweak.eyesort ||
      t.bindless != g_tweak.bindless)
    Log.info("XR tweak: lead {}, set {}, cutout {}, trans {}, epoch {}, gap {}, late {}, eyesort {}, bindless {}",
             t.lead, t.set, t.cutout, t.trans, t.epoch, t.gap, t.late, t.eyesort, t.bindless && g_bindlessOk);
  g_tweak = t;
  gfx::set_xr_motion_classes((t.cutout ? 1u : 0u) | (t.trans ? 2u : 0u));
  gfx::set_xr_eye_sort(t.eyesort);
  gfx::set_xr_bindless(t.bindless && g_bindlessOk);
}

void log_slots(const char* why);

// ---------------------------------------------------------------- XR thread: placing the board on a surface
//
// A game whose stage is a board on a real table or the floor
// (aurora_xr_set_board_proxy) has it placed in three steps, once at launch
// (AURORA_XR_PLACE_AT_START), before anything else shows:
//   where   a laser on a level surface (the floor, a table) stands the
//           board there, broadside to the head; the trigger or a pinch
//           puts it down
//   size    the board grows or shrinks so that its edge follows the laser
//           on that surface
//   height  the board rises off the surface on a stand, its top where the
//           laser meets the upright plane through it
// In every step the stick turns the board, and B, Y or the grip goes back a
// step (B on the first skips placing). The board shown is a proxy: quad
// layers (its top with the pitch, the sides, the stand's), drawn by the
// compositor, so it shows over the menus before any 3D frame exists. The
// placement becomes every stage's start (as aurora_xr_adopt_stage), and the
// game draws the stand from aurora_xr_stand_height.
//
// Surfaces come from XR_META_environment_raycast (the surface worker casts
// each hand's laser), then from planes: the floor (XR_EXT_local_floor) and,
// on the desktop, tables given as AURORA_XR_PLACE_TABLE=x,y,z,halfX,halfZ
// (LOCAL meters; several separated by ';'). AURORA_XR_PLACE_AUTO=k,height[,d]
// places unattended (a pointer aimed by the code: the first table, or the
// floor d meters ahead (1.4), then that size and stand height) and, with
// AURORA_XR_DUMP, writes each step as seen from the eye, turned to look at
// the board (xr_place1-3).

constexpr size_t kMaxBoardLayers = 14;

struct BoardLayers {
  std::array<XrCompositionLayerQuad, kMaxBoardLayers> quads;
  std::array<XrCompositionLayerColorScaleBiasKHR, kMaxBoardLayers> tints;
  std::array<const PlacingImage*, kMaxBoardLayers> images{}; // for the dump; null: a laser
  uint32_t count = 0;
};

struct BoardPlacer {
  enum class Step { Off, Where, Size, Height, Done };
  Step step = Step::Off;
  AuroraXrBoardProxy proxy{};
  bool haveProxy = false;
  int hand = 1; // the laser that places: the last whose trigger or pinch went
  // The placement so far (LOCAL meters): the spot on the surface, the
  // surface's height, the board's turn, its size against the proxy's, and
  // the stand's height (the surface to the board's foot).
  XrVector3f spot{};
  float surfaceY = 0.f, yaw = 0.f, turn = 0.f, k = 1.f, height = 0.f;
  bool aimed = false; // this frame the active laser is on a level surface (where) or its plane
  XrVector3f aim{};
  std::array<float, 2> planeY{};     // each laser's surface, smoothed
  std::array<bool, 2> planeKnown{};
  bool backWas = false;
  int64_t stepNs = 0;
  // The card: where it floats, and what it says.
  XrPosef card{{0.f, 0.f, 0.f, 1.f}, {}};
  float cardYaw = 0.f;
  bool cardPlaced = false;
  int cardDrawn = -1; // the step and input it was drawn for
  PlacingImage cardImg, topImg;
  std::array<PlacingImage, 4> sideImg, standImg;
  bool imagesMade = false;
  struct Table {
    XrVector3f c;
    float hx, hz;
  };
  std::vector<Table> tables;
  // Unattended (AURORA_XR_PLACE_AUTO).
  bool autoOn = false;
  float autoK = 0.8f, autoHeight = 0.15f, autoDist = 1.4f;
  std::array<bool, 4> dumped{};
};
BoardPlacer BP;
std::mutex g_boardProxyMutex; // BP.proxy and haveProxy, set by the game

// The surface the placement stands on, for the stand's height (game threads).
bool g_boardPlaced = false; // under g_arenaMutex
float g_boardSurfaceY = 0.f;

bool board_place_wanted() {
  static const bool on = env_flag("AURORA_XR_PLACE_AT_START", false);
  return on;
}

XrVector3f board_axis_x(float yaw) { return rot_y({1.f, 0.f, 0.f}, yaw); }
XrVector3f board_axis_z(float yaw) { return rot_y({0.f, 0.f, 1.f}, yaw); }

// The laser of hand h on the horizontal plane y: false if it points up or away.
bool ray_plane_y(const Hand& hand, float y, XrVector3f& out) {
  if (hand.dir.y > -0.02f)
    return false;
  const float t = (y - hand.origin.y) / hand.dir.y;
  if (t <= 0.f || t > 12.f)
    return false;
  out = hand.origin + hand.dir * t;
  return true;
}

// Without the raycaster: the nearest of the tables and the floor.
bool plane_hit(const Hand& hand, XrTime time, XrVector3f& out, float& y) {
  float best = 1e9f;
  for (const auto& t : BP.tables) {
    XrVector3f p;
    if (ray_plane_y(hand, t.c.y, p) && std::fabs(p.x - t.c.x) <= t.hx && std::fabs(p.z - t.c.z) <= t.hz) {
      const float d = vlen(p - hand.origin);
      if (d < best) {
        best = d;
        out = p;
        y = t.c.y;
      }
    }
  }
  float floorY;
  if (best > 1e8f && locate_floor(time, floorY)) {
    XrVector3f p;
    if (ray_plane_y(hand, floorY, p)) {
      out = p;
      y = floorY;
      best = 0.f;
    }
  }
  return best < 1e8f;
}

// Where hand h's laser meets a level surface, and that surface's height.
bool surface_hit(int h, XrTime time, XrVector3f& out, float& y) {
  const Hand& hand = G.hands[h];
  if (!hand.valid)
    return false;
  SurfaceShared::PointHit hit;
  bool ready, gone;
  {
    std::lock_guard lock{g_surface.mutex};
    hit = g_surface.pointHits[h];
    ready = g_surface.casterReady;
    gone = g_surface.casterGone || !g_surfaceThread.joinable();
  }
  if (!ready) {
    // Until the raycaster is up (and for good without one), the planes.
    if (!gone && BP.tables.empty() && !B.floorSpace)
      return false;
    return plane_hit(hand, time, out, y);
  }
  if (hit.ns == 0 || monotonic_ns() - hit.ns > 300 * kMsNs || !hit.hit || !hit.level) {
    BP.planeKnown[h] = false;
    return false;
  }
  // The latest cast says which surface; the laser meets it now (the casts
  // lag the hand), at a height smoothed against the depth's jitter.
  if (!BP.planeKnown[h] || std::fabs(hit.p.y - BP.planeY[h]) > 0.03f)
    BP.planeY[h] = hit.p.y;
  else
    BP.planeY[h] += (hit.p.y - BP.planeY[h]) * 0.2f;
  BP.planeKnown[h] = true;
  y = BP.planeY[h];
  return ray_plane_y(hand, y, out) && vlen(out - hit.p) < 0.5f;
}

// The proxy's pictures: the top (the pitch on the board), the sides lit four
// ways (+x, +z, -x, -z) and the stand's.
void make_board_images() {
  const AuroraXrBoardProxy& p = BP.proxy;
  const int tw = 512, th = std::clamp(static_cast<int>(std::lround(512.f * p.footHalfZ / p.footHalfX)), 64, 1024);
  Canvas c{tw, th};
  const auto px = [&](float x) { return static_cast<int>(std::lround((x / p.footHalfX * 0.5f + 0.5f) * tw)); };
  const auto pz = [&](float z) { return static_cast<int>(std::lround((z / p.footHalfZ * 0.5f + 0.5f) * th)); };
  c.fill(0, 0, tw, th, {40, 43, 50, 255});                                           // the flared foot
  c.fill(px(-p.topHalfX), pz(-p.topHalfZ), px(p.topHalfX), pz(p.topHalfZ), {104, 108, 116, 255}); // bevel
  c.fill(px(-p.topHalfX) + 3, pz(-p.topHalfZ) + 3, px(p.topHalfX) - 3, pz(p.topHalfZ) - 3, {46, 49, 56, 255});
  const int x0 = px(-p.pitchHalfX), x1 = px(p.pitchHalfX), z0 = pz(-p.pitchHalfZ), z1 = pz(p.pitchHalfZ);
  const int stripes = 10;
  for (int i = 0; i < stripes; ++i)
    c.fill(x0 + (x1 - x0) * i / stripes, z0, x0 + (x1 - x0) * (i + 1) / stripes, z1,
           i % 2 ? Rgba{58, 132, 62, 255} : Rgba{50, 118, 55, 255});
  const Rgba white{235, 240, 235, 255};
  const int lw = std::max(2, tw / 200);
  c.fill(x0, z0, x1, z0 + lw, white);
  c.fill(x0, z1 - lw, x1, z1, white);
  c.fill(x0, z0, x0 + lw, z1, white);
  c.fill(x1 - lw, z0, x1, z1, white);
  const int mx = (x0 + x1) / 2, mz = (z0 + z1) / 2;
  c.fill(mx - lw / 2, z0, mx - lw / 2 + lw, z1, white);
  const float r = (z1 - z0) * 0.17f;
  for (int y = static_cast<int>(mz - r - lw); y <= mz + r + lw; ++y)
    for (int x = static_cast<int>(mx - r - lw); x <= mx + r + lw; ++x) {
      const float d = std::hypot(static_cast<float>(x - mx), static_cast<float>(y - mz));
      if (d >= r - lw * 0.5f && d <= r + lw * 0.5f)
        c.fill(x, y, x + 1, y + 1, white);
    }
  // The goal boxes, and the goals behind the lines.
  const int bx = (x1 - x0) / 9, bz = (z1 - z0) * 3 / 10, gz = (z1 - z0) / 7, gd = std::max(4, (x1 - x0) / 60);
  for (int end = 0; end < 2; ++end) {
    const int xa = end ? x1 - bx : x0, xb = end ? x1 : x0 + bx;
    c.fill(xa, mz - bz, xb, mz - bz + lw, white);
    c.fill(xa, mz + bz - lw, xb, mz + bz, white);
    c.fill(end ? xa : xb - lw, mz - bz, end ? xa + lw : xb, mz + bz, white);
    c.fill(end ? x1 : x0 - gd, mz - gz, end ? x1 + gd : x0, mz + gz, {210, 210, 220, 255});
  }
  BP.topImg.rgba = std::move(c.px);
  BP.topImg.width = static_cast<uint32_t>(tw);
  BP.topImg.height = static_cast<uint32_t>(th);
  // Sides: a bright bevel along the top, the side darkening downwards, a
  // dark foot band; each side its own light so the block reads as solid.
  constexpr float kLight[4] = {0.95f, 0.8f, 0.7f, 1.05f};
  for (int i = 0; i < 4; ++i) {
    for (int stand = 0; stand < 2; ++stand) {
      Canvas s{8, 64};
      for (int y = 0; y < 64; ++y) {
        const float t = y / 63.f;
        float v;
        if (stand)
          v = (34.f + 14.f * (1.f - t)) * kLight[i];
        else if (t < 0.18f)
          v = 108.f * kLight[i] * (1.f - 0.15f * t / 0.18f);
        else if (t < 0.82f)
          v = 58.f * kLight[i] * (1.1f - 0.4f * (t - 0.18f) / 0.64f);
        else
          v = 20.f * kLight[i];
        const uint8_t g = static_cast<uint8_t>(std::clamp(v, 0.f, 255.f));
        s.fill(0, y, 8, y + 1, {g, static_cast<uint8_t>(std::min(255.f, g * 1.06f)),
                                static_cast<uint8_t>(std::min(255.f, g * 1.18f)), 255});
      }
      PlacingImage& img = stand ? BP.standImg[i] : BP.sideImg[i];
      img.rgba = std::move(s.px);
      img.width = 8;
      img.height = 64;
    }
  }
  BP.imagesMade = true;
}

constexpr int kCardW = 640, kCardH = 200;

void draw_board_card() {
  const bool hands = G.hands[BP.hand].tracked;
  const int key = static_cast<int>(BP.step) * 2 + (hands ? 1 : 0);
  if (key == BP.cardDrawn && BP.cardImg.swapchain)
    return;
  Canvas c{kCardW, kCardH};
  c.fill(0, 0, kCardW, kCardH, {18, 20, 26, 225});
  c.fill(0, 0, kCardW, 4, {120, 200, 255, 255});
  const char* press = hands ? "PINCH" : "TRIGGER";
  std::string title, line1, line2, line3;
  switch (BP.step) {
  case BoardPlacer::Step::Where:
    title = "STEP 1 OF 3: WHERE";
    line1 = "POINT AT THE FLOOR OR A TABLE";
    line2 = std::string(press) + ": PUT IT HERE";
    line3 = hands ? "" : "STICK: TURN    B: SKIP";
    break;
  case BoardPlacer::Step::Size:
    title = "STEP 2 OF 3: SIZE";
    line1 = "POINT WHERE ITS EDGE GOES";
    line2 = std::string(press) + ": THIS SIZE";
    line3 = hands ? "" : "STICK: TURN    B OR GRIP: BACK";
    break;
  default:
    title = "STEP 3 OF 3: HEIGHT";
    line1 = "POINT WHERE ITS TOP GOES";
    line2 = std::string(press) + ": DONE";
    line3 = hands ? "" : "STICK: TURN    B OR GRIP: BACK";
    break;
  }
  c.text(24, 22, 5, title, {255, 255, 255, 255});
  c.text(24, 82, 3, line1, {220, 226, 235, 255});
  c.text(24, 118, 3, line2, {220, 226, 235, 255});
  c.text(24, 154, 3, line3, {160, 168, 180, 255});
  if (BP.cardImg.swapchain)
    xrDestroySwapchain(BP.cardImg.swapchain);
  BP.cardImg.swapchain = XR_NULL_HANDLE;
  BP.cardImg.rgba = std::move(c.px);
  BP.cardImg.width = kCardW;
  BP.cardImg.height = kCardH;
  upload_placing_image(BP.cardImg);
  BP.cardDrawn = key;
}

void board_go(BoardPlacer::Step step) {
  static const char* const kNames[] = {"off", "where", "size", "height", "done"};
  BP.step = step;
  BP.stepNs = monotonic_ns();
  Log.info("Board placement: {}", kNames[static_cast<int>(step)]);
}

// Every stage starts where the board was put (as aurora_xr_adopt_stage).
void board_adopt() {
  std::lock_guard lock{g_arenaMutex};
  init_arena();
  g_adoptedArena.pos = BP.spot + XrVector3f{0.f, BP.height, 0.f};
  g_adoptedArena.yaw = BP.yaw;
  g_adoptedArena.scale = g_defaultArena.scale * BP.k;
  g_adoptedSet = true;
  g_stageArenas.clear();
  g_arenaToHead = false;
  g_boardPlaced = true;
  g_boardSurfaceY = BP.surfaceY;
  if (g_stage >= 0) {
    g_arena = g_adoptedArena;
    g_arena.scale *= g_stageScale;
  }
  Log.info("Board placed: at {:.2f},{:.2f},{:.2f} on a surface at y {:.3f}, yaw {:.0f} deg, {:.2f} m long, on a "
           "{:.2f} m stand",
           BP.spot.x, BP.spot.y, BP.spot.z, BP.surfaceY, BP.yaw * 57.2958f, 2.f * BP.proxy.pitchHalfX * BP.k,
           BP.height);
}

// The broadside turn: the board's length across the line from the head.
float board_facing(XrVector3f spot) {
  const XrVector3f f = spot - B.head;
  return std::atan2(-f.x, -f.z) + BP.turn;
}

// Unattended: a right hand a little under and ahead of the head, aimed at
// this step's target, pressing its trigger 1.2 s into the step.
void board_auto_hand(XrTime time) {
  Hand& hand = G.hands[1];
  const XrVector3f fwd = qrot(B.headOrientation, {0.f, 0.f, -1.f});
  const float headYaw = std::atan2(-fwd.x, -fwd.z);
  hand.valid = true;
  hand.tracked = false;
  hand.origin = B.head + rot_y({0.15f, -0.3f, -0.25f}, headYaw);
  XrVector3f target;
  if (BP.step == BoardPlacer::Step::Where) {
    float floorY = B.head.y - 1.6f;
    locate_floor(time, floorY);
    target = BP.tables.empty() ? B.head + rot_y({0.f, 0.f, -BP.autoDist}, headYaw) : BP.tables[0].c;
    if (BP.tables.empty())
      target.y = floorY;
  } else if (BP.step == BoardPlacer::Step::Size) {
    target = BP.spot + board_axis_x(BP.yaw) * (BP.proxy.footHalfX * BP.autoK);
  } else {
    target = BP.spot + XrVector3f{0.f, BP.autoHeight + BP.proxy.depth * BP.k, 0.f};
  }
  hand.dir = vnorm(target - hand.origin);
  // One frame's press per step.
  static int64_t pressedIn = 0;
  hand.wasSel = false;
  hand.sel = monotonic_ns() - BP.stepNs > 1200 * kMsNs && pressedIn != BP.stepNs;
  if (hand.sel)
    pressedIn = BP.stepNs;
}

void dump_board_view(int step, const BoardLayers& layers, XrTime time);

// XR thread, every frame: runs the placement while it's on; true while it
// shows (nothing else does then).
bool board_place_frame(XrTime time, bool viewsValid) {
  {
    std::lock_guard lock{g_boardProxyMutex};
    if (BP.step == BoardPlacer::Step::Off) {
      if (!board_place_wanted() || !BP.haveProxy || !B.focused || !viewsValid)
        return false;
      static bool once = false;
      if (std::exchange(once, true))
        return false;
      if (const char* v = std::getenv("AURORA_XR_PLACE_TABLE")) {
        for (std::string_view rest{v}; !rest.empty();) {
          const size_t semi = rest.find(';');
          const std::string one{rest.substr(0, semi)};
          BoardPlacer::Table t{};
          if (std::sscanf(one.c_str(), "%f,%f,%f,%f,%f", &t.c.x, &t.c.y, &t.c.z, &t.hx, &t.hz) == 5)
            BP.tables.push_back(t);
          rest = semi == std::string_view::npos ? std::string_view{} : rest.substr(semi + 1);
        }
      }
      if (const char* v = std::getenv("AURORA_XR_PLACE_AUTO")) {
        BP.autoOn = true;
        std::sscanf(v, "%f,%f,%f", &BP.autoK, &BP.autoHeight, &BP.autoDist);
      }
      board_go(BoardPlacer::Step::Where);
    }
  }
  if (BP.step == BoardPlacer::Step::Done)
    return false;
  if (!BP.imagesMade)
    make_board_images();
  g_boardPlacing = true;

  static auto last = std::chrono::steady_clock::now();
  const auto now = std::chrono::steady_clock::now();
  const float dt = std::min(std::chrono::duration<float>(now - last).count(), 0.1f);
  last = now;

  if (BP.autoOn && B.focused)
    board_auto_hand(time);
  // The laser that places: the one whose trigger or pinch went last.
  int pressed = -1;
  for (int h = 0; h < 2; ++h) {
    const Hand& hand = G.hands[h];
    if (hand.valid && (hand.tracked ? hand.held && !hand.wasHeld : hand.sel && !hand.wasSel))
      pressed = h;
  }
  if (pressed >= 0)
    BP.hand = pressed;
  else if (!G.hands[BP.hand].valid && G.hands[1 - BP.hand].valid)
    BP.hand = 1 - BP.hand;
  const Hand& hand = G.hands[BP.hand];
  const bool gripBack = !hand.tracked && hand.held && !hand.wasHeld;
  const bool back = (B.backButton && !BP.backWas) || gripBack;
  BP.backWas = B.backButton;
  BP.turn -= B.stickX[BP.hand] * 1.5f * dt; // radians a second at full tilt

  // Each laser to the worker, for the room's surfaces.
  {
    std::lock_guard lock{g_surface.mutex};
    const bool pointing = BP.step == BoardPlacer::Step::Where;
    if (pointing && !g_surface.pointing)
      g_surface.cv.notify_one(); // it may be asleep for a second
    g_surface.pointing = pointing;
    g_surface.displayTime = time;
    for (int h = 0; h < 2; ++h) {
      g_surface.rayOrigin[h] = G.hands[h].origin;
      g_surface.rayDir[h] = G.hands[h].dir;
      g_surface.rayValid[h] = G.hands[h].valid;
    }
  }

  BP.aimed = false;
  switch (BP.step) {
  case BoardPlacer::Step::Where: {
    float y;
    if (surface_hit(BP.hand, time, BP.aim, y)) {
      BP.aimed = true;
      BP.spot = BP.aim;
      BP.surfaceY = y;
      BP.yaw = board_facing(BP.spot);
    }
    if (back && !gripBack) {
      Log.info("Board placement skipped: stages start in front of the head");
      board_go(BoardPlacer::Step::Done);
    } else if (pressed >= 0 && BP.aimed) {
      board_go(BoardPlacer::Step::Size);
    }
    break;
  }
  case BoardPlacer::Step::Size: {
    BP.yaw = board_facing(BP.spot);
    if (ray_plane_y(hand, BP.surfaceY, BP.aim)) {
      BP.aimed = true;
      const XrVector3f d = BP.aim - BP.spot;
      const float lx = std::fabs(vdot(d, board_axis_x(BP.yaw))), lz = std::fabs(vdot(d, board_axis_z(BP.yaw)));
      BP.k = std::clamp(std::max(lx / BP.proxy.footHalfX, lz / BP.proxy.footHalfZ), 0.25f, 4.f);
    }
    if (back)
      board_go(BoardPlacer::Step::Where);
    else if (pressed >= 0)
      board_go(BoardPlacer::Step::Height);
    break;
  }
  case BoardPlacer::Step::Height: {
    BP.yaw = board_facing(BP.spot);
    XrVector3f n = B.head - BP.spot;
    n.y = 0.f;
    n = vnorm(n);
    const float den = vdot(hand.dir, n);
    if (hand.valid && std::fabs(den) > 0.05f) {
      const float t = vdot(BP.spot - hand.origin, n) / den;
      if (t > 0.f) {
        BP.aimed = true;
        BP.aim = hand.origin + hand.dir * t;
        BP.height = std::clamp(BP.aim.y - BP.surfaceY - BP.proxy.depth * BP.k, 0.f, 2.f);
        if (BP.height < 0.01f)
          BP.height = 0.f; // resting on the surface
      }
    }
    if (back) {
      BP.height = 0.f;
      board_go(BoardPlacer::Step::Size);
    } else if (pressed >= 0) {
      board_adopt();
      board_go(BoardPlacer::Step::Done);
    }
    break;
  }
  default:
    break;
  }
  if (BP.step == BoardPlacer::Step::Done) {
    g_boardPlacing = false;
    std::lock_guard lock{g_surface.mutex};
    g_surface.pointing = false;
    return false;
  }

  // The card: ahead of the head and a little over it, facing it; it follows
  // once the head has turned well away.
  const XrVector3f fwd = qrot(B.headOrientation, {0.f, 0.f, -1.f});
  const float headYaw = std::atan2(-fwd.x, -fwd.z);
  float off = std::remainder(headYaw - BP.cardYaw, 6.2831853f);
  if (!BP.cardPlaced || std::fabs(off) > 0.6f) {
    BP.cardYaw = headYaw;
    BP.cardPlaced = true;
    const XrVector3f pos = B.head + rot_y({0.f, 0.22f, -1.1f}, headYaw);
    const XrVector3f z = vnorm(B.head - pos);
    const XrVector3f x = vnorm(vcross({0.f, 1.f, 0.f}, z));
    BP.card = {quat_from_axes(x, vcross(z, x), z), pos};
  }
  draw_board_card();
  return true;
}

void add_board_quad(BoardLayers& out, const PlacingImage& img, XrVector3f center, XrVector3f x, XrVector3f y,
                    XrVector3f z, XrExtent2Df size) {
  if (out.count >= kMaxBoardLayers || !img.swapchain)
    return;
  auto& q = out.quads[out.count];
  q = {XR_TYPE_COMPOSITION_LAYER_QUAD};
  q.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
  q.space = B.space;
  q.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
  q.subImage.swapchain = img.swapchain;
  q.subImage.imageRect = {{0, 0}, {static_cast<int32_t>(img.width), static_cast<int32_t>(img.height)}};
  q.pose = {quat_from_axes(x, y, z), center};
  q.size = size;
  out.images[out.count] = &img;
  ++out.count;
}

// A box's sides that face the head (those facing away are hidden by it):
// half extents hx, hz about `foot` (the middle of its underside), h tall.
void add_board_box(BoardLayers& out, const std::array<PlacingImage, 4>& sides, XrVector3f foot, float yaw, float hx,
                   float hz, float h) {
  const XrVector3f up{0.f, 1.f, 0.f};
  const XrVector3f ax = board_axis_x(yaw), az = board_axis_z(yaw);
  const std::array<XrVector3f, 4> normals{ax, az, ax * -1.f, az * -1.f};
  const std::array<float, 4> reach{hx, hz, hx, hz}, across{hz, hx, hz, hx};
  for (int i = 0; i < 4; ++i) {
    const XrVector3f c = foot + normals[i] * reach[i] + up * (h * 0.5f);
    if (vdot(B.head - c, normals[i]) <= 0.f)
      continue;
    add_board_quad(out, sides[i], c, vcross(up, normals[i]), up, normals[i], {across[i] * 2.f, h});
  }
}

void build_board_layers(BoardLayers& out) {
  for (PlacingImage* img : {&BP.topImg, &BP.sideImg[0], &BP.sideImg[1], &BP.sideImg[2], &BP.sideImg[3],
                            &BP.standImg[0], &BP.standImg[1], &BP.standImg[2], &BP.standImg[3]})
    if (!img->swapchain && !img->rgba.empty())
      upload_placing_image(*img);
  const AuroraXrBoardProxy& p = BP.proxy;
  const bool shown = BP.step != BoardPlacer::Step::Where || BP.aimed;
  if (shown) {
    const float k = BP.k, depth = p.depth * k;
    const XrVector3f foot = BP.spot + XrVector3f{0.f, BP.height, 0.f};
    // From under the board (a tall stand, a low head) the stand is in front
    // of its underside: drawn after it.
    const bool under = B.head.y < foot.y;
    const auto stand = [&] {
      if (BP.height > 0.f)
        add_board_box(out, BP.standImg, BP.spot, BP.yaw, p.standHalfX * k, p.standHalfZ * k, BP.height);
    };
    if (!under)
      stand();
    add_board_box(out, BP.sideImg, foot, BP.yaw, p.footHalfX * k, p.footHalfZ * k, depth);
    const XrVector3f top = foot + XrVector3f{0.f, depth, 0.f};
    if (B.head.y > top.y) {
      const XrVector3f up{0.f, 1.f, 0.f}, ax = board_axis_x(BP.yaw);
      add_board_quad(out, BP.topImg, top, ax, vcross(up, ax), up, {p.footHalfX * k * 2.f, p.footHalfZ * k * 2.f});
    }
    if (under)
      stand();
  }
  if (BP.cardImg.swapchain) {
    const float w = 0.5f;
    add_board_quad(out, BP.cardImg, BP.card.position, qrot(BP.card.orientation, {1.f, 0.f, 0.f}),
                   qrot(BP.card.orientation, {0.f, 1.f, 0.f}), qrot(BP.card.orientation, {0.f, 0.f, 1.f}),
                   {w, w * kCardH / kCardW});
  }
  // The lasers: the placing one to where it meets the surface (or 2 m, faint,
  // while it's on none), the other faint and short.
  if (!B.beamSwapchain || !B.dotSwapchain)
    return;
  for (int h = 0; h < 2; ++h) {
    const Hand& hand = G.hands[h];
    if (!hand.valid || out.count + 2 > kMaxBoardLayers)
      continue;
    const bool placing = h == BP.hand;
    const bool on = placing && BP.aimed;
    const float length = on ? vlen(BP.aim - hand.origin) : placing ? 2.f : 0.5f;
    const XrColor4f color = on ? XrColor4f{0.45f, 0.85f, 1.f, 1.f} : XrColor4f{0.85f, 0.9f, 1.f, placing ? 0.5f : 0.3f};
    const XrVector3f center = hand.origin + hand.dir * (length * 0.5f);
    const XrVector3f toHead = B.head - center;
    const XrVector3f z = vnorm(toHead - hand.dir * vdot(toHead, hand.dir));
    const auto tinted = [&](XrSwapchain sc, uint32_t w, uint32_t hgt, XrVector3f c, XrQuaternionf q, XrExtent2Df size) {
      auto& l = out.quads[out.count];
      l = {XR_TYPE_COMPOSITION_LAYER_QUAD};
      l.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
      l.space = B.space;
      l.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
      l.subImage.swapchain = sc;
      l.subImage.imageRect = {{0, 0}, {static_cast<int32_t>(w), static_cast<int32_t>(hgt)}};
      l.pose = {q, c};
      l.size = size;
      if (B.hasColorScaleBias) {
        auto& t = out.tints[out.count];
        t = {XR_TYPE_COMPOSITION_LAYER_COLOR_SCALE_BIAS_KHR};
        t.colorScale = {color.r * color.a, color.g * color.a, color.b * color.a, color.a};
        l.next = &t;
      }
      out.images[out.count] = nullptr;
      ++out.count;
    };
    if (vlen(z) > 0.5f)
      tinted(B.beamSwapchain, kBeamW, kBeamH, center, quat_from_axes(vcross(hand.dir, z), hand.dir, z),
             {0.004f, length});
    if (on) {
      const XrVector3f dz = vnorm(B.head - BP.aim);
      const XrVector3f dx = vnorm(vcross({0.f, 1.f, 0.f}, dz));
      if (vlen(dx) > 0.5f)
        tinted(B.dotSwapchain, kDotSize, kDotSize, BP.aim, quat_from_axes(dx, vcross(dz, dx), dz), {0.025f, 0.025f});
    }
  }
}

// AURORA_XR_DUMP with AURORA_XR_PLACE_AUTO: what the left eye sees of the
// placement's layers (lasers left out, the aim a red cross), over the
// planes the desktop stands in for the room: a checked floor, brown tables.
void dump_board_view(int step, const BoardLayers& layers, XrTime time) {
  if (B.dumpDir.empty() || BP.dumped[step])
    return;
  XrView view;
  {
    std::lock_guard lock{g_viewMutex};
    if (!g_viewsValid)
      return;
    view = g_latestViews[0];
  }
  BP.dumped[step] = true;
  // From the eye, turned to look at the board (the eye looks level, the
  // board is well under it), 60 degrees across.
  const XrVector3f look = BP.spot + XrVector3f{0.f, BP.height + BP.proxy.depth * BP.k * 0.5f, 0.f};
  {
    const XrVector3f z = vnorm(view.pose.position - look);
    const XrVector3f x = vnorm(vcross({0.f, 1.f, 0.f}, z));
    view.pose.orientation = quat_from_axes(x, vcross(z, x), z);
    view.fov = {-0.5236f, 0.5236f, 0.5236f, -0.5236f};
  }
  constexpr int W = 640, H = 640;
  std::vector<uint8_t> out(W * H * 3);
  float floorY = -100.f;
  locate_floor(time, floorY);
  const XrVector3f o = view.pose.position;
  const float tl = std::tan(view.fov.angleLeft), tr = std::tan(view.fov.angleRight);
  const float tu = std::tan(view.fov.angleUp), td = std::tan(view.fov.angleDown);
  for (int py = 0; py < H; ++py)
    for (int px = 0; px < W; ++px) {
      const float u = (px + 0.5f) / W, v = (py + 0.5f) / H;
      const XrVector3f d = vnorm(qrot(view.pose.orientation, {tl + (tr - tl) * u, tu + (td - tu) * v, -1.f}));
      float col[3] = {70.f, 74.f, 80.f};
      // The room: the nearest of the floor and the tables.
      float best = 1e9f;
      if (d.y < 0.f) {
        const float t = (floorY - o.y) / d.y;
        if (t > 0.f) {
          const XrVector3f p = o + d * t;
          best = t;
          const bool chk = (static_cast<int>(std::floor(p.x * 2.f)) + static_cast<int>(std::floor(p.z * 2.f))) & 1;
          col[0] = col[1] = col[2] = chk ? 120.f : 100.f;
        }
        for (const auto& tb : BP.tables) {
          const float t2 = (tb.c.y - o.y) / d.y;
          const XrVector3f p = o + d * t2;
          if (t2 > 0.f && t2 < best && std::fabs(p.x - tb.c.x) <= tb.hx && std::fabs(p.z - tb.c.z) <= tb.hz) {
            best = t2;
            col[0] = 120.f, col[1] = 84.f, col[2] = 52.f;
          }
        }
      }
      // The layers in order, each over what's under it.
      for (uint32_t i = 0; i < layers.count; ++i) {
        const PlacingImage* img = layers.images[i];
        if (!img)
          continue;
        const auto& q = layers.quads[i];
        const XrVector3f n = qrot(q.pose.orientation, {0.f, 0.f, 1.f});
        const float den = vdot(d, n);
        if (std::fabs(den) < 1e-6f)
          continue;
        const float t = vdot(q.pose.position - o, n) / den;
        if (t <= 0.f)
          continue;
        const XrVector3f rel = o + d * t - q.pose.position;
        const float lx = vdot(rel, qrot(q.pose.orientation, {1.f, 0.f, 0.f})) / q.size.width + 0.5f;
        const float ly = 0.5f - vdot(rel, qrot(q.pose.orientation, {0.f, 1.f, 0.f})) / q.size.height;
        if (lx < 0.f || lx >= 1.f || ly < 0.f || ly >= 1.f)
          continue;
        const size_t at = (static_cast<size_t>(ly * img->height) * img->width + static_cast<size_t>(lx * img->width)) * 4;
        const float a = img->rgba[at + 3] / 255.f;
        for (int ch = 0; ch < 3; ++ch)
          col[ch] = col[ch] * (1.f - a) + img->rgba[at + ch] * a;
      }
      for (int ch = 0; ch < 3; ++ch)
        out[(py * W + px) * 3 + ch] = static_cast<uint8_t>(std::clamp(col[ch], 0.f, 255.f));
    }
  // The aim: a red cross where it projects.
  if (BP.aimed) {
    const XrVector3f rel = BP.aim - o;
    const XrQuaternionf inv = qconj(view.pose.orientation);
    const XrVector3f e = qrot(inv, rel);
    if (e.z < -0.01f) {
      const float u = (e.x / -e.z - tl) / (tr - tl), v = (e.y / -e.z - tu) / (td - tu);
      const int cx = static_cast<int>(u * W), cy = static_cast<int>(v * H);
      for (int i = -8; i <= 8; ++i)
        for (int w = -1; w <= 1; ++w)
          for (const auto [x, y] : {std::pair{cx + i, cy + w}, std::pair{cx + w, cy + i}})
            if (x >= 0 && x < W && y >= 0 && y < H) {
              uint8_t* p = &out[(y * W + x) * 3];
              p[0] = 255, p[1] = 40, p[2] = 40;
            }
    }
  }
  const std::string path = B.dumpDir + "/xr_place" + std::to_string(step) + ".ppm";
  if (FILE* f = std::fopen(path.c_str(), "wb")) {
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    std::fwrite(out.data(), 1, out.size(), f);
    std::fclose(f);
    Log.info("Dumped place{} ({})", step, path);
  }
}

void destroy_board_placer() {
  for (PlacingImage* img : {&BP.cardImg, &BP.topImg, &BP.sideImg[0], &BP.sideImg[1], &BP.sideImg[2], &BP.sideImg[3],
                            &BP.standImg[0], &BP.standImg[1], &BP.standImg[2], &BP.standImg[3]}) {
    if (img->swapchain)
      xrDestroySwapchain(img->swapchain);
    img->swapchain = XR_NULL_HANDLE;
  }
  BP.cardDrawn = -1;
}

bool render_xr_frame() {
  XrFrameState fs{XR_TYPE_FRAME_STATE};
  {
    TraceSection trace{"xrWaitFrame"};
    XR_TRY(xrWaitFrame(B.session, nullptr, &fs));
  }
  // Application SpaceWarp halves the loop's rate while it's active (frames
  // with SpaceWarp info): pace the game on the loop, not the display.
  if (B.appSw && fs.predictedDisplayPeriod > 0) {
    const int64_t period = fs.predictedDisplayPeriod;
    const int64_t current = g_loopPeriodNs > 0 ? g_loopPeriodNs
                                               : static_cast<int64_t>(1e9 / std::max(g_displayHz.load(), 1.f));
    if (std::llabs(period - current) * 10 > current) {
      g_loopPeriodNs = period;
      update_pacing();
      // Latency counts loop frames: keep its time (rounded up, within the
      // cap) and start its samples and bans over in the new unit.
      auto& p = g_present;
      if (p.forced < 0 && p.latency > 0)
        p.latency = std::min(static_cast<int>((p.latency * current + period - 1) / period), latency_cap());
      p.count = p.next = 0;
      p.sinceUpdate = p.steadyLower = 0;
      p.lateLatency = -1;
      p.bannedUntilNs = {};
      p.bans = {};
      Log.info("XR loop period {:.2f} ms: SpaceWarp {}", period / 1e6,
               g_displayHz.load() < 100.f ? "active" : "idle");
    }
  }
  B.loopWaitNs = monotonic_ns();
  if (B.loopLastWaitNs != 0) {
    const int64_t gap = B.loopWaitNs - B.loopLastWaitNs;
    B.gapMaxNs = std::max(B.gapMaxNs, gap);
    B.gapOver10 += gap > 10000000 ? 1 : 0;
  }
  B.loopLastWaitNs = B.loopWaitNs;
  std::array<int64_t, 5> pre{B.loopWaitNs};
  free_finished_uploads();
  pre[1] = monotonic_ns();
  if (const int per = g_displayPerGameFrame; per > 0 && ++B.displayFrame % static_cast<uint64_t>(per) == 0) {
    {
      std::lock_guard lock{g_paceMutex};
      ++g_paceTick;
      g_lastTickFrame = B.displayFrame;
#ifdef __ANDROID__
      trace_counter("XR tick", static_cast<int64_t>(B.displayFrame & 1)); // lines traces up with ticks
#endif
    }
    g_paceCv.notify_all();
  }
  if (g_displayPerGameFrame > 0)
    pace_display(B.displayFrame, fs.predictedDisplayTime);
  {
    std::vector<uint8_t> done;
    std::vector<std::pair<uint8_t, bool>> lateness;
    {
      std::lock_guard lock{g_pace.mutex};
      done.swap(g_pace.doneLatency);
      lateness.swap(g_pace.lateness);
    }
    for (uint8_t lat : done)
      note_ready_latency(lat);
    for (const auto& [lat, late] : lateness)
      note_lateness(lat, late);
  }
  pre[2] = monotonic_ns();
  update_input();
  pre[3] = monotonic_ns();
  publish_views(fs.predictedDisplayTime);
  pre[4] = monotonic_ns();
  static const bool atHead = env_flag("AURORA_XR_ARENA_AT_HEAD", false);
  bool viewsValid;
  {
    std::lock_guard lock{g_viewMutex};
    viewsValid = g_viewsValid;
  }
  // Unattended runs wait for a tracked head: placed from the identity pose
  // before it, the stage stood wherever the lenses weren't looking.
  if (B.focused && (!atHead || viewsValid) && g_arenaToHead.exchange(false)) {
    const XrVector3f fwd = qrot(B.headOrientation, {0.f, 0.f, -1.f});
    const float yaw = std::atan2(-fwd.x, -fwd.z); // rot_y's sense
    std::lock_guard lock{g_arenaMutex};
    g_arena.yaw = g_defaultArena.yaw + yaw;
    // From the head's spot on the floor, at the default height; unattended
    // runs (AURORA_XR_ARENA_AT_HEAD) from the head itself, wherever it lies
    // and looks, up or down too (a headset on a desk may face anywhere).
    g_arena.pos = atHead ? B.head + qrot(B.headOrientation, g_defaultArena.pos)
                         : XrVector3f{B.head.x, 0.f, B.head.z} + rot_y(g_defaultArena.pos, yaw);
    Log.info("Arena placed at the head: {:.2f},{:.2f},{:.2f}, yaw {:.0f} deg", g_arena.pos.x, g_arena.pos.y,
             g_arena.pos.z, g_arena.yaw * 57.2958f);
  }
  int64_t mark = monotonic_ns();
  auto section = [&](int i) {
    const int64_t t = monotonic_ns();
    B.sectionNs[i] = t - mark;
    mark = t;
  };
  B.sectionNs = {};
  for (int i = 0; i < 4; ++i)
    B.sectionNs[9 + i] = pre[i + 1] - pre[i];
  {
    TraceSection trace{"xrBeginFrame"};
    XR_TRY(xrBeginFrame(B.session, nullptr));
  }
  section(0);

  const auto now = std::chrono::steady_clock::now();
  poll_tweaks();
  if (!compute_gate())
    return false;
  // Hand finished images to the runtime, then line up the next ones for the
  // render worker.
  for (int i = 0; i < kStreamCount; ++i) {
    bool released = false;
    if (!release_ready(g_streams[i], released))
      return false;
    if (i == kStereo && appsw_on()) {
      if (released) {
        ++g_missWhy[4];
      } else {
        std::lock_guard lock{g_mutex};
        int why = 0;
        for (const Slot& sl : g_streams[i].ring) {
          if (sl.state == SlotState::Rendering)
            why = std::max(why, 1);
          else if (sl.state == SlotState::Ready)
            why = std::max(why, held_for_slot(g_streams[i], const_cast<Slot&>(sl)) ? 2 : 3);
        }
        ++g_missWhy[why];
      }
    }
    section(1 + i * 2);
    if (!acquire_ahead(g_streams[i]))
      return false;
    section(2 + i * 2);
    B.releasedSinceStats[i] += released ? 1 : 0;
  }
  // A fight is on while 3D frames keep coming, or until the game draws a
  // flat frame instead: a hitch in a fight (the Underground Maze stalled
  // 0.6 s loading mid-fight) holds the last 3D frame rather than flashing the
  // virtual screen, which during fights holds a stale black frame.
  const auto& stereo = g_streams[kStereo];
  const auto& hud = g_streams[kHud];
  const bool flatSince = g_streams[kScreen].haveImage && g_streams[kScreen].lastRelease > stereo.lastRelease;
  const bool fight =
      stereo.haveImage && (now - stereo.lastRelease < std::chrono::milliseconds(250) || !flatSince);
  {
    // Logged: a fight dropping to the virtual screen (no 3D image for a
    // quarter second) and coming back, with how long the screen showed.
    static bool wasFight = false;
    static std::chrono::steady_clock::time_point screenSince{};
    if (wasFight && !fight) {
      screenSince = now;
      Log.info("3D frames stopped: the virtual screen shows");
    } else if (!wasFight && fight && screenSince.time_since_epoch().count() != 0) {
      Log.info("3D frames back after {} ms on the virtual screen",
               std::chrono::duration_cast<std::chrono::milliseconds>(now - screenSince).count());
    }
    wasFight = fight;
  }
  // Paused, or held before it starts for placing the stage.
  const int placing = g_placing.load();
  const bool held = g_fightPaused || placing > 0;
  {
    // Each 3D image is due for g_displayPerGameFrame display frames under
    // lock-step pacing; one shown longer is late. Counted once per image.
    static uint64_t lastReleases = 0;
    static int shownFor = 0;
    const int due = g_displayPerGameFrame.load();
    if (stereo.releases != lastReleases) {
      if (fight && !held && shownFor > 0)
        ++B.shownHistogram[std::min(shownFor, 4) - 1];
      lastReleases = stereo.releases;
      shownFor = 0;
    }
    if (fight && due > 0 && !held && ++shownFor == due + 1)
      ++g_stereoLate;
  }
  // Held fights place the arena; menus place the screen.
  const bool screenShown = !fight && g_streams[kScreen].haveImage;
  {
    std::lock_guard lock{g_viewMutex};
    g_audioScreen.shown = screenShown;
    if (screenShown) {
      const ScreenPose& s = screen_pose();
      g_audioScreen.pose = {s.orientation, s.pos};
      g_audioScreen.width = s.width;
    }
  }
  const bool pointing = fight && held && B.focused;
  const bool screenPointing = screenShown && B.focused;
  locate_hands(fs.predictedDisplayTime);
  // Placing the board on a surface (at launch): it alone shows.
  const bool boardPlacing = board_place_frame(fs.predictedDisplayTime, viewsValid);
  // The dev tools panel, whatever shows, but for the hold before a fight.
  const bool roomWanted = B.passthroughLayer && B.passthroughRunning && g_passthroughWanted;
  update_dev_panel(dev_tools_wanted() && placing == 0 && !boardPlacing && B.focused && (fight || screenShown),
                   roomWanted && B.hasColorLutExt, screenShown);
  update_grab(fight ? Target::Arena : Target::Screen, (pointing || screenPointing) && !boardPlacing);
  if (const int h = dev_panel_reel_hand(); h >= 0)
    B.reelHand = h;
  bool onPanel = false;
  for (Hand& hand : G.hands) {
    if (hand.onPanel) {
      hand.hit = hand.panelHit; // the laser ends on the panel
      onPanel = true;
    }
  }
  // On the screen the grips stay Z unless a laser is on it.
  const bool onScreen = screenPointing && std::any_of(G.hands.begin(), G.hands.end(), [](const Hand& h) {
                          return h.engaged || h.clicking || h.inside || h.hit >= 0.f;
                        });
  B.pointerMode = pointing || onScreen || onPanel || boardPlacing;
  // The triggers click the screen or the panel instead of pressing L and R.
  B.triggersClick = onScreen || onPanel || boardPlacing;
  // The surface under the stage is measured while it's being placed: the
  // hold before a fight, or a grab in a pause.
  surface_frame(fight, fight && (placing > 0 || (G.target == Target::Arena && (G.oneHand >= 0 || G.twoHands))),
                fs.predictedDisplayTime);
  room_survey_apply();
  spot_frame(fs.predictedDisplayTime);

  // The passthrough, the shadow, the HUD and the 3D view, the HUD on top in
  // full VR, then the placing cards and the lasers.
  std::array<const XrCompositionLayerBaseHeader*, 5 + std::max(kMaxPlacingLayers + kMaxPointerLayers, kMaxBoardLayers)>
      layers{};
  uint32_t layerCount = 0;
  XrCompositionLayerPassthroughFB ptLayer{XR_TYPE_COMPOSITION_LAYER_PASSTHROUGH_FB};
  sync_passthrough();
  sync_boundary();
  const bool roomShown = B.passthroughLayer && B.passthroughRunning && g_passthroughWanted;
  if (roomShown) {
    ptLayer.flags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
    ptLayer.layerHandle = B.passthroughLayer;
    layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&ptLayer);
  }
  XrCompositionLayerQuad shadowQuad{XR_TYPE_COMPOSITION_LAYER_QUAD};
  XrCompositionLayerColorScaleBiasKHR shadowTint{XR_TYPE_COMPOSITION_LAYER_COLOR_SCALE_BIAS_KHR};
  const bool shadow = build_shadow_layer(shadowQuad, shadowTint, fs.shouldRender && fight && roomShown);
  // The room's grade, while a fight that isn't held for placing shows over it.
  grade_frame(fight && placing == 0 && roomShown);
  int shadowIndex = -1;
  std::array<XrCompositionLayerProjectionView, 2> projViews{};
  std::array<XrCompositionLayerSpaceWarpInfoFB, 2> spaceWarp{};
  XrCompositionLayerProjection proj{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
  XrCompositionLayerQuad hudQuad{XR_TYPE_COMPOSITION_LAYER_QUAD};
  XrCompositionLayerQuad screenQuad{XR_TYPE_COMPOSITION_LAYER_QUAD};
  PointerLayers pointers;
  PlacingLayers placingLayers;
  XrCompositionLayerQuad panelQuad{XR_TYPE_COMPOSITION_LAYER_QUAD};
  const bool panel = build_dev_panel_layer(panelQuad);
  BoardLayers boardLayers;
  if (fs.shouldRender && boardPlacing) {
    build_board_layers(boardLayers);
    for (uint32_t i = 0; i < boardLayers.count; ++i)
      layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&boardLayers.quads[i]);
    // Unattended: each step as it's about to be pressed.
    if (BP.autoOn && monotonic_ns() - BP.stepNs > 1100 * kMsNs)
      dump_board_view(static_cast<int>(BP.step), boardLayers, fs.predictedDisplayTime);
  } else if (fs.shouldRender && fight) {
    // Premultiplied alpha (no UNPREMULTIPLIED bit): the arena's coverage
    // hides the room, effects outside it add light over passthrough.
    // Multiview: one layer per eye. Otherwise the eyes sit side by side.
    const int32_t eyeW = static_cast<int32_t>(stereo.layers > 1 ? stereo.width : stereo.width / 2);
    for (int i = 0; i < 2; ++i) {
      projViews[i] = {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
      projViews[i].pose = stereo.shownViews[i].pose;
      projViews[i].fov = stereo.shownViews[i].fov;
      projViews[i].subImage.swapchain = stereo.swapchain;
      const int32_t x = stereo.layers > 1 ? 0 : i * eyeW;
      // Dynamic resolution: only the rendered corner, which the compositor
      // stretches over the eye's field of view.
      const bool part = stereo.shownRect[0] > 0 && stereo.shownRect[1] > 0;
      projViews[i].subImage.imageRect = {
          {x, 0}, {part ? stereo.shownRect[0] : eyeW, part ? stereo.shownRect[1] : static_cast<int32_t>(stereo.height)}};
      projViews[i].subImage.imageArrayIndex = stereo.layers > 1 ? static_cast<uint32_t>(i) : 0;
      // Application SpaceWarp: this image's motion vectors and depth (handed
      // over with it). How much of the head's motion the vectors carry:
      // render_eyes (AURORA_XR_APPSW_HEAD); the app's space never moves, so
      // its delta is identity. Depth is standard: 0 at the 0.05 m near
      // plane, 1 infinitely far.
      const auto& mvs = g_streams[kMotion];
      const auto& mds = g_streams[kMotionDepth];
      if (i == 0 && appsw_on()) {
        g_tickMismatch[0] += mvs.shownTick != stereo.shownTick ? 1 : 0;
        g_tickMismatch[1] += mds.shownTick != stereo.shownTick ? 1 : 0;
        g_tickMismatch[2] += hud.haveImage && hud.shownTick != stereo.shownTick ? 1 : 0;
      }
      // An image submitted again (no new one since) goes with zero vectors,
      // so the compositor reprojects it for the head instead of
      // extrapolating the same motion a second time. Not without SpaceWarp
      // info: a frame without it turns SpaceWarp off for the runtime, which
      // re-paces the loop to the display's rate and back on the next frame;
      // each late frame flipped it twice and the flips made more late
      // frames (138 flips in 40 s of four-CPU Brinstar, images held 4+
      // display frames). AURORA_XR_APPSW_REPEAT=1: the image's own vectors
      // on every submit, as before 096d7da.
      static const bool swOnRepeat = env_flag("AURORA_XR_APPSW_REPEAT", false);
      static uint64_t lastSwTick = 0;
      const bool repeat = stereo.shownTick != 0 && stereo.shownTick == lastSwTick;
      if (i == 1)
        lastSwTick = stereo.shownTick;
      const bool zeroVectors = repeat && !swOnRepeat && B.zeroMotionSwapchain != XR_NULL_HANDLE;
      if (appsw_on() && mvs.haveImage && mds.haveImage && (swOnRepeat || !repeat || zeroVectors)) {
        auto& sw = spaceWarp[i];
        sw = {XR_TYPE_COMPOSITION_LAYER_SPACE_WARP_INFO_FB};
        sw.motionVectorSubImage.swapchain = zeroVectors ? B.zeroMotionSwapchain : mvs.swapchain;
        sw.motionVectorSubImage.imageRect = {{0, 0}, {static_cast<int32_t>(mvs.width), static_cast<int32_t>(mvs.height)}};
        sw.motionVectorSubImage.imageArrayIndex = static_cast<uint32_t>(i);
        sw.appSpaceDeltaPose = {{0.f, 0.f, 0.f, 1.f}, {0.f, 0.f, 0.f}};
        sw.depthSubImage.swapchain = mds.swapchain;
        sw.depthSubImage.imageRect = {{0, 0}, {static_cast<int32_t>(mds.width), static_cast<int32_t>(mds.height)}};
        sw.depthSubImage.imageArrayIndex = static_cast<uint32_t>(i);
        sw.minDepth = 0.f;
        sw.maxDepth = 1.f;
        // Standard depth, as Meta's XrSpaceWarp sample has it: 0 at the
        // near plane, 1 infinitely far (flip_motion_depth). Our reversed
        // depth, told as nearZ = inf and farZ = 0.05 (which the spec allows),
        // doubled the image when the head moved sideways.
        sw.nearZ = 0.05f;
        sw.farZ = std::numeric_limits<float>::infinity();
        projViews[i].next = &sw;
      }
    }
    proj.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
    proj.space = B.space;
    proj.viewCount = 2;
    proj.views = projViews.data();
    // No HUD while the stage is held for placing, cards open or not.
    const bool showHud = hud.haveImage && (now - hud.lastRelease < std::chrono::milliseconds(250) || !flatSince) &&
                         placing == 0;
    if (showHud) {
      hudQuad.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
      hudQuad.space = B.space;
      hudQuad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
      hudQuad.subImage.swapchain = hud.swapchain;
      hudQuad.subImage.imageRect = {{0, 0}, {static_cast<int32_t>(hud.width), static_cast<int32_t>(hud.height)}};
      hud_placement(hudQuad.pose, hudQuad.size);
    }
    // Mixed reality: the HUD goes under the 3D view, which is see-through
    // wherever nothing drew, so fighters passing over it hide it. Nothing
    // of the stage reaches it (it sits above the highest floor). Full VR
    // draws the whole stage and its sky, so the HUD stays on top there.
    // AURORA_XR_HUD_ON_TOP=1 keeps it on top in mixed reality too, as does
    // a stage whose scenery reaches it (aurora_xr_set_hud_on_top).
    static const bool hudOnTopEnv = env_flag("AURORA_XR_HUD_ON_TOP", false);
    const bool hudUnder = g_passthroughWanted && !hudOnTopEnv && !g_hudOnTop;
    if (shadow) {
      shadowIndex = static_cast<int>(layerCount);
      layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&shadowQuad);
    }
    if (showHud && hudUnder)
      layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&hudQuad);
    layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&proj);
    if (showHud && !hudUnder)
      layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&hudQuad);
    if (placing > 0) {
      build_placing_layers(placingLayers, placing);
      for (uint32_t i = 0; i < placingLayers.count; ++i)
        layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&placingLayers.quads[i]);
    }
    if (panel) {
      layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&panelQuad);
      add_panel_bar(pointers);
    }
    if (pointing || onPanel) {
      build_pointer_layers(pointers, false, !pointing);
      for (uint32_t i = 0; i < pointers.count; ++i)
        layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&pointers.quads[i]);
    }
    ++B.fightFrames;
  } else if (fs.shouldRender && screenShown) {
    // A swapchain with no new release shows its last released image, so the
    // screen keeps its picture on display frames the game did not produce.
    const auto& screen = g_streams[kScreen];
    const ScreenPose& pose = screen_pose();
    screenQuad.space = B.space;
    screenQuad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
    screenQuad.subImage.swapchain = screen.swapchain;
    screenQuad.subImage.imageRect = {{0, 0}, {static_cast<int32_t>(screen.width), static_cast<int32_t>(screen.height)}};
    screenQuad.pose = {pose.orientation, pose.pos};
    screenQuad.size = {pose.width, pose.width * screen_aspect()};
    layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&screenQuad);
    if (panel) {
      layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&panelQuad);
      add_panel_bar(pointers);
    }
    add_screen_bar(pointers);
    if (onScreen)
      build_pointer_layers(pointers, true);
    for (uint32_t i = 0; i < pointers.count; ++i)
      layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&pointers.quads[i]);
  }
  // Within the runtime's limit (16 at least, by the spec): the shadow goes
  // first, then the topmost layers.
  if (layerCount > B.maxLayers) {
    const uint32_t wanted = layerCount;
    if (shadowIndex >= 0) {
      std::copy(layers.begin() + shadowIndex + 1, layers.begin() + layerCount, layers.begin() + shadowIndex);
      --layerCount;
    }
    static bool warned = false;
    if (!warned) {
      Log.warn("{} composition layers wanted, {} allowed: dropped {}{}", wanted, B.maxLayers,
               shadowIndex >= 0 ? "the stage's shadow" : "the topmost layers",
               shadowIndex >= 0 && layerCount > B.maxLayers ? " and the topmost layers" : "");
      warned = true;
    }
    layerCount = std::min(layerCount, B.maxLayers);
  }
  layerCount = sanitize_layers(layers.data(), layerCount);
  XrFrameEndInfo fei{XR_TYPE_FRAME_END_INFO};
  fei.displayTime = fs.predictedDisplayTime;
  fei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
  fei.layerCount = layerCount;
  fei.layers = layers.data();
  section(7);
  {
    const int64_t now = monotonic_ns();
    B.loopMaxNs = std::max(B.loopMaxNs, now - B.loopWaitNs);
    B.loopOver4 += now - B.loopWaitNs > 4000000 ? 1 : 0;
    B.leadMinNs = std::min(B.leadMinNs, static_cast<int64_t>(fs.predictedDisplayTime) - now);
  }
  {
    TraceSection trace{"xrEndFrame"};
    // A frame the runtime refuses (a layer it finds invalid) is ended again
    // with the room alone, rather than ending the XR thread over it.
    if (const XrResult r = xrEndFrame(B.session, &fei); XR_FAILED(r)) {
      static int refused = 0;
      if (refused++ < 5)
        Log.error("xrEndFrame failed: XrResult {} ({} layers); this frame shows the room alone",
                  static_cast<int>(r), layerCount);
      fei.layerCount = roomShown ? 1 : 0;
      XR_TRY(xrEndFrame(B.session, &fei));
    }
  }
  section(8);
  if (monotonic_ns() - B.loopWaitNs > 4000000) {
    int worst = 0;
    for (int i = 0; i < 13; ++i) {
      B.sectionMaxNs[i] = std::max(B.sectionMaxNs[i], B.sectionNs[i]);
      worst = B.sectionNs[i] > B.sectionNs[worst] ? i : worst;
    }
    ++B.sectionWorst[worst];
  }
  ++B.framesShown;
#ifdef XR_META_performance_metrics
  // About twice a second: the compositor's GPU time per display frame.
  if (B.queryPerfMetric && B.compositorGpuPath != XR_NULL_PATH && B.framesShown % 36 == 0) {
    XrPerformanceMetricsCounterMETA c{XR_TYPE_PERFORMANCE_METRICS_COUNTER_META};
    if (XR_SUCCEEDED(B.queryPerfMetric(B.session, B.compositorGpuPath, &c)) &&
        (c.counterFlags & XR_PERFORMANCE_METRICS_COUNTER_FLOAT_VALUE_VALID_BIT_META) && c.floatValue > 0.f) {
      static bool logged = false;
      if (!logged) {
        Log.info("Compositor GPU time counter: {} (unit {})", c.floatValue, static_cast<int>(c.counterUnit));
        logged = true;
      }
      if (c.counterUnit == XR_PERFORMANCE_METRICS_COUNTER_UNIT_MILLISECONDS_META)
        g_compositorGpuMs = c.floatValue;
    }
  }
#endif

  const double secs = std::chrono::duration<double>(now - B.statsStart).count();
  if (secs >= 10.0) {
    static constexpr const char* kSections[] = {"begin",           "release screen", "acquire screen",
                                                "release 3D",      "acquire 3D",     "release HUD",
                                                "acquire HUD",     "layers",         "end",
                                                "uploads",         "pacing",         "input",
                                                "views"};
    std::string sections;
    for (int i = 0; i < 13; ++i)
      if (B.sectionWorst[i] > 0 || B.sectionMaxNs[i] > 2000000)
        sections += fmt::format("{}{} {:.1f}/{}", sections.empty() ? "" : ", ", kSections[i],
                                B.sectionMaxNs[i] / 1e6, B.sectionWorst[i]);
    Log.info("{:.1f} display fps ({:.0f}% 3D); frames/s released: screen {:.1f}, 3D {:.1f}, HUD {:.1f}; "
             "3D images shown 1/2/3/4+ display frames: {}/{}/{}/{}; presented {} display frames after the tick "
             "({} held, raised {} times for late frames); XR loop: wait to end {:.1f} ms at most, {} over 4 ms; gaps {:.1f} ms at most, {} over 10 ms; "
             "{:.1f} ms least before display; slow loops' sections (most ms, times largest): {}; "
             "loops by 3D hand-over (none in flight/drawing/held/no image/released) {}/{}/{}/{}/{}; "
             "no image by stream (screen/3D/HUD/motion/depth) {}/{}/{}/{}/{}; "
             "submitted from another tick than the 3D image (motion/depth/HUD) {}/{}/{}; "
             "AppSW: {} submitted late, {} planned skips, {} motion passes cut after a gap; "
             "early eyes {} ({} redrawn, {} dropped late, {} staged retired stale, {} without an image); image waits (most ms, over 2 ms) 3D {:.1f}/{} HUD {:.1f}/{} screen {:.1f}/{}{}",
             B.framesShown / secs, 100.0 * B.fightFrames / std::max<uint64_t>(B.framesShown, 1),
             B.releasedSinceStats[kScreen] / secs, B.releasedSinceStats[kStereo] / secs,
             B.releasedSinceStats[kHud] / secs, B.shownHistogram[0], B.shownHistogram[1], B.shownHistogram[2],
             B.shownHistogram[3], g_present.latency, g_present.held, g_present.raisedForLate, B.loopMaxNs / 1e6, B.loopOver4,
             B.gapMaxNs / 1e6, B.gapOver10, B.leadMinNs / 1e6, sections, g_missWhy[0], g_missWhy[1], g_missWhy[2],
             g_missWhy[3], g_missWhy[4], g_noImageBy[0].exchange(0),
             g_noImageBy[1].exchange(0), g_noImageBy[2].exchange(0), g_noImageBy[3].exchange(0),
             g_noImageBy[4].exchange(0), g_tickMismatch[0], g_tickMismatch[1], g_tickMismatch[2],
             g_appSwLate.exchange(0), g_appSwPlannedSkip.exchange(0), g_appSwGapCut.exchange(0), g_earlyEyes.exchange(0),
             g_earlyResumed.exchange(0), g_appSwDropped.exchange(0), g_stagedStale.exchange(0), g_stagedNoImage.exchange(0),
             g_acquireWaitMaxNs[kStereo] / 1e6, g_acquireWaitsOver2[kStereo],
             g_acquireWaitMaxNs[kHud] / 1e6, g_acquireWaitsOver2[kHud], g_acquireWaitMaxNs[kScreen] / 1e6,
             g_acquireWaitsOver2[kScreen], pace_summary());
    g_acquireWaitMaxNs = {};
    g_acquireWaitsOver2 = {};
    g_missWhy = {};
    g_tickMismatch = {};
    B.sectionMaxNs = {};
    B.sectionWorst = {};
    B.loopMaxNs = B.gapMaxNs = 0;
    B.leadMinNs = INT64_MAX;
    B.loopOver4 = B.gapOver10 = 0;
    B.shownHistogram = {};
    g_present.held = 0;
    g_present.raisedForLate = 0;
    B.framesShown = 0;
    B.fightFrames = 0;
    if (B.fightFrames > 0 && B.releasedSinceStats[kStereo] < secs)
      log_slots("with the 3D stream stalled");
    B.releasedSinceStats = {};
    B.statsStart = now;
  }
  return true;
}

// Each stream's images, for a stall: state (R released, A acquired, F free,
// D drawing, Y ready), acquisition order, tick; '*' claimed unwaited.
void log_slots(const char* why) {
  static constexpr char kState[] = {'R', 'A', 'F', 'D', 'Y'};
  std::string out;
  std::lock_guard lock{g_mutex};
  for (auto& st : g_streams) {
    if (st.slots.empty())
      continue;
    out += fmt::format(" {}:", st.name);
    for (const auto& sl : worker_slots(st))
      out += fmt::format(" {}{}#{}@{}", kState[static_cast<int>(sl.state)], sl.claimed ? "*" : "", sl.acquireSeq,
                         sl.tickFrame);
    out += fmt::format(" (drawing {})", st.renderingSlot);
  }
  Log.info("Images {} (display frame {}, latency {}):{}", why, B.displayFrame, g_present.latency, out);
}

// Returns false when the thread should exit.
bool poll_events() {
  XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
  while (xrPollEvent(B.instance, &ev) == XR_SUCCESS) {
    if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
      const auto state = reinterpret_cast<const XrEventDataSessionStateChanged&>(ev).state;
      Log.info("Session state {}", static_cast<int>(state));
      log_slots("at a session change");
      if (state == XR_SESSION_STATE_READY) {
        XrSessionBeginInfo bi{XR_TYPE_SESSION_BEGIN_INFO};
        bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        XR_TRY(xrBeginSession(B.session, &bi));
        B.running = true;
        request_refresh_rate();
        request_performance_levels();
        update_pacing();
#ifdef XR_META_performance_metrics
        if (B.setPerfMetricsState && B.compositorGpuPath != XR_NULL_PATH) {
          XrPerformanceMetricsStateMETA pm{XR_TYPE_PERFORMANCE_METRICS_STATE_META};
          pm.enabled = XR_TRUE;
          B.setPerfMetricsState(B.session, &pm);
        }
#endif
        B.statsStart = std::chrono::steady_clock::now();
      } else if (state == XR_SESSION_STATE_STOPPING) {
        XR_TRY(xrEndSession(B.session));
        B.running = false;
      } else if (state == XR_SESSION_STATE_EXITING || state == XR_SESSION_STATE_LOSS_PENDING) {
        B.running = false;
        return false;
      }
      // Back on the head: the room may have changed meanwhile.
      if (state == XR_SESSION_STATE_FOCUSED && !B.focused)
        surface_request(1000, false);
      B.focused = state == XR_SESSION_STATE_FOCUSED;
      if (!B.focused) {
        std::lock_guard lock{g_padMutex};
        g_padValid = false;
      }
      g_sessionRunning = B.running;
    } else if (ev.type == XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING) {
      // A recenter moves LOCAL: whatever was measured in it is gone, and the
      // stage keeps its place in the room.
      const auto& e = reinterpret_cast<const XrEventDataReferenceSpaceChangePending&>(ev);
      Log.info("Reference space {} recentered", static_cast<int>(e.referenceSpaceType));
      spot_recentered(e);
      surface_request(1000, true);
      room_resurvey();
    } else if (B.anchors && spot_event(ev)) {
      // the stage spot's anchor
    } else if (ev.type == XR_TYPE_EVENT_DATA_DISPLAY_REFRESH_RATE_CHANGED_FB) {
      update_pacing();
    } else if (ev.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
      return false;
#ifdef XR_META_boundary_visibility
    } else if (ev.type == XR_TYPE_EVENT_DATA_BOUNDARY_VISIBILITY_CHANGED_META) {
      const auto vis = reinterpret_cast<const XrEventDataBoundaryVisibilityChangedMETA&>(ev).boundaryVisibility;
      Log.info("Boundary {}", vis == XR_BOUNDARY_VISIBILITY_SUPPRESSED_META ? "hidden" : "shown");
#endif
    }
    ev = {XR_TYPE_EVENT_DATA_BUFFER};
  }
  return true;
}

bool setup() {
  // The OpenXR instance and system already exist (prepare_device), and Dawn's
  // device was created through them.
  if (!B.prepared || !adopt_dawn_device() || !size_stereo_stream() || !create_session())
    return false;
  for (auto& st : g_streams)
    if (!create_swapchain(st) || !create_ring(st))
      return false;
  if (g_streams[kStereo].staged)
    Log.info("3D images staged: {} of our own per stream, copied into the swapchains", g_streams[kStereo].ring.size());
  if (const auto& mv = g_streams[kMotion]; B.appSw && mv.width > 0 &&
                                           !create_static_swapchain(B.zeroMotionSwapchain, mv.width, mv.height, {},
                                                                    mv.swapFormat, mv.layers))
    Log.warn("SpaceWarp: no zero motion vectors; images submitted again go without SpaceWarp info");
  if (!create_input())
    Log.warn("Controller input unavailable");
  if (!create_pointer_textures())
    Log.warn("Laser pointers unavailable");
  if (!create_hand_trackers())
    Log.info("Hand tracking unavailable");
  if (!create_shadow_texture())
    Log.warn("The stage's shadow unavailable");
  start_surface_worker();
  start_grade_worker();
  if (const char* dir = std::getenv("AURORA_XR_DUMP"); dir != nullptr && *dir != '\0')
    B.dumpDir = dir;
  // Not every exit path calls aurora::shutdown, so stop the XR thread from an
  // exit handler too. It must be registered only now: exit handlers run in
  // reverse order, and the runtime and Vulkan driver libraries loaded above
  // register their own teardown, which has to run after ours.
  std::atexit([] { shutdown(); });
  return true;
}

// Destroy everything the XR thread created, newest first. Dawn owns the
// device and instance; the OpenXR instance outlives them (release_instance).
void teardown() {
  spot_destroy();
  stop_surface_worker();
  stop_grade_worker();
  // Only our queue: Dawn may be submitting on its own, and vkDeviceWaitIdle
  // would need every queue externally synchronized.
  if (B.queue)
    vkQueueWaitIdle(B.queue);
  if (B.dev)
    free_finished_uploads(true);
  if (B.dev) {
    for (auto& st : g_streams) {
      for (int i = 0; i < static_cast<int>(st.slots.size()); ++i)
        retire_slot(st, st.slots[i], i);
      for (auto& r : st.ring) {
        if (r.inFlight)
          vkWaitForFences(B.dev, 1, &r.fence, VK_TRUE, UINT64_MAX);
        r.inFlight = false;
        for (VkSemaphore sem : r.waitSems)
          vkDestroySemaphore(B.dev, sem, nullptr);
        r.waitSems.clear();
      }
    }
  }
  for (auto& sp : B.aimSpace) {
    if (sp)
      xrDestroySpace(sp);
    sp = XR_NULL_HANDLE;
  }
  for (auto& ht : B.handTrackers) {
    if (ht && B.destroyHandTracker)
      B.destroyHandTracker(ht);
    ht = XR_NULL_HANDLE;
  }
  for (XrSwapchain* sc : {&B.beamSwapchain, &B.dotSwapchain, &B.barSwapchain, &B.backSwapchain,
                          &B.zeroMotionSwapchain, &B.shadowSwapchain}) {
    if (*sc)
      xrDestroySwapchain(*sc);
    *sc = XR_NULL_HANDLE;
  }
  destroy_dev_panel();
  destroy_board_placer();
  {
    std::lock_guard lock{g_placingMutex};
    for (auto& clip : g_placingClips) {
      if (clip.atlas.swapchain)
        xrDestroySwapchain(clip.atlas.swapchain);
      clip.atlas.swapchain = XR_NULL_HANDLE;
    }
    for (auto& img : g_placingImages) {
      if (img.swapchain)
        xrDestroySwapchain(img.swapchain);
      img.swapchain = XR_NULL_HANDLE;
    }
  }
  if (B.actionSet)
    xrDestroyActionSet(B.actionSet); // destroys its actions too
  B.actionSet = XR_NULL_HANDLE;
  {
    std::lock_guard lock{g_padMutex};
    g_padValid = false;
  }
  if (B.passthroughLayer && B.destroyPassthroughLayer)
    B.destroyPassthroughLayer(B.passthroughLayer);
  if (B.passthrough && B.destroyPassthrough)
    B.destroyPassthrough(B.passthrough);
  B.passthroughLayer = XR_NULL_HANDLE;
  B.passthrough = XR_NULL_HANDLE;
  // Dawn's textures over the swapchain images never touch them again: XR is
  // stopped (Phase::Failed or g_stop) before this runs.
  for (auto& st : g_streams) {
    if (st.swapchain)
      xrDestroySwapchain(st.swapchain);
    st.swapchain = XR_NULL_HANDLE;
  }
  if (B.floorSpace)
    xrDestroySpace(B.floorSpace);
  B.floorSpace = XR_NULL_HANDLE;
  if (B.space)
    xrDestroySpace(B.space);
  if (B.session)
    xrDestroySession(B.session);
  B.space = XR_NULL_HANDLE;
  B.session = XR_NULL_HANDLE;
  if (B.dev) {
    for (auto& st : g_streams) {
      // Dawn's textures over the ring images never touch them again either.
      for (auto& r : st.ring) {
        if (r.fence)
          vkDestroyFence(B.dev, r.fence, nullptr);
        if (r.cmd)
          vkFreeCommandBuffers(B.dev, B.pool, 1, &r.cmd);
        if (r.image)
          vkDestroyImage(B.dev, r.image, nullptr);
        if (r.memory)
          vkFreeMemory(B.dev, r.memory, nullptr);
      }
      st.ring.clear();
      st.staged = false;
      for (auto& s : st.slots) {
        if (s.fence)
          vkDestroyFence(B.dev, s.fence, nullptr);
        if (s.cmd)
          vkFreeCommandBuffers(B.dev, B.pool, 1, &s.cmd);
        s.fence = VK_NULL_HANDLE;
        s.cmd = VK_NULL_HANDLE;
      }
      if (st.dumpBuf)
        vkDestroyBuffer(B.dev, st.dumpBuf, nullptr);
      if (st.dumpMem)
        vkFreeMemory(B.dev, st.dumpMem, nullptr);
      st.dumpBuf = VK_NULL_HANDLE;
      st.dumpMem = VK_NULL_HANDLE;
    }
    if (B.pool)
      vkDestroyCommandPool(B.dev, B.pool, nullptr);
    B.pool = VK_NULL_HANDLE;
  }
}

void thread_main() {
  if (!setup()) {
    Log.error("OpenXR unavailable; presenting to the window instead");
    g_phase = Phase::Failed;
    teardown();
    return;
  }
  g_phase = Phase::SlotsReady;
  while (!g_stop) {
    if (!poll_events())
      break;
    if (!B.running) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    if (!render_xr_frame())
      break;
  }
  g_sessionRunning = false;
  g_displayPerGameFrame = 0;
  g_paceCv.notify_all();
  Log.info("XR thread exiting");
  // Stop the render worker taking images, and let a frame it is drawing into
  // one finish, before the swapchains go.
  g_phase = Phase::Failed;
  for (int i = 0; i < 50; ++i) {
    bool drawing = false;
    {
      std::lock_guard lock{g_mutex};
      for (auto& st : g_streams)
        for (const auto& s : worker_slots(st))
          drawing |= s.state == SlotState::Rendering;
    }
    if (!drawing)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  teardown();
}

// ---------------------------------------------------------------- render worker: swapchain images

// Wraps every swapchain image as Dawn shared texture memory, once, after the
// XR thread created the swapchains.
bool wrap_swapchains_into_dawn() {
  for (auto& st : g_streams) {
    const VkImageCreateInfo ici = st.staged ? ring_image_info(st) : swapchain_image_info(st);
    for (auto& s : worker_slots(st)) {
      s.stm = wgpu::SharedTextureMemory::Acquire(dawn::native::vulkan::CreateSharedTextureMemoryFromVkImage(
          webgpu::g_device.Get(), reinterpret_cast<uint64_t>(s.image), &ici, st.name));
      if (!s.stm) {
        Log.error("Wrapping a {} swapchain image for Dawn failed", st.name);
        return false;
      }
      s.texture = s.stm.CreateTexture();
      if (!s.texture) {
        Log.error("SharedTextureMemory::CreateTexture failed ({})", st.name);
        return false;
      }
    }
  }
  Log.info("Swapchain images wrapped for Dawn: screen {}x{}, 3D {}x{}, HUD {}x{} ({} images each)",
           g_streams[kScreen].width, g_streams[kScreen].height, g_streams[kStereo].width, g_streams[kStereo].height,
           g_streams[kHud].width, g_streams[kHud].height, g_streams[kScreen].slots.size());
  return true;
}

// The oldest image the XR thread has ready, so images come back in the
// order they were acquired.
int take_slot_locked(Stream& st, bool unwaited = false) {
  auto& slots = worker_slots(st);
  int best = -1;
  for (int i = 0; i < static_cast<int>(slots.size()); ++i)
    if (slots[i].state == SlotState::Free && (best < 0 || slots[i].acquireSeq < slots[best].acquireSeq))
      best = i;
  // A staged image whose copy out is done is free, though the XR thread only
  // notices on its next loop (16.7 ms apart with SpaceWarp): the render worker
  // waited ~11 ms for one nearly every frame. The XR thread still cleans up
  // after the copy (release_staged) before it reuses the fence.
  if (best < 0 && st.staged)
    for (int i = 0; i < static_cast<int>(slots.size()); ++i)
      if (slots[i].state == SlotState::Released && slots[i].inFlight &&
          vkGetFenceStatus(B.dev, slots[i].fence) == VK_SUCCESS) {
        best = i;
        break;
      }
  if (best < 0 && unwaited && !st.staged)
    for (int i = 0; i < static_cast<int>(st.slots.size()); ++i)
      if (st.slots[i].state == SlotState::Acquired && !st.slots[i].claimed)
        best = i;
  return best;
}

// Waits up to `wait` for the XR thread to line up an image: it releases the
// previous one and acquires the next on its display-frame loop, which can
// land just after the render worker asks.
// `unwaited`: may take an image acquired but not yet waited (g_deferEyes);
// its contents may be written by the GPU only after await_slot_waited.

wgpu::Texture acquire_slot(Stream& st, std::chrono::milliseconds wait = std::chrono::milliseconds(10),
                           bool unwaited = false) {
  int index;
  {
    const int64_t start = monotonic_ns();
    std::unique_lock lock{g_mutex};
    g_slotFreeCv.wait_for(lock, wait,
                          [&] { return take_slot_locked(st, unwaited) >= 0 || g_phase != Phase::Imported; });
    const int64_t waited = monotonic_ns() - start;
    if (&st == &g_streams[kStereo])
      g_pace.slotWaitNs = waited;
    // Logged with the frame stats: the longest wait for an image, per stream.
    auto& most = g_acquireWaitMaxNs[&st - g_streams.data()];
    most = std::max<int64_t>(most, waited);
    if (waited > 2000000)
      ++g_acquireWaitsOver2[&st - g_streams.data()];
    index = take_slot_locked(st, unwaited);
    if (index < 0)
      return {};
    auto& slots = worker_slots(st);
    const bool claimed = unwaited && slots[index].state == SlotState::Acquired;
    st.unwaited = claimed;
    if (claimed) {
      slots[index].claimed = true;
    } else {
      slots[index].state = SlotState::Rendering;
      if (st.staged)
        slots[index].acquireSeq = ++st.acquireSeq; // handed over in this order
    }
  }
  Slot& s = worker_slots(st)[index];
  // xrWaitSwapchainImage already returned: no fences. The contents are not
  // kept, so the image starts UNDEFINED.
  wgpu::SharedTextureMemoryVkImageLayoutBeginState bs{};
  bs.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  bs.newLayout = st.depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  wgpu::SharedTextureMemoryBeginAccessDescriptor bd{};
  bd.nextInChain = &bs;
  // Dynamic resolution renders a corner of the stereo image; marked
  // initialized, the rest doesn't force Dawn to a full render area (the
  // compositor only reads the corner, through imageRect).
  bd.initialized = &st == &g_streams[kStereo] && g_dynres.on;
  if (s.stm.BeginAccess(s.texture, &bd) != wgpu::Status::Success) {
    Log.error("BeginAccess failed ({}); XR presentation disabled", st.name);
    g_phase = Phase::Failed;
    return {};
  }
  st.renderingSlot = index;
  return s.texture;
}

// Render worker: until the XR thread has waited the image it took unwaited
// (acquire_slot), counted as this frame's wait for an image.
void await_slot_waited(Stream& st) {
  if (!st.unwaited || st.renderingSlot < 0)
    return;
  const int64_t start = monotonic_ns();
  std::unique_lock lock{g_mutex};
  g_slotFreeCv.wait(lock, [&] {
    return worker_slots(st)[st.renderingSlot].state != SlotState::Acquired || g_phase != Phase::Imported;
  });
  st.unwaited = false;
  const int64_t waited = monotonic_ns() - start;
  if (&st == &g_streams[kStereo]) {
    g_pace.slotWaitNs += waited;
    auto& most = g_acquireWaitMaxNs[kStereo];
    most = std::max<int64_t>(most, waited);
    if (waited > 2000000)
      ++g_acquireWaitsOver2[kStereo];
  }
}

void release_slot(Stream& st, const std::array<XrView, 2>* views, std::array<int32_t, 2> rect = {},
                  uint64_t tickFrame = 0) {
  if (st.renderingSlot < 0)
    return;
  Slot& s = worker_slots(st)[st.renderingSlot];
  st.renderingSlot = -1;
  wgpu::SharedTextureMemoryVkImageLayoutEndState es{};
  wgpu::SharedTextureMemoryEndAccessState state{};
  state.nextInChain = &es;
  if (s.stm.EndAccess(s.texture, &state) != wgpu::Status::Success) {
    Log.error("EndAccess failed ({}); XR presentation disabled", st.name);
    g_phase = Phase::Failed;
    return;
  }
  PendingAccess release;
  release.oldLayout = static_cast<VkImageLayout>(es.oldLayout);
  release.newLayout = static_cast<VkImageLayout>(es.newLayout);
  for (size_t i = 0; i < state.fenceCount; ++i) {
    DawnFenceExportInfo oi{};
    wgpu::SharedFenceExportInfo ei{};
    ei.nextInChain = &oi;
    state.fences[i].ExportInfo(&ei);
    if (ei.type != kDawnFenceType) {
      Log.error("Unexpected Dawn fence type {}", static_cast<int>(ei.type));
      continue;
    }
    if (oi.handle >= 0)
      release.fds.push_back(dup(oi.handle)); // the SharedFence keeps its own handle
  }
  if (&st == &g_streams[kStereo] && tickFrame != 0 && !release.fds.empty())
    pace_submit(tickFrame, dup(release.fds[0]));
  std::lock_guard lock{g_mutex};
  s.fromDawn = std::move(release);
  if (views != nullptr)
    s.views = *views;
  s.rect = rect;
  s.tickFrame = tickFrame;
  s.readySeen = 0;
  s.state = SlotState::Ready;
}

// ---------------------------------------------------------------- render worker: 3D fights
//
// Every frame with fight geometry, the world draws are replayed once per eye
// with GX bind group 3 set to
//     C = P_eye · V_eye · A · V_game⁻¹
// which takes a vertex from the game camera's space (what the shader has
// after the position matrix) to the eye's clip space: undo the game camera,
// place the arena in the room (A), then look at it from the eye. Lighting,
// skinning, texgen and projected shadows happen before this, so they stay as
// the game computed them. Matrices here are column-vector math stored
// row-major, which is what `vec4(p, 1) * m` in WGSL expects.

using Mat4 = std::array<float, 16>;

Mat4 mul(const Mat4& a, const Mat4& b) {
  Mat4 r{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 4; ++k)
        r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
  return r;
}

// Inverse of an affine 3x4 (GX Mtx layout: rows, translation in column 3).
Mat4 inverse_affine(const std::array<float, 12>& m) {
  const float a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9], i = m[10];
  const float A = e * i - f * h, Bc = -(d * i - f * g), Cc = d * h - e * g;
  const float det = a * A + b * Bc + c * Cc;
  const float s = std::abs(det) > 1e-12f ? 1.f / det : 0.f;
  const std::array<float, 9> inv{A * s,  -(b * i - c * h) * s, (b * f - c * e) * s,
                                 Bc * s, (a * i - c * g) * s,  -(a * f - c * d) * s,
                                 Cc * s, -(a * h - b * g) * s, (a * e - b * d) * s};
  const float tx = m[3], ty = m[7], tz = m[11];
  return {inv[0], inv[1], inv[2], -(inv[0] * tx + inv[1] * ty + inv[2] * tz),
          inv[3], inv[4], inv[5], -(inv[3] * tx + inv[4] * ty + inv[5] * tz),
          inv[6], inv[7], inv[8], -(inv[6] * tx + inv[7] * ty + inv[8] * tz),
          0.f,    0.f,    0.f,    1.f};
}

// World (head-space) -> eye: the inverse of the eye's pose.
Mat4 view_from_pose(const XrPosef& p) {
  const float x = p.orientation.x, y = p.orientation.y, z = p.orientation.z, w = p.orientation.w;
  // Rotation matrix of the quaternion (rows), then transpose for the inverse.
  const std::array<float, 9> r{1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
                               2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                               2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)};
  const float tx = p.position.x, ty = p.position.y, tz = p.position.z;
  return {r[0], r[3], r[6], -(r[0] * tx + r[3] * ty + r[6] * tz),
          r[1], r[4], r[7], -(r[1] * tx + r[4] * ty + r[7] * tz),
          r[2], r[5], r[8], -(r[2] * tx + r[5] * ty + r[8] * tz),
          0.f,  0.f,  0.f,  1.f};
}

// From pose a toward b by t (position lerp, orientation nlerp).
XrPosef blend_pose(const XrPosef& a, const XrPosef& b, float t) {
  XrQuaternionf qb = b.orientation;
  const float d = a.orientation.x * qb.x + a.orientation.y * qb.y + a.orientation.z * qb.z + a.orientation.w * qb.w;
  if (d < 0.f)
    qb = {-qb.x, -qb.y, -qb.z, -qb.w};
  XrQuaternionf q{a.orientation.x + (qb.x - a.orientation.x) * t, a.orientation.y + (qb.y - a.orientation.y) * t,
                  a.orientation.z + (qb.z - a.orientation.z) * t, a.orientation.w + (qb.w - a.orientation.w) * t};
  const float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (n > 0.f)
    q = {q.x / n, q.y / n, q.z / n, q.w / n};
  return {q, {a.position.x + (b.position.x - a.position.x) * t, a.position.y + (b.position.y - a.position.y) * t,
              a.position.z + (b.position.z - a.position.z) * t}};
}

// Asymmetric perspective for an OpenXR fov, reversed Z with an infinite far
// plane (depth 1 at `near`, 0 at infinity), matching aurora's GX depth.
Mat4 projection(const XrFovf& fov, float near) {
  const float l = std::tan(fov.angleLeft), r = std::tan(fov.angleRight);
  const float u = std::tan(fov.angleUp), d = std::tan(fov.angleDown);
  return {2.f / (r - l), 0.f, (r + l) / (r - l), 0.f,
          0.f, 2.f / (u - d), (u + d) / (u - d), 0.f,
          0.f, 0.f, 0.f, near,
          0.f, 0.f, -1.f, 0.f};
}

// Game units -> meters in the starting head space: T(pos) · R_y(yaw) ·
// S(scale). The scale starts at AURORA_XR_ARENA_SCALE (default 0.0035: Final
// Destination's ~170-unit stage is about 0.6 m wide); the player can move,
// turn and scale the arena during a pause.
Mat4 arena_transform() {
  ArenaPose a = arena_pose();
  std::array<float, 3> ctr;
  {
    std::lock_guard lock{g_arenaMutex};
    ctr = g_arenaCenter;
    a.scale *= g_stageZoom;
  }
  const float c = std::cos(a.yaw) * a.scale, s = std::sin(a.yaw) * a.scale;
  // T(pos) · R_y(yaw) · S(scale) · T(-center)
  const float tx = a.pos.x - (c * ctr[0] + s * ctr[2]);
  const float ty = a.pos.y - a.scale * ctr[1];
  const float tz = a.pos.z - (-s * ctr[0] + c * ctr[2]);
  return {c, 0.f, s, tx, 0.f, a.scale, 0.f, ty, -s, 0.f, c, tz, 0.f, 0.f, 0.f, 1.f};
}

constexpr char kComposeShader[] = R"(
struct Params { offset: vec2f, mode: u32, alpha: f32 };
@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: DEPTH_TYPE;
@group(0) @binding(2) var<uniform> params: Params;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Premultiplied output for the compositor. 3D (mode 0): alpha is depth
// coverage (reversed Z, cleared to 0), so geometry hides the room and
// effects that write no depth add light over it. HUD (mode 1): opaque where
// the HUD drew something bright, see-through where it left black, over an
// optional constant backdrop (AURORA_XR_HUD_BACKDROP).
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let texel = vec2i(pos.xy - params.offset);
  let c = textureLoad(color_tex, texel, 0);
  if (params.mode == 1u) {
    let ink = clamp(max(c.r, max(c.g, c.b)) * 3.0, 0.0, 1.0);
    return vec4f(c.rgb, max(params.alpha, ink));
  }
  let d = textureLoad(depth_tex, texel, 0);
  return vec4f(c.rgb, select(0.0, 1.0, d > 0.0));
}
)";

struct ReplayTarget {
  gfx::XrReplayTarget target;
  std::vector<wgpu::Texture> textures; // keeps the views' textures alive
  wgpu::TextureView sampleColor;       // single-sample scene color
  wgpu::TextureView depth;
};

// Render worker only.
// Render-side GPU timing (AURORA_XR_TIMING, Dawn timestamp queries): each
// zone is a pass's begin/end pair, averaged and logged every 10 s.
// kZone3D times both eyes on the direct path (one pass), or the left eye on
// the fallback path, where kZone3DRight is the right eye.
// kZoneFrame has only a begin, written by the frame's first flat pass; the
// whole frame runs from it to the last end written (the GPU time dynamic
// resolution steers by, compositor preemptions included).
// kZoneMotion: the SpaceWarp motion vectors (render_motion).
enum TimingZone : uint32_t {
  kZone3D,
  kZone3DRight,
  kZoneCompose3D,
  kZoneHud,
  kZoneComposeHud,
  kZoneFrame,
  kZoneMotion,
  kZoneCount
};
std::atomic<bool> g_heavyHint{false}; // aurora_xr_heavy_effect, for dynamic resolution
std::atomic<bool> g_spikeSeen{false}; // a GPU frame spike was logged (RenderDoc trigger)
constexpr std::array<const char*, kZoneCount> kZoneNames{"3D eyes", "3D right eye (fallback)", "3D compose", "HUD",
                                                         "HUD compose", "frame", "motion"};

struct GpuTiming {
  wgpu::QuerySet queries;
  wgpu::Buffer resolve;
  struct Readback {
    wgpu::Buffer buffer;
    uint32_t zones = 0; // zones written in the frame it holds
    bool busy = false;  // copied into or mapped; not yet read
  };
  std::array<Readback, 3> readbacks;
  int pending = -1;   // readback filled this frame, mapped after submit
  uint32_t zones = 0; // zones written this frame
  // Filled from map callbacks (render worker, via ProcessEvents).
  std::array<double, kZoneCount> ns{};
  std::array<uint64_t, kZoneCount> samples{};
  uint64_t frames = 0;
  double last3DNs = 0;    // the newest frame's kZone3D
  double lastFrameNs = 0; // the newest frame, first flat pass to last XR pass (dynamic resolution)
  uint64_t frameSeq = 0;  // frames measured, so the controller sees each once
  std::atomic<bool> frameBegun{false}; // this frame's first pass took the kZoneFrame begin
  uint64_t worldDraws = 0, worldFrames = 0; // draws replayed for the eyes (both views)
  std::chrono::steady_clock::time_point lastLog = std::chrono::steady_clock::now();
  std::array<wgpu::PassTimestampWrites, kZoneCount> writes{};
};

// The multiview shader's XrEye: one matrix per eye, the enabled flags (y:
// 1 - the clipped draws' opacity, as float bits), then the clip planes in
// game camera space (aurora_xr_world_clip) and their fade bands, four to a
// vec4.
static_assert(gfx::XrMaxClipPlanes == AURORA_XR_MAX_CLIPS && gfx::XrMaxClipPlanes % 4 == 0);
// Last, for Application SpaceWarp's motion vectors, both eyes' previous
// frame matrices: for a point fixed in the game world (from this frame's
// game camera space), then last frame's own (from last frame's).
constexpr uint64_t kMultiviewEyeSize = 2 * 64 + 16 + gfx::XrMaxClipPlanes * 16 + gfx::XrMaxClipPlanes * 4 + 4 * 64;

struct Renderer3D {
  uint64_t layoutKey = 0;
  uint32_t sampleCount = 0;
  std::array<ReplayTarget, 3> targets; // eye 0, eye 1, HUD
  // Per eye, per world transform index (0 = none).
  std::array<std::array<wgpu::Buffer, gfx::XrMaxTransforms>, 2> eyeUniforms;
  std::array<std::array<wgpu::BindGroup, gfx::XrMaxTransforms>, 2> eyeGroups;
  // Multiview: both eyes' matrices in one uniform per world transform.
  std::array<wgpu::Buffer, gfx::XrMaxTransforms> mvUniforms;
  std::array<wgpu::BindGroup, gfx::XrMaxTransforms> mvGroups;
  wgpu::RenderPipeline compose;
  wgpu::BindGroupLayout composeLayout;
  std::array<wgpu::Buffer, 3> params;
  std::array<wgpu::BindGroup, 3> composeGroups;
  std::array<XrView, 2> renderedViews{};
  bool renderedStereo = false;
  // A fight frame that found no free 3D image: the headset keeps the last
  // one, and the flat frame is still nobody's to see.
  bool missedStereo = false;
  bool early = false; // render_eyes runs for the early eyes (early_eyes)
  uint32_t directSamples = 0; // g_xrSamples the direct attachments were made for
  std::array<int32_t, 2> renderedRect{}; // dynamic resolution: this frame's eye size (0: all)
  uint64_t renderedTick = 0;              // the frame's pacing tick (FramePacket::xrTickFrame)
  bool failed = false;
  GpuTiming timing;
  // Direct path: both eyes straight into the shared 3D image, one pass.
  wgpu::Texture directDepth;
  std::vector<wgpu::Texture> directExtras; // non-scene attachments (normal buffer)
  std::array<wgpu::TextureView, gfx::MaxColorAttachments> directExtraViews{};
  wgpu::RenderPipeline coverClear, coverSet;
  uint64_t directKey = 0;
  bool loggedPath = false;
  // SpaceWarp: per world transform, last frame's game world -> room matrix
  // (A · T) and game camera -> room (A · T · V_game⁻¹), for the motion
  // vectors' previous positions; prevCount 0: none yet.
  // The state of a frame whose eyes went out (the poses are last frame's eye
  // poses, AURORA_XR_APPSW_HEAD), by its game tick: the newest, and the one
  // before it (a tick's eyes can be encoded twice, early and resumed, and
  // both pair with the frame before it).
  struct PrevEpoch {
    std::array<Mat4, gfx::XrMaxTransforms> worldToRoom{}, cameraToRoom{};
    size_t count = 0; // 0: none yet
    std::array<XrView, 2> views{};
    uint64_t tick = 0;
  };
  PrevEpoch epochCur, epochBefore;
  // The late-submit policy: the frame after one submitted late is skipped if
  // it is late too; one tick's decision is made once.
  bool skipNextLate = false;
  uint64_t decidedTick = 0;
  bool decidedDrop = false;
  // SpaceWarp, early eyes: the motion pass goes to the GPU before the eyes
  // wait for their image, into these, copied into the SpaceWarp images after.
  wgpu::Texture motionColor, motionDepth;
  // flip_motion_depth: the motion pass's own (reversed) depth, and the pass
  // writing the standard one.
  wgpu::Texture motionDepthFlip;
  wgpu::BindGroupLayout motionDepthFlipLayout;
  wgpu::RenderPipeline motionDepthFlipPipeline;
  std::array<wgpu::BindGroup, 2> motionDepthFlipGroups;
  bool motionCopy = false; // this frame's motion pass is in them
};
Renderer3D R;

const wgpu::PassTimestampWrites* zone_writes(TimingZone zone) {
  if (!R.timing.queries)
    return nullptr;
  R.timing.zones |= 1u << zone;
  return &R.timing.writes[zone];
}

void setup_timing() {
  auto& device = webgpu::g_device;
  if (!env_flag("AURORA_XR_TIMING", true) || !device.HasFeature(wgpu::FeatureName::TimestampQuery))
    return;
  auto& t = R.timing;
  const wgpu::QuerySetDescriptor qsd{.label = "XR timing", .type = wgpu::QueryType::Timestamp, .count = kZoneCount * 2};
  t.queries = device.CreateQuerySet(&qsd);
  const uint64_t bytes = kZoneCount * 2 * sizeof(uint64_t);
  const wgpu::BufferDescriptor rd{.label = "XR timing resolve",
                                  .usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc,
                                  .size = bytes};
  t.resolve = device.CreateBuffer(&rd);
  for (auto& rb : t.readbacks) {
    const wgpu::BufferDescriptor bd{.label = "XR timing readback",
                                    .usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst,
                                    .size = bytes};
    rb.buffer = device.CreateBuffer(&bd);
  }
  for (uint32_t z = 0; z < kZoneCount; ++z)
    t.writes[z] = {.querySet = t.queries, .beginningOfPassWriteIndex = z * 2, .endOfPassWriteIndex = z * 2 + 1};
  t.writes[kZoneFrame].endOfPassWriteIndex = wgpu::kQuerySetIndexUndefined;
  gfx::set_xr_first_pass_timing([]() -> const wgpu::PassTimestampWrites* {
    if (!g_sessionRunning || R.timing.frameBegun.exchange(true))
      return nullptr;
    return zone_writes(kZoneFrame);
  });
}

// After the frame's passes are encoded: resolve this frame's timestamps into
// a free readback buffer (mapped in end_frame, after the submit).
void resolve_timing(const wgpu::CommandEncoder& cmd) {
  auto& t = R.timing;
  if (!t.queries || t.zones == 0)
    return;
  for (int i = 0; i < static_cast<int>(t.readbacks.size()); ++i) {
    auto& rb = t.readbacks[i];
    if (rb.busy)
      continue;
    cmd.ResolveQuerySet(t.queries, 0, kZoneCount * 2, t.resolve, 0);
    cmd.CopyBufferToBuffer(t.resolve, 0, rb.buffer, 0, kZoneCount * 2 * sizeof(uint64_t));
    rb.busy = true;
    rb.zones = t.zones;
    t.pending = i;
    break;
  }
  t.zones = 0;
  t.frameBegun = false;
}

void map_timing() {
  auto& t = R.timing;
  if (t.pending < 0)
    return;
  const int index = t.pending;
  t.pending = -1;
  t.readbacks[index].buffer.MapAsync(
      wgpu::MapMode::Read, 0, kZoneCount * 2 * sizeof(uint64_t), wgpu::CallbackMode::AllowProcessEvents,
      [index](wgpu::MapAsyncStatus status, wgpu::StringView) {
        auto& t = R.timing;
        auto& rb = t.readbacks[index];
        if (status == wgpu::MapAsyncStatus::Success) {
          const auto* ts = static_cast<const uint64_t*>(rb.buffer.GetConstMappedRange());
          for (uint32_t z = 0; z < kZoneCount; ++z) {
            if (z != kZoneFrame && (rb.zones & (1u << z)) && ts[z * 2 + 1] > ts[z * 2]) {
              t.ns[z] += static_cast<double>(ts[z * 2 + 1] - ts[z * 2]);
              ++t.samples[z];
              if (z == kZone3D)
                t.last3DNs = static_cast<double>(ts[z * 2 + 1] - ts[z * 2]);
            }
          }
          if (rb.zones & (1u << kZoneFrame)) {
            uint64_t end = 0;
            for (uint32_t z = 0; z < kZoneCount; ++z)
              if (z != kZoneFrame && (rb.zones & (1u << z)))
                end = std::max(end, ts[z * 2 + 1]);
            // From the eye pass when there is one: the passes before it
            // (the fighters' shadows, AURORA_XR_PRESUBMIT) run while the GPU
            // would idle waiting for the eyes' image, and counting that wait
            // made every frame look 3-4 ms dearer (four-CPU Brinstar and
            // Brinstar Depths: 5.5-6.5 ms "outside the eyes", 2.5 without
            // the wait), so dynamic resolution sat at its floor.
            const uint64_t begin = (rb.zones & (1u << kZone3D)) && ts[kZone3D * 2] > ts[kZoneFrame * 2]
                                       ? ts[kZone3D * 2]
                                       : ts[kZoneFrame * 2];
            if (end > begin) {
              t.lastFrameNs = static_cast<double>(end - begin);
              ++t.frameSeq;
              // AURORA_XR_LOG_GPU_SPIKES=<ms> (default 12, 0 off): frames
              // whose GPU span runs that long, pass by pass, to tell our own
              // work getting dearer from the GPU being taken from us.
              static const float spikeMs = env_float("AURORA_XR_LOG_GPU_SPIKES", 12.f);
              static auto spikeWindow = std::chrono::steady_clock::now();
              static int spikesLogged = 0;
              if (spikeMs > 0.f && t.lastFrameNs >= spikeMs * 1e6) {
                const auto now = std::chrono::steady_clock::now();
                if (now - spikeWindow > std::chrono::seconds(10)) {
                  spikeWindow = now;
                  spikesLogged = 0;
                }
                if (++spikesLogged <= 30) {
                  std::string passes;
                  for (uint32_t z = 0; z < kZoneCount; ++z)
                    if (z != kZoneFrame && (rb.zones & (1u << z)) && ts[z * 2 + 1] > ts[z * 2])
                      passes += fmt::format(", {} {:.1f}", kZoneNames[z], (ts[z * 2 + 1] - ts[z * 2]) / 1e6);
                  Log.info("GPU frame spike: {:.1f} ms{}; head turning {:.0f} deg/s", t.lastFrameNs / 1e6, passes,
                           g_headTurnLast.load(std::memory_order_relaxed));
                  g_spikeSeen = true;
                }
              }
            }
          }
          ++t.frames;
          rb.buffer.Unmap();
        }
        rb.busy = false;
        const auto now = std::chrono::steady_clock::now();
        if (now - t.lastLog >= log_period() && t.frames > 0) {
          std::string line;
          double total = 0;
          for (uint32_t z = 0; z < kZoneCount; ++z) {
            const double us = t.samples[z] ? t.ns[z] / t.samples[z] / 1000.0 : 0.0;
            total += us;
            line += fmt::format("{}{} {:.0f}", z ? ", " : "", kZoneNames[z], us);
          }
          float turnMax, turnMean;
          {
            std::lock_guard lock{g_viewMutex};
            turnMax = g_headTurnMax;
            turnMean = g_headTurnN ? static_cast<float>(g_headTurnSum / g_headTurnN) : 0.f;
            g_headTurnMax = 0.f;
            g_headTurnSum = 0.0;
            g_headTurnN = 0;
          }
          Log.info("GPU 3D passes per frame (us): {} (total {:.0f}); {:.0f} world draws per frame; head turning "
                   "{:.0f} deg/s mean, {:.0f} most",
                   line, total, t.worldFrames ? static_cast<double>(t.worldDraws) / t.worldFrames : 0.0, turnMean,
                   turnMax);
          t.worldDraws = 0;
          t.worldFrames = 0;
          t.ns = {};
          t.samples = {};
          t.frames = 0;
          t.lastLog = now;
        }
      });
}

// The 3D frame hook drew this frame (render worker), so the flat present can
// be skipped: nothing shows the virtual screen during a fight.
bool g_skipPresent = false;
uint32_t g_xrSamples = 1; // AURORA_XR_MSAA: the 3D eyes' sample count

ReplayTarget make_replay_target(const gfx::RenderTargetLayout& layout, uint32_t width, uint32_t height,
                                const char* label) {
  auto& device = webgpu::g_device;
  ReplayTarget rt;
  auto& t = rt.target;
  t.layout = layout;
  t.size = {width, height, 1};
  t.clearColor = {0, 0, 0, 0};
  t.clearDepth = gx::UseReversedZ ? 0.f : 1.f;
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    const bool scene = i == gfx::SceneColorAttachmentIndex;
    wgpu::TextureDescriptor td{
        .label = label,
        .usage = wgpu::TextureUsage::RenderAttachment |
                 (scene && layout.sampleCount == 1 ? wgpu::TextureUsage::TextureBinding : wgpu::TextureUsage::None),
        .size = t.size,
        .format = layout.colorAttachments[i].format,
        .sampleCount = layout.sampleCount,
    };
    auto tex = device.CreateTexture(&td);
    t.colorViews[i] = tex.CreateView();
    rt.textures.push_back(tex);
    if (scene) {
      if (layout.sampleCount > 1) {
        td.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
        td.sampleCount = 1;
        auto resolve = device.CreateTexture(&td);
        t.resolveViews[i] = resolve.CreateView();
        rt.sampleColor = t.resolveViews[i];
        rt.textures.push_back(resolve);
      } else {
        rt.sampleColor = t.colorViews[i];
      }
    }
  }
  const wgpu::TextureDescriptor dd{
      .label = label,
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .size = t.size,
      .format = layout.depthStencilFormat,
      .sampleCount = layout.sampleCount,
  };
  auto depth = device.CreateTexture(&dd);
  t.depthView = depth.CreateView();
  rt.depth = t.depthView;
  rt.textures.push_back(depth);
  return rt;
}

wgpu::Buffer make_buffer(uint64_t size, wgpu::BufferUsage usage, const char* label) {
  const wgpu::BufferDescriptor bd{.label = label, .usage = usage | wgpu::BufferUsage::CopyDst, .size = size};
  return webgpu::g_device.CreateBuffer(&bd);
}

bool ensure_renderer(const gfx::RenderTargetLayout& layout) {
  if (R.failed)
    return false;
  if (R.compose && R.layoutKey == layout.key && R.sampleCount == layout.sampleCount)
    return true;
  auto& device = webgpu::g_device;
  const uint32_t eyeW = g_streams[kStereo].width / 2, eyeH = g_streams[kStereo].height;
  R.targets[0] = make_replay_target(layout, eyeW, eyeH, "XR eye 0");
  R.targets[1] = make_replay_target(layout, eyeW, eyeH, "XR eye 1");
  R.targets[2] = make_replay_target(layout, g_streams[kHud].width, g_streams[kHud].height, "XR HUD");
  R.targets[2].target.fullViewport = false; // the HUD keeps the game's viewports

  for (int i = 0; i < 2; ++i) {
    for (uint32_t t = 0; t < gfx::XrMaxTransforms; ++t) {
      R.eyeUniforms[i][t] = make_buffer(gx::XrEyeUniformSize, wgpu::BufferUsage::Uniform, "XR eye matrix");
      const wgpu::BindGroupEntry e{.binding = 0, .buffer = R.eyeUniforms[i][t], .size = gx::XrEyeUniformSize};
      const wgpu::BindGroupDescriptor bg{.layout = gx::g_xrEyeBindGroupLayout, .entryCount = 1, .entries = &e};
      R.eyeGroups[i][t] = device.CreateBindGroup(&bg);
    }
    R.targets[i].target.views[0].xrBindGroups = R.eyeGroups[i];
  }
  if (g_multiview) {
    for (uint32_t t = 0; t < gfx::XrMaxTransforms; ++t) {
      R.mvUniforms[t] = make_buffer(kMultiviewEyeSize, wgpu::BufferUsage::Uniform, "XR eye matrices (multiview)");
      const wgpu::BindGroupEntry e{.binding = 0, .buffer = R.mvUniforms[t], .size = kMultiviewEyeSize};
      const wgpu::BindGroupDescriptor bg{.layout = gx::g_xrEyeBindGroupLayout, .entryCount = 1, .entries = &e};
      R.mvGroups[t] = device.CreateBindGroup(&bg);
    }
  }

  // Composite pipeline into the shared images (RGBA8Unorm).
  std::string source = kComposeShader;
  const std::string depthType =
      layout.sampleCount > 1 ? "texture_depth_multisampled_2d" : "texture_depth_2d";
  source.replace(source.find("DEPTH_TYPE"), std::string_view{"DEPTH_TYPE"}.size(), depthType);
  wgpu::ShaderSourceWGSL wgsl{};
  wgsl.code = source.c_str();
  const wgpu::ShaderModuleDescriptor smd{.nextInChain = &wgsl, .label = "XR compose"};
  const auto module = device.CreateShaderModule(&smd);
  const std::array entries{
      wgpu::BindGroupLayoutEntry{.binding = 0,
                                 .visibility = wgpu::ShaderStage::Fragment,
                                 .texture = {.sampleType = wgpu::TextureSampleType::UnfilterableFloat,
                                             .viewDimension = wgpu::TextureViewDimension::e2D}},
      wgpu::BindGroupLayoutEntry{.binding = 1,
                                 .visibility = wgpu::ShaderStage::Fragment,
                                 .texture = {.sampleType = wgpu::TextureSampleType::Depth,
                                             .viewDimension = wgpu::TextureViewDimension::e2D,
                                             .multisampled = layout.sampleCount > 1}},
      wgpu::BindGroupLayoutEntry{.binding = 2,
                                 .visibility = wgpu::ShaderStage::Fragment,
                                 .buffer = {.type = wgpu::BufferBindingType::Uniform, .minBindingSize = 16}},
  };
  const wgpu::BindGroupLayoutDescriptor bgld{.label = "XR compose", .entryCount = entries.size(),
                                             .entries = entries.data()};
  R.composeLayout = device.CreateBindGroupLayout(&bgld);
  const wgpu::PipelineLayoutDescriptor pld{.bindGroupLayoutCount = 1, .bindGroupLayouts = &R.composeLayout};
  const auto pipelineLayout = device.CreatePipelineLayout(&pld);
  const wgpu::ColorTargetState colorTarget{.format = wgpu::TextureFormat::RGBA8Unorm};
  const wgpu::FragmentState fragment{.module = module, .entryPoint = "fs", .targetCount = 1, .targets = &colorTarget};
  const wgpu::RenderPipelineDescriptor rpd{
      .label = "XR compose",
      .layout = pipelineLayout,
      .vertex = {.module = module, .entryPoint = "vs"},
      .fragment = &fragment,
  };
  R.compose = device.CreateRenderPipeline(&rpd);

  const float hudAlpha = std::clamp(env_float("AURORA_XR_HUD_BACKDROP", 0.f), 0.f, 1.f);
  for (int i = 0; i < 3; ++i) {
    struct {
      float offset[2];
      uint32_t mode;
      float alpha;
    } p{{i == 1 ? static_cast<float>(eyeW) : 0.f, 0.f}, i == 2 ? 1u : 0u, hudAlpha};
    R.params[i] = make_buffer(sizeof(p), wgpu::BufferUsage::Uniform, "XR compose params");
    webgpu::g_queue.WriteBuffer(R.params[i], 0, &p, sizeof(p));
    const std::array bge{
        wgpu::BindGroupEntry{.binding = 0, .textureView = R.targets[i].sampleColor},
        wgpu::BindGroupEntry{.binding = 1, .textureView = R.targets[i].depth},
        wgpu::BindGroupEntry{.binding = 2, .buffer = R.params[i], .size = sizeof(p)},
    };
    const wgpu::BindGroupDescriptor bgd{.layout = R.composeLayout, .entryCount = bge.size(), .entries = bge.data()};
    R.composeGroups[i] = device.CreateBindGroup(&bgd);
  }
  if (!R.timing.queries)
    setup_timing();
  R.layoutKey = layout.key;
  R.sampleCount = layout.sampleCount;
  Log.info("Flat-layout targets ready (HUD, per-eye fallback): eyes {}x{}, HUD {}x{}, {}x MSAA", eyeW, eyeH,
           g_streams[kHud].width,
           g_streams[kHud].height, layout.sampleCount);
  return true;
}

// Coverage into alpha at the end of the direct 3D pass, in tile memory: two
// full-screen triangles at the far plane (reversed Z: depth 0) that write
// only alpha. Where the depth is still cleared ("equal") alpha becomes 0;
// where geometry wrote depth ("less" than it) alpha becomes 1. Depth-less
// draws (translucent effects) keep their color over alpha 0, which the
// premultiplied projection layer adds over passthrough as light.
constexpr char kCoverageShader[] = R"(
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs_clear() -> @location(0) vec4f { return vec4f(0.0, 0.0, 0.0, 0.0); }
@fragment fn fs_set() -> @location(0) vec4f { return vec4f(0.0, 0.0, 0.0, 1.0); }
)";

bool ensure_direct(const gfx::RenderTargetLayout& layout) {
  if (R.coverClear && R.directKey == layout.key && R.directSamples == g_xrSamples)
    return true;
  auto& device = webgpu::g_device;
  const auto& stereo = g_streams[kStereo];
  const wgpu::Extent3D size{stereo.width, stereo.height, stereo.layers};
  const wgpu::TextureViewDescriptor layered{.dimension = wgpu::TextureViewDimension::e2DArray,
                                            .arrayLayerCount = stereo.layers};
  const wgpu::TextureViewDescriptor* viewDesc = stereo.layers > 1 ? &layered : nullptr;
  // MSAA: every attachment multisampled and, where the device allows,
  // transient: on a tiler they then live only in tile memory, and only the
  // resolve into the shared image reaches memory.
  // AURORA_XR_TRANSIENT=0: ordinary attachments. Dawn's validation
  // (AURORA_DAWN_VALIDATION) rejects transient arrays, which Vulkan allows.
  const bool msaa = g_xrSamples > 1;
  const auto usage = wgpu::TextureUsage::RenderAttachment |
                     (msaa && device.HasFeature(wgpu::FeatureName::TransientAttachments) &&
                              env_flag("AURORA_XR_TRANSIENT", true)
                          ? wgpu::TextureUsage::TransientAttachment
                          : wgpu::TextureUsage::None);
  const wgpu::TextureDescriptor dd{.label = "XR 3D depth",
                                   .usage = usage,
                                   .size = size,
                                   .format = layout.depthStencilFormat,
                                   .sampleCount = g_xrSamples};
  R.directDepth = device.CreateTexture(&dd);
  R.directExtras.clear();
  R.directExtraViews = {};
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    if (i == gfx::SceneColorAttachmentIndex && !msaa)
      continue; // the shared image itself
    const wgpu::TextureDescriptor td{.label = i == gfx::SceneColorAttachmentIndex ? "XR 3D color (MSAA)"
                                                                                   : "XR 3D extra attachment",
                                     .usage = usage,
                                     .size = size,
                                     .format = layout.colorAttachments[i].format,
                                     .sampleCount = g_xrSamples};
    auto tex = device.CreateTexture(&td);
    R.directExtraViews[i] = tex.CreateView(viewDesc);
    R.directExtras.push_back(tex);
  }
  wgpu::ShaderSourceWGSL wgsl{};
  wgsl.code = kCoverageShader;
  const wgpu::ShaderModuleDescriptor smd{.nextInChain = &wgsl, .label = "XR coverage"};
  const auto module = device.CreateShaderModule(&smd);
  const wgpu::PipelineLayoutDescriptor pld{.bindGroupLayoutCount = 0};
  const auto pipelineLayout = device.CreatePipelineLayout(&pld);
  std::array<wgpu::ColorTargetState, gfx::MaxColorAttachments> targets{};
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    targets[i] = {.format = layout.colorAttachments[i].format,
                  .writeMask = i == gfx::SceneColorAttachmentIndex ? wgpu::ColorWriteMask::Alpha
                                                                   : wgpu::ColorWriteMask::None};
  }
  const auto make = [&](const char* entry, wgpu::CompareFunction compare) {
    const wgpu::FragmentState fragment{.module = module,
                                       .entryPoint = entry,
                                       .targetCount = layout.colorAttachmentCount,
                                       .targets = targets.data()};
    const wgpu::DepthStencilState depth{.format = layout.depthStencilFormat,
                                        .depthWriteEnabled = wgpu::OptionalBool::False,
                                        .depthCompare = compare};
    wgpu::RenderPipelineMultiview multiview{};
    multiview.viewMask = (1u << stereo.layers) - 1u;
    const wgpu::RenderPipelineDescriptor rpd{
        .nextInChain = stereo.layers > 1 ? &multiview : nullptr,
        .label = "XR coverage",
        .layout = pipelineLayout,
        .vertex = {.module = module, .entryPoint = "vs"},
        .depthStencil = &depth,
        .multisample = {.count = g_xrSamples},
        .fragment = &fragment,
    };
    return device.CreateRenderPipeline(&rpd);
  };
  // Reversed Z: cleared depth is 0 (far); the triangles sit at 0.
  R.coverClear = make("fs_clear", gx::UseReversedZ ? wgpu::CompareFunction::Equal : wgpu::CompareFunction::Equal);
  R.coverSet = make("fs_set", gx::UseReversedZ ? wgpu::CompareFunction::Less : wgpu::CompareFunction::Greater);
  R.directKey = layout.key;
  R.directSamples = g_xrSamples;
  // What every eye sample carries in tile memory: more bytes, smaller bins,
  // and every draw is replayed per bin.
  std::string attachments;
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i)
    attachments += fmt::format("{}{}", i ? ", " : "", static_cast<int>(layout.colorAttachments[i].format));
  Log.info("3D eye attachments: color formats [{}], depth format {}, {}x MSAA, {}x{} x{} layers", attachments,
           static_cast<int>(layout.depthStencilFormat), g_xrSamples, stereo.width, stereo.height, stereo.layers);
  return true;
}

void draw_coverage(const wgpu::RenderPassEncoder& pass, void*) {
  pass.SetPipeline(R.coverClear);
  pass.Draw(3);
  pass.SetPipeline(R.coverSet);
  pass.Draw(3);
}

void compose(const wgpu::CommandEncoder& cmd, const wgpu::Texture& dst, std::initializer_list<int> sources,
             TimingZone zone) {
  const auto view = dst.CreateView();
  const wgpu::RenderPassColorAttachment ca{
      .view = view,
      .loadOp = wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {0, 0, 0, 0},
  };
  const wgpu::RenderPassDescriptor rpd{.label = "XR compose",
                                       .colorAttachmentCount = 1,
                                       .colorAttachments = &ca,
                                       .timestampWrites = zone_writes(zone)};
  auto pass = cmd.BeginRenderPass(&rpd);
  pass.SetPipeline(R.compose);
  for (int i : sources) {
    const auto& size = R.targets[i].target.size;
    const float x = i == 1 ? static_cast<float>(size.width) : 0.f;
    pass.SetViewport(x, 0.f, static_cast<float>(size.width), static_cast<float>(size.height), 0.f, 1.f);
    pass.SetBindGroup(0, R.composeGroups[i]);
    pass.Draw(3);
  }
  pass.End();
}

// gfx frame hook: runs on the render worker once every pass of a frame is
// encoded. Re-draws the fight per eye and the HUD into the shared images.
void render_3d_frame(const wgpu::CommandEncoder& cmd, gfx::detail::FramePacket& frame);

// Dynamic resolution's controller, once per 3D frame (render worker), after
// Martin Fuller's DRS practice (a live cost table): steer by the whole
// frame's GPU time (timestamps from the first flat pass to the last XR pass,
// compositor preemptions included, so the jumbotron's grab or anything else
// in the frame counts), record it per scale step for the stage and mode, and
// pick from what each step actually cost. Steps not measured lately are
// predicted from a fit of cost against pixels over the ones that were,
// blended with their own history by how much there is of it and how fresh.
// The fit is only trusted across a spread of scales; until then a quarter
// of the frame is taken to scale with pixels.
// Much of a frame's cost doesn't scale with pixels at all (Brinstar: about
// 9.6 of 11 ms), so a fixed model either gives resolution away for nothing
// or overshoots. Down past target + band at once, to the best step that
// fits; up one step at a time after a dwell, only to a step that fits. A
// run of missed swapchain images (two within a second) with the GPU near the
// target is a panic: the step is marked dearer than measured and the scale
// drops a step. Single misses come at the same rate at any scale (Brinstar:
// about one in 13 s), and with the GPU well under the target they are the
// CPU's, so those are only counted. Images the headset held long (g_stereoLate) are only
// logged.
namespace {
constexpr double kDynresFullSamples = 30;  // samples for full trust in a step
constexpr double kDynresForget = 1200;     // frames for a step's trust to fall to 1/e (20 s)
constexpr int kDynresUpDwell = 30;         // frames between steps up

double dynres_trust(const DynresCost& c, uint64_t frame) {
  if (c.samples <= 0)
    return 0;
  return std::min(c.samples / kDynresFullSamples, 1.0) *
         std::exp(-static_cast<double>(frame - c.lastFrame) / kDynresForget);
}

float dynres_scale(int level) { return g_dynres.minScale + kDynresStep * static_cast<float>(level); }

int dynres_levels() {
  const auto& dr = g_dynres;
  return std::clamp(static_cast<int>((dr.maxScale - dr.minScale) / kDynresStep + 0.5f) + 1, 1, kDynresLevels);
}

// Each step's predicted frame GPU time: its own history where trusted,
// otherwise a weighted fit ns = a + b * scale^2 over the trusted steps (b at
// least 0); with a single measured point, half of it is taken to scale with
// pixels until more are in.
void dynres_predict(const std::array<DynresCost, kDynresLevels>& costs, uint64_t frame, int levels,
                    std::array<double, kDynresLevels>& out) {
  double sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (int i = 0; i < levels; ++i) {
    const double w = dynres_trust(costs[i], frame);
    if (w <= 0)
      continue;
    const double x = static_cast<double>(dynres_scale(i)) * dynres_scale(i);
    sw += w;
    sx += w * x;
    sy += w * costs[i].ns;
    sxx += w * x * x;
    sxy += w * x * costs[i].ns;
  }
  double a = 0, b = 0;
  if (sw > 0) {
    const double mx = sx / sw, my = sy / sw;
    const double var = sxx / sw - mx * mx;
    // A fit across a narrow spread of scales mostly measures how the scene
    // changed between visits: until the steps span enough, assume half of
    // the eye pass scales with pixels (Mute City's: 9.8 ms at 1.0, 11.8 at
    // 1.3, so about 0.4 of it).
    double lo = 1e9, hi = 0;
    for (int i = 0; i < levels; ++i)
      if (dynres_trust(costs[i], frame) > 0) {
        const double x = static_cast<double>(dynres_scale(i)) * dynres_scale(i);
        lo = std::min(lo, x);
        hi = std::max(hi, x);
      }
    // At most all of the eye pass scales with pixels: a fit that says more
    // is reading a scene change as a resolution cost.
    b = hi - lo >= 0.1 && var > 1e-6 ? std::clamp((sxy / sw - mx * my) / var, 0.0, my / mx) : 0.5 * my / mx;
    a = my - b * mx;
  }
  for (int i = 0; i < levels; ++i) {
    const double x = static_cast<double>(dynres_scale(i)) * dynres_scale(i);
    const double w = dynres_trust(costs[i], frame);
    const double fit = sw > 0 ? a + b * x : 0;
    out[i] = sw > 0 ? w * costs[i].ns + (1 - w) * fit : 0;
  }
}

// The GPU time one display frame leaves a frame of ours, for a game drawing every display frame.
double display_frame_budget_ns(double headroomNs) {
  const float hz = g_displayHz > 0.f ? g_displayHz.load() : 90.f;
  const float compositorMs = g_compositorGpuMs > 0.f ? g_compositorGpuMs.load() : 2.3f;
  return 1.0e9 / hz - compositorMs * 1.0e6 - headroomNs;
}
} // namespace

void update_dynamic_resolution(bool missed) {
  auto& dr = g_dynres;
  const auto now = std::chrono::steady_clock::now();
  ++dr.frameCount;
  if (dr.random) {
    // Exercise every size: resolution-dependent bugs show up as flicker.
    dr.scale = dr.minScale + (dr.maxScale - dr.minScale) * static_cast<float>(std::rand() % 1000) / 999.f;
  } else {
    int stage;
    {
      std::lock_guard lock{g_arenaMutex};
      stage = g_stage;
    }
    auto& costs = g_dynresCosts[stage * 2 + (g_passthroughWanted ? 1 : 0)];
    const int levels = dynres_levels();
    dr.level = std::clamp(dr.level, 0, levels - 1);
    // A heavy effect announced: down at once, before its first frame, to the
    // step keeping the share of pixels the last one needed. Off by default
    // (AURORA_XR_HINT=1): with the KO explosion's ring, beam and particles
    // out of the 3D view, KOs on four-CPU Brinstar held 11 images over 16 KOs
    // without it and 2 with it, but it dropped the scale (to 0.80) on all 15
    // it saw, where the readings below dropped it twice.
    static const bool hints = env_flag("AURORA_XR_HINT", false);
    if (g_heavyHint.exchange(false) && hints && dr.transientFrom < 0 && dr.level > 0) {
      const double fit = dynres_scale(dr.level) * std::sqrt(dr.hintPixels);
      int to = 0;
      for (int i = dr.level - 1; i > 0; --i)
        if (dynres_scale(i) <= fit) {
          to = i;
          break;
        }
      dr.transientFrom = dr.level;
      dr.transientBaseNs = dr.emaNs > 0 ? dr.emaNs : dr.spanNs > 0 ? dr.spanNs : dr.targetNs * 0.7;
      dr.transientSince = now;
      dr.transientTo = to;
      dr.overRun = dr.fitRun = 0;
      dr.transientHinted = true;
      dr.transientPeakEyeNs = 0;
      ++dr.transients;
      Log.info("Dynamic resolution: heavy effect ahead, {:.3f} -> {:.3f} ({:.0f}% of the pixels)",
               dynres_scale(dr.level), dynres_scale(to), 100.0 * dr.hintPixels);
    }
    // Margins to the ideal latency's slot, from the pacing probe.
    {
      std::vector<float> margins;
      {
        std::lock_guard lock{g_pace.mutex};
        margins.swap(g_pace.idealMargins);
      }
      for (float m : margins) {
        dr.margins[dr.marginNext++ % dr.margins.size()] = m;
        dr.marginCount = std::min(dr.marginCount + 1, dr.margins.size());
      }
      if (!margins.empty() && dr.marginCount >= 30) {
        std::array<float, 120> sorted{};
        std::copy_n(dr.margins.begin(), dr.marginCount, sorted.begin());
        const size_t at = dr.marginCount / 20;
        std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(at),
                         sorted.begin() + static_cast<std::ptrdiff_t>(dr.marginCount));
        dr.marginNs = sorted[at];
        dr.haveMargin = true;
      }
    }
    if (!dr.fixedTarget) {
      const double span = dr.emaNs > 0 ? dr.emaNs : dr.spanNs;
      if (dr.haveMargin && span > 0) {
        dr.targetNs = std::clamp(span + dr.marginNs - dr.wantMarginNs, 3.0e6, 20.0e6);
        // SpaceWarp: a staged frame finishing early says nothing about how
        // much GPU a game frame has. The budget is the frame less the
        // compositor's two display frames: margins of 18 ms had the target
        // at 20 and the scale climbing until frames missed.
        if (appsw_on()) {
          const float refresh = g_refreshHz > 0.f ? g_refreshHz.load() : 120.f;
          const float compositorMs = g_compositorGpuMs > 0.f ? g_compositorGpuMs.load() : 2.4f;
          const double budget = 1.0e9 / 60.0 - compositorMs * 1.0e6 * refresh / 60.0 - dr.headroomNs;
          dr.targetNs = std::clamp(std::min(dr.targetNs, budget), 3.0e6, 20.0e6);
        } else if (g_displayPerGameFrame == 0) {
          // A game on its own timer, drawing a frame per display frame (smstrikers-port at 72 or
          // 90 Hz): the span is how late frames have been coming, so a slow stretch would raise
          // its own budget. The budget is a display frame less the compositor's share.
          dr.targetNs = std::clamp(std::min(dr.targetNs, display_frame_budget_ns(dr.headroomNs)), 3.0e6, 20.0e6);
        }
      } else if (dr.spanNs == 0) {
        // Until margins come in: what a game frame leaves us on the GPU.
        const float hz = g_displayHz > 0.f ? g_displayHz.load() : 120.f;
        const float compositorMs = g_compositorGpuMs > 0.f ? g_compositorGpuMs.load() : 2.3f;
        dr.targetNs = g_displayPerGameFrame == 0
                          ? std::clamp(display_frame_budget_ns(dr.headroomNs), 3.0e6, 16.0e6)
                          : std::clamp(1.0e9 / 60.0 - compositorMs * 1.0e6 * hz / 60.0 - dr.headroomNs, 5.0e6, 16.0e6);
      }
    }
    // Shaders compiling (a stage's first fight of the session) stall the
    // frame on the CPU and inflate its span; less resolution can't help, and
    // those frames would make every step look dear for a long while. They're
    // left out of the costs and don't cut the scale (a panic still does).
    if (const uint32_t created = gfx::pipelines_created(); created != dr.pipelinesSeen) {
      dr.pipelinesSeen = created;
      dr.compileQuiet = 0;
    } else {
      ++dr.compileQuiet;
    }
    const bool compiling = dr.compileQuiet < 30;
    const auto& t = R.timing;
    if (t.frameSeq != dr.seenSeq && t.lastFrameNs > 0) {
      dr.seenSeq = t.frameSeq;
      // Readbacks lag a few frames: give a new scale time to show. A spike
      // half again over this visit's average (a hitch elsewhere) is left out,
      // but five in a row are the scene getting dearer: the average starts
      // over from them. (A bogus low reading, under 2 ms, once seeded the
      // average and every real frame after it looked like a spike.)
      const double ns = t.lastFrameNs;
      // The eye pass of the same frame (its share of the span), when timed.
      const double eyeNs = t.last3DNs > 0 && t.last3DNs < ns ? t.last3DNs : ns;
      const double emaBefore = dr.emaNs;
      bool spike = dr.emaNs > 0 && ns > dr.emaNs * 1.5;
      if (spike && ++dr.spikeRun >= 5) {
        dr.emaNs = 0;
        spike = false;
      }
      if (!spike)
        dr.spikeRun = 0;
      ++dr.readings;
      if (++dr.settle <= 4) {
        ++dr.skipSettle;
      } else if (compiling) {
        ++dr.skipCompile;
      } else if (dr.transientFrom >= 0) {
        ++dr.skipSpike; // a heavy moment's frames say nothing of the scene
      } else if (spike || ns < 2.0e6) {
        ++dr.skipSpike;
      } else {
        dr.emaNs = dr.emaNs == 0 ? ns : dr.emaNs * 0.85 + ns * 0.15;
        dr.spanNs = dr.spanNs == 0 ? ns : dr.spanNs * 0.95 + ns * 0.05;
        dr.restNs = dr.restNs == 0 ? ns - eyeNs : dr.restNs * 0.95 + (ns - eyeNs) * 0.05;
        dr.nsSum += ns;
        ++dr.nsFrames;
        auto& c = costs[dr.level];
        // What is left of the old history, then this sample on top.
        c.samples *= std::exp(-static_cast<double>(dr.frameCount - c.lastFrame) / kDynresForget);
        c.ns = c.samples <= 0 ? eyeNs : c.ns + (eyeNs - c.ns) / std::min(c.samples + 1, kDynresFullSamples);
        c.samples = std::min(c.samples + 1, kDynresFullSamples);
        c.lastFrame = dr.frameCount;
      }
      // Heavy moments: two readings in a row over the target and half again
      // over this step's average go straight down to the step whose pixels
      // fit the target (the eye pass scales with them). The step taken from
      // comes back as soon as three readings in a row say it would fit
      // again, without the brake's hold: a momentary effect mustn't leave
      // the fight at a low resolution for the brake's 15-60 s. Over 4 s, it
      // is the scene: the usual steps take over from where it is.
      if (dr.settle > 4 && !compiling && ns >= 2.0e6) {
        const auto fitStep = [&] {
          const double want = std::max(dr.targetNs - (ns - eyeNs), 0.0);
          const double fit = dynres_scale(dr.level) * std::sqrt(want / std::max(eyeNs, 1.0));
          for (int i = dr.level - 1; i > 0; --i)
            if (dynres_scale(i) <= fit)
              return i;
          return 0;
        };
        const bool over = ns > dr.targetNs + dr.bandNs;
        // AURORA_XR_TRANSIENT=0: no drop from the readings (announced effects
        // still drop: AURORA_XR_HINT).
        static const bool reactive = env_flag("AURORA_XR_TRANSIENT", true);
        if (dr.transientFrom < 0) {
          dr.overRun = reactive && over && emaBefore > 0 && ns > emaBefore * 1.5 ? dr.overRun + 1 : 0;
          if (dr.overRun >= 2 && dr.level > 0) {
            dr.transientFrom = dr.level;
            dr.transientBaseNs = emaBefore;
            dr.transientSince = now;
            dr.transientTo = fitStep();
            dr.overRun = dr.fitRun = 0;
            dr.transientHinted = false;
            ++dr.transients;
            Log.info("Dynamic resolution: heavy moment ({:.1f} ms against {:.1f}), {:.3f} -> {:.3f}", ns / 1e6,
                     emaBefore / 1e6, dynres_scale(dr.level), dynres_scale(dr.transientTo));
          }
        } else {
          const double from = dynres_scale(dr.transientFrom), at = dynres_scale(dr.level);
          const double atFrom = (ns - eyeNs) + eyeNs * (from * from) / (at * at);
          dr.transientPeakEyeNs = std::max(dr.transientPeakEyeNs, eyeNs * (from * from) / (at * at));
          // Back as soon as the step it came from fits the target again. (Not
          // the frame time before it: a busy fight's frames grew past that and
          // the moment never ended, leaving the scale at its floor.)
          dr.fitRun = atFrom <= dr.targetNs ? dr.fitRun + 1 : 0;
          dr.overRun = over ? dr.overRun + 1 : 0;
          if (dr.fitRun >= 3) {
            dr.transientTo = dr.transientFrom;
            // What the next announced one keeps: the pixels this one's worst
            // frame could have had within the target, half way from before.
            if (dr.transientHinted && dr.transientPeakEyeNs > 0) {
              const double keep =
                  std::clamp((dr.targetNs - dr.restNs) / dr.transientPeakEyeNs, 0.25, 1.0);
              dr.hintPixels = std::clamp(dr.hintPixels * 0.5 + keep * 0.5, 0.25, 1.0);
            }
          } else if (dr.overRun >= 2 && dr.level > 0) {
            dr.transientTo = fitStep();
            dr.overRun = 0;
          } else if (now - dr.transientSince > std::chrono::seconds(4)) {
            // Never low for longer than this: back to where it was, and the
            // usual steps take it down gradually if the scene really is
            // dearer now.
            Log.info("Dynamic resolution: heavy moment lasted over 4 s; back to {:.3f}", from);
            dr.transientTo = dr.transientFrom;
          }
        }
      }
    }
    const bool heavyMoment = dr.transientFrom >= 0 || now < dr.brakeQuietUntil;
    // A run of missed 3D images (two within a second) is a panic; a lone
    // miss comes at the same rate at any scale. (The render worker's own
    // cadence jitters, so a long gap between 3D frames is no signal.)
    bool panic = false;
    if (missed) {
      dr.recentMisses[dr.recentIndex++ % dr.recentMisses.size()] = dr.frameCount;
      int recent = 0;
      for (uint64_t f : dr.recentMisses)
        recent += f != 0 && dr.frameCount - f < 60 ? 1 : 0;
      // With the GPU well under the target the misses are the CPU's, and
      // less resolution can't help (Mushroom Kingdom II at 8.3 ms).
      panic = recent >= 2 && (dr.emaNs == 0 || dr.emaNs > dr.targetNs - 2.5e6) && !heavyMoment;
      if (!panic)
        ++dr.loneMisses;
    }
    // The brake. Wanting 1 ms of margin at the 5th percentile, the scale
    // climbed into the frames' tail: with eyes deferred on four-fighter
    // Stadium rock it went to 1.20 and stale frames rose from 1.3 a second
    // to 2.2. More than 2 frames over a millisecond late within 2 s, above
    // the lowest step, and the scale goes down a step and holds there for
    // 15 s (30, 60 if it brakes again within that).
    bool brake = false;
    if (const uint32_t lateNow = g_lateFrames.load(std::memory_order_relaxed); lateNow != dr.lateFramesSeen) {
      for (uint32_t i = dr.lateFramesSeen; i != lateNow && !heavyMoment; ++i)
        dr.recentLate[dr.recentLateIndex++ % dr.recentLate.size()] = dr.frameCount;
      dr.lateFramesSeen = lateNow;
      int recent = 0;
      for (uint64_t f : dr.recentLate)
        recent += f != 0 && dr.frameCount - f < 240 ? 1 : 0;
      if (recent > 2 && dr.level > 0) {
        brake = true;
        std::fill(dr.recentLate.begin(), dr.recentLate.end(), 0);
        const int held = now < dr.ceilingUntil ? std::min(dr.brakes + 1, 2) : 0;
        dr.brakes = held;
        dr.ceiling = dr.level - 1;
        dr.ceilingUntil = now + std::chrono::seconds(15 << held);
        ++dr.brakesSinceLog;
      }
    }
    const int ceiling = now < dr.ceilingUntil ? dr.ceiling : INT32_MAX;
    int next = dr.level;
    bool back = false; // back from a heavy moment: above any ceiling
    if (dr.transientTo >= 0) {
      next = dr.transientTo;
      if (dr.transientTo == dr.transientFrom) {
        back = true;
        Log.info("Dynamic resolution: back to {:.3f} after {:.2f} s", dynres_scale(next),
                 std::chrono::duration<double>(now - dr.transientSince).count());
        dr.transientFrom = -1;
        // Frames made late by the moment, still coming in, mustn't brake.
        dr.brakeQuietUntil = now + std::chrono::seconds(1);
        std::fill(dr.recentLate.begin(), dr.recentLate.end(), 0);
        std::fill(dr.recentMisses.begin(), dr.recentMisses.end(), 0);
      }
      dr.transientTo = -1;
    } else if (heavyMoment) {
      // While one lasts, the readings above decide.
    } else if (brake) {
      next = dr.level - 1;
    } else if (panic) {
      // This step is dearer than it measured: remembered past the band.
      auto& c = costs[dr.level];
      c.ns = std::max(c.ns, dr.targetNs + dr.bandNs - dr.restNs) + 0.5e6;
      c.samples = std::max(c.samples, kDynresFullSamples / 2);
      c.lastFrame = dr.frameCount;
      std::fill(dr.recentMisses.begin(), dr.recentMisses.end(), 0);
      next = dr.level - 1;
      ++dr.panics;
    } else if (dr.emaNs > 0 && dr.settle > 4) {
      std::array<double, kDynresLevels> predicted{};
      dynres_predict(costs, dr.frameCount, levels, predicted);
      // Eye passes predicted, plus the rest of the frame: whole frames.
      for (int i = 0; i < levels; ++i)
        predicted[i] += dr.restNs;
      if (dr.emaNs > dr.targetNs + dr.bandNs && !compiling) {
        // Down at once to the best step predicted to fit, at least one.
        next = 0;
        for (int i = dr.level - 1; i > 0; --i)
          if (predicted[i] <= dr.targetNs) {
            next = i;
            break;
          }
      } else if (++dr.sinceStep >= kDynresUpDwell && dr.level + 1 < levels &&
                 // The margins come from lock-step pacing's probe; a game on its own timer has none,
                 // and its target is a display frame's budget from the start.
                 (dr.haveMargin || dr.fixedTarget || g_displayPerGameFrame == 0) &&
                 predicted[dr.level + 1] <= dr.targetNs) {
        next = dr.level + 1;
      }
    }
    if (next > dr.level && !back)
      next = std::min(next, std::max(ceiling, dr.level));
    // Below 1.0 only while SpaceWarp is on (see minScale), or for a game drawing every display
    // frame on its own timer, which has no repeated frame to fall back on: less resolution is its
    // only way to keep up.
    const bool belowNative = appsw_on() || g_displayPerGameFrame == 0;
    const int floor = belowNative ? 0 : std::max(static_cast<int>((1.f - dr.minScale) / kDynresStep + 0.5f), 0);
    next = std::clamp(next, std::min(floor, levels - 1), levels - 1);
    if (next != dr.level) {
      dr.level = next;
      dr.emaNs = 0;
      dr.settle = 0;
      dr.sinceStep = 0;
      // Margins at the old step say nothing about this one.
      dr.marginCount = dr.marginNext = 0;
      dr.haveMargin = false;
    }
    dr.scale = dynres_scale(dr.level);
  }
  dr.lastFrame = now;
  const uint32_t late = g_stereoLate.load();
  dr.lateImages += late - dr.lateSeen;
  dr.lateSeen = late;
  dr.scaleSum += dr.scale;
  ++dr.frames;
  dr.lowest = std::min(dr.lowest, dr.scale);
  dr.highest = std::max(dr.highest, dr.scale);
  if (now - dr.lastLog >= log_period()) {
    if (dr.lastLog.time_since_epoch().count() != 0 && dr.frames > 0) {
      // The stage's cost table: each step measured lately, as scale:ms.
      std::string table;
      if (!dr.random) {
        int stage;
        {
          std::lock_guard lock{g_arenaMutex};
          stage = g_stage;
        }
        const auto& costs = g_dynresCosts[stage * 2 + (g_passthroughWanted ? 1 : 0)];
        for (int i = 0; i < dynres_levels(); ++i)
          if (dynres_trust(costs[i], dr.frameCount) > 0.1)
            table += fmt::format(" {:.3f}:{:.1f}", dynres_scale(i), costs[i].ns / 1.0e6);
      }
      Log.info("Dynamic resolution: scale {:.2f} average ({:.2f}-{:.2f}); frame GPU {:.1f} ms average against "
               "{:.1f} (margin to the ideal slot {:.1f} ms at 5%), {:.1f} of it outside the eyes; {} panics, "
               "{} lone misses; {} brakes for late frames; {} images held long; {} timings ({} settling, {} compiling, {} spikes or bogus left out); "
               "eye pass costs{}",
               dr.scaleSum / dr.frames, dr.lowest, dr.highest, dr.nsFrames ? dr.nsSum / dr.nsFrames / 1.0e6 : 0.0,
               dr.targetNs / 1.0e6, dr.haveMargin ? dr.marginNs / 1.0e6 : 0.0, dr.restNs / 1.0e6,
               dr.panics, dr.loneMisses, dr.brakesSinceLog, dr.lateImages, dr.readings, dr.skipSettle, dr.skipCompile, dr.skipSpike,
               table);
    }
    dr.scaleSum = dr.nsSum = 0;
    dr.frames = dr.nsFrames = dr.panics = dr.loneMisses = dr.lateImages = 0;
    dr.brakesSinceLog = 0;
    dr.readings = dr.skipSettle = dr.skipCompile = dr.skipSpike = 0;
    dr.lowest = 99.f;
    dr.highest = 0.f;
    dr.lastLog = now;
  }
}

// Frame hook: the 3D views, then whether the next frame's flat world draws
// are needed (not while fights go to the headset with no flat present).
// A missed 3D image counts too: drawing the flat world then made the next
// frame heavier and the next miss likelier, so slow stages kept sliding.
void render_3d(const wgpu::CommandEncoder& cmd, gfx::detail::FramePacket& frame) {
  // After early eyes, R describes their image (already released).
  if (frame.xrEyesDone && frame.xrWorldResumed)
    ++g_earlyResumed;
  if (!frame.xrEyesDone || frame.xrWorldResumed) {
    R.renderedStereo = false;
    R.missedStereo = false;
    R.renderedRect = {};
    R.renderedTick = frame.xrTickFrame;
    pace_frame(frame.xrRecordStartNs, frame.xrWorldEndNs, frame.xrGameDoneNs, frame.xrRecordEndNs);
  }
  render_3d_frame(cmd, frame);
  gfx::set_xr_drop_flat_world((R.renderedStereo || R.missedStereo) && g_skipPresent &&
                              !env_flag("AURORA_XR_FLAT_WORLD", false));
}

// The eyes: the frame's world draws (only `passes`, when given) replayed per
// eye into the shared 3D image, with `transforms` as the world placements.
// False where nothing was encoded (no views, renderer, or the fallback
// compose can't apply); R says whether a 3D image was rendered or missed.
// Application SpaceWarp: this frame's motion vectors and depth, at the
// runtime's recommended size, for the two SpaceWarp images (handed over
// with the 3D image, release_motion). The opaque world draws are replayed
// once more, both eyes at once, with their motion twins: each writes its
// surface's current minus previous NDC position (xr.prev, xr.prev_m) and its
// depth. Blended and cut-out draws are left out; where nothing opaque is,
// the motion is zero and the depth infinitely far. AURORA_XR_APPSW_MOTION=0
// only clears them.
void encode_motion(const wgpu::CommandEncoder& cmd, gfx::detail::FramePacket& frame,
                   const std::vector<const gfx::detail::RenderPass*>* passes, const wgpu::Texture& color,
                   const wgpu::Texture& depth) {
  const auto* layout = gfx::xr_motion_layout();
  const wgpu::TextureViewDescriptor layered{.dimension = wgpu::TextureViewDimension::e2DArray, .arrayLayerCount = 2};
  if (layout != nullptr && env_flag("AURORA_XR_APPSW_MOTION", true)) {
    const auto source = gfx::scene_render_target_layout();
    gfx::XrReplayTarget t;
    t.layout = *layout;
    t.sourceLayout = &source;
    t.motion = true;
    t.size = {g_streams[kMotion].width, g_streams[kMotion].height, 1};
    t.colorViews[gfx::SceneColorAttachmentIndex] = color.CreateView(&layered);
    t.depthView = depth.CreateView(&layered);
    t.clearColor = {0, 0, 0, 0};
    t.clearDepth = 0.f; // reversed Z: infinitely far
    t.depthStore = wgpu::StoreOp::Store;
    t.views[0] = {R.mvGroups, 0.f, 0.f, static_cast<float>(t.size.width), static_cast<float>(t.size.height)};
    t.viewCount = 1;
    t.viewMask = 0b11;
    t.passes = passes;
    t.timestampWrites = zone_writes(kZoneMotion);
    gfx::encode_xr_replay(cmd, frame, gfx::XrCategory::World, t);
    return;
  }
  const wgpu::RenderPassColorAttachment colorAttachment{
      .view = color.CreateView(&layered),
      .loadOp = wgpu::LoadOp::Clear,
      .storeOp = wgpu::StoreOp::Store,
      .clearValue = {0, 0, 0, 0},
  };
  const bool stencil = g_streams[kMotionDepth].format == VK_FORMAT_D24_UNORM_S8_UINT;
  const wgpu::RenderPassDepthStencilAttachment ds{
      .view = depth.CreateView(&layered),
      .depthLoadOp = wgpu::LoadOp::Clear,
      .depthStoreOp = wgpu::StoreOp::Store,
      .depthClearValue = 0.f, // reversed Z: infinitely far
      .stencilLoadOp = stencil ? wgpu::LoadOp::Clear : wgpu::LoadOp::Undefined,
      .stencilStoreOp = stencil ? wgpu::StoreOp::Discard : wgpu::StoreOp::Undefined,
  };
  wgpu::RenderPassMultiview multiview{};
  multiview.viewMask = 0b11;
  const wgpu::RenderPassDescriptor rp{.nextInChain = &multiview,
                                      .label = "XR motion vectors (cleared)",
                                      .colorAttachmentCount = 1,
                                      .colorAttachments = &colorAttachment,
                                      .depthStencilAttachment = &ds};
  cmd.BeginRenderPass(&rp).End();
}

// Early eyes: the motion pass costs as much vertex work as the eyes (4.8 ms
// on four-CPU Brinstar), and with SpaceWarp the eyes' image comes free only
// as the next loop starts, ~17 ms after the tick, with the GPU idle until
// then. So it goes first, on its own, into images of ours (the frame's data
// went with the presubmit), and only a copy waits for the SpaceWarp images.
// Not with a 24-bit depth image, which can't be copied into.
void render_motion_early(gfx::detail::FramePacket& frame,
                         const std::vector<const gfx::detail::RenderPass*>* passes) {
  R.motionCopy = false;
  static const bool presubmit = env_flag("AURORA_XR_PRESUBMIT", true);
  // Staged, the motion pass draws straight into images of ours anyway.
  if (!appsw_on() || !R.early || !presubmit || g_streams[kMotion].staged ||
      g_streams[kMotionDepth].format != VK_FORMAT_D32_SFLOAT ||
      !env_flag("AURORA_XR_APPSW_EARLY", true))
    return;
  auto& device = webgpu::g_device;
  if (!R.motionColor) {
    const wgpu::Extent3D size{g_streams[kMotion].width, g_streams[kMotion].height, 2};
    const wgpu::TextureDescriptor cd{.label = "XR motion vectors",
                                     .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
                                     .size = size,
                                     .format = wgpu::TextureFormat::RGBA16Float};
    R.motionColor = device.CreateTexture(&cd);
    const wgpu::TextureDescriptor dd{.label = "XR motion depth",
                                     .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc,
                                     .size = size,
                                     .format = wgpu::TextureFormat::Depth32Float};
    R.motionDepth = device.CreateTexture(&dd);
  }
  constexpr wgpu::CommandEncoderDescriptor desc{.label = "XR motion vectors (early)"};
  const auto enc = device.CreateCommandEncoder(&desc);
  encode_motion(enc, frame, passes, R.motionColor, R.motionDepth);
  const wgpu::CommandBuffer commands = enc.Finish();
  TraceSection trace{"Submit motion"};
  webgpu::g_queue.Submit(1, &commands);
  R.motionCopy = true;
}

void ensure_motion_depth_flip();
void flip_motion_depth(const wgpu::CommandEncoder& cmd, const wgpu::Texture& dst);

// The SpaceWarp images for this frame: the early pass copied in, or the
// motion pass drawn straight into them.
void render_motion(const wgpu::CommandEncoder& cmd, gfx::detail::FramePacket& frame,
                   const std::vector<const gfx::detail::RenderPass*>* passes) {
  const bool unwaited = R.early && g_deferEyes;
  auto mv = acquire_slot(g_streams[kMotion], std::chrono::milliseconds(10), unwaited);
  auto depth = acquire_slot(g_streams[kMotionDepth], std::chrono::milliseconds(10), unwaited);
  if (!mv || !depth)
    return;
  if (!R.motionCopy) {
    // Standard depth for the runtime (flip_motion_depth); a 24-bit image
    // keeps ours.
    if (g_streams[kMotionDepth].format == VK_FORMAT_D32_SFLOAT) {
      ensure_motion_depth_flip();
      encode_motion(cmd, frame, passes, mv, R.motionDepthFlip);
      flip_motion_depth(cmd, depth);
    } else {
      encode_motion(cmd, frame, passes, mv, depth);
    }
    return;
  }
  const wgpu::Extent3D size{g_streams[kMotion].width, g_streams[kMotion].height, 2};
  const wgpu::TexelCopyTextureInfo colorSrc{.texture = R.motionColor}, colorDst{.texture = mv};
  cmd.CopyTextureToTexture(&colorSrc, &colorDst, &size);
  const wgpu::TexelCopyTextureInfo depthSrc{.texture = R.motionDepth}, depthDst{.texture = depth};
  cmd.CopyTextureToTexture(&depthSrc, &depthDst, &size);
}

// SpaceWarp's depth in the standard convention (Meta's sample: 0 at the
// near plane, 1 infinitely far): for our reversed infinite projection that
// is exactly 1 - ours. The motion pass draws its depth into R.motionDepthFlip,
// and this writes 1 - it into the SpaceWarp image, a layer at a time.
void ensure_motion_depth_flip() {
  if (R.motionDepthFlipPipeline)
    return;
  auto& device = webgpu::g_device;
  static constexpr char kSource[] = R"(
@group(0) @binding(0) var src: texture_depth_2d;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
struct Out { @builtin(frag_depth) depth: f32 };
@fragment fn fs_main(@builtin(position) p: vec4f) -> Out {
  var o: Out;
  o.depth = 1.0 - textureLoad(src, vec2i(p.xy), 0);
  return o;
}
)";
  wgpu::ShaderSourceWGSL wgsl{};
  wgsl.code = kSource;
  const wgpu::ShaderModuleDescriptor smd{.nextInChain = &wgsl, .label = "XR SpaceWarp depth flip"};
  const auto module = device.CreateShaderModule(&smd);
  const wgpu::BindGroupLayoutEntry entry{
      .binding = 0,
      .visibility = wgpu::ShaderStage::Fragment,
      .texture = {.sampleType = wgpu::TextureSampleType::Depth, .viewDimension = wgpu::TextureViewDimension::e2D},
  };
  const wgpu::BindGroupLayoutDescriptor bgld{.entryCount = 1, .entries = &entry};
  R.motionDepthFlipLayout = device.CreateBindGroupLayout(&bgld);
  const wgpu::PipelineLayoutDescriptor pld{.bindGroupLayoutCount = 1, .bindGroupLayouts = &R.motionDepthFlipLayout};
  const auto pipelineLayout = device.CreatePipelineLayout(&pld);
  const wgpu::DepthStencilState ds{
      .format = wgpu::TextureFormat::Depth32Float,
      .depthWriteEnabled = true,
      .depthCompare = wgpu::CompareFunction::Always,
  };
  const wgpu::FragmentState fs{.module = module, .entryPoint = "fs_main", .targetCount = 0};
  const wgpu::RenderPipelineDescriptor rpd{
      .label = "XR SpaceWarp depth flip",
      .layout = pipelineLayout,
      .vertex = {.module = module, .entryPoint = "vs_main"},
      .depthStencil = &ds,
      .fragment = &fs,
  };
  R.motionDepthFlipPipeline = device.CreateRenderPipeline(&rpd);
  const wgpu::TextureDescriptor td{
      .label = "XR SpaceWarp depth (reversed)",
      .usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding,
      .size = {g_streams[kMotionDepth].width, g_streams[kMotionDepth].height, 2},
      .format = wgpu::TextureFormat::Depth32Float};
  R.motionDepthFlip = device.CreateTexture(&td);
  for (uint32_t layer = 0; layer < 2; ++layer) {
    const wgpu::TextureViewDescriptor vd{.dimension = wgpu::TextureViewDimension::e2D,
                                         .baseArrayLayer = layer,
                                         .arrayLayerCount = 1,
                                         .aspect = wgpu::TextureAspect::DepthOnly};
    const wgpu::BindGroupEntry e{.binding = 0, .textureView = R.motionDepthFlip.CreateView(&vd)};
    const wgpu::BindGroupDescriptor bgd{.layout = R.motionDepthFlipLayout, .entryCount = 1, .entries = &e};
    R.motionDepthFlipGroups[layer] = device.CreateBindGroup(&bgd);
  }
}

void flip_motion_depth(const wgpu::CommandEncoder& cmd, const wgpu::Texture& dst) {
  for (uint32_t layer = 0; layer < 2; ++layer) {
    const wgpu::TextureViewDescriptor vd{.dimension = wgpu::TextureViewDimension::e2D,
                                         .baseArrayLayer = layer,
                                         .arrayLayerCount = 1};
    const wgpu::RenderPassDepthStencilAttachment da{
        .view = dst.CreateView(&vd),
        .depthLoadOp = wgpu::LoadOp::Clear,
        .depthStoreOp = wgpu::StoreOp::Store,
        .depthClearValue = 1.f,
    };
    const wgpu::RenderPassDescriptor rp{.label = "XR SpaceWarp depth flip", .depthStencilAttachment = &da};
    auto pass = cmd.BeginRenderPass(&rp);
    pass.SetPipeline(R.motionDepthFlipPipeline);
    pass.SetBindGroup(0, R.motionDepthFlipGroups[layer]);
    pass.Draw(3);
    pass.End();
  }
}

// Hands the SpaceWarp images over with the 3D image of the same frame.
void release_motion(uint64_t tickFrame) {
  if (!B.appSw)
    return;
  release_slot(g_streams[kMotion], nullptr, {}, tickFrame);
  release_slot(g_streams[kMotionDepth], nullptr, {}, tickFrame);
}

bool render_eyes(const wgpu::CommandEncoder& cmd, gfx::detail::FramePacket& frame,
                 const std::vector<std::array<float, 12 + gfx::XrClipFloats>>& worldTransforms,
                 const std::vector<const gfx::detail::RenderPass*>* passes) {
  std::array<XrView, 2> views;
  {
    std::lock_guard lock{g_viewMutex};
    if (!g_viewsValid)
      return false;
    views = g_latestViews;
  }
  const auto layout = gfx::scene_render_target_layout();
  if (!ensure_renderer(layout))
    return false;

  // C = P_eye · V_eye · A · T · V_game⁻¹, with T a draw's extra placement
  // (aurora_xr_world_transform; identity for index 0).
  const Mat4 cameraToWorld = inverse_affine(frame.xrWorldView);
  const Mat4 arena = arena_transform();
  const size_t transforms = std::min<size_t>(worldTransforms.size() + 1, gfx::XrMaxTransforms);
  struct {
    std::array<Mat4, 2> m;
    uint32_t enabled[4];
    float clip[gfx::XrMaxClipPlanes][4];
    float fade[gfx::XrMaxClipPlanes]; // bands (game units)
    std::array<Mat4, 2> prev, prevM;
  } mv[gfx::XrMaxTransforms]{};
  static_assert(sizeof(mv[0]) == kMultiviewEyeSize);
  std::array<Mat4, gfx::XrMaxTransforms> worldToRoom{}, cameraToRoom{};
  // SpaceWarp: the eyes the previous positions are seen with. Meta's
  // XrSpaceWarp sample uses last frame's (the head's motion in the
  // vectors). AURORA_XR_APPSW_HEAD=<0..1> blends toward this frame's (0:
  // the head's motion left out; with our reversed depth that doubled the
  // image on sideways head motion).
  static const float headShare = std::clamp(env_float("AURORA_XR_APPSW_HEAD", 1.f), 0.f, 1.f);
  // The frame these vectors pair with: the last whose eyes went out (before
  // this tick, if this tick's eyes were already encoded once).
  const uint64_t thisTick = frame.xrTickFrame;
  const Renderer3D::PrevEpoch& base =
      g_tweak.epoch && thisTick != 0 && R.epochCur.tick == thisTick ? R.epochBefore : R.epochCur;
  const int perGame = std::max(g_displayPerGameFrame.load(), 1);
  const bool gap = g_tweak.gap && thisTick != 0 && base.tick != 0 && thisTick > base.tick + static_cast<uint64_t>(perGame);
  // AURORA_XR_GAP_LOG=1: what the gap is judged from, about once a second.
  static const bool gapLog = env_flag("AURORA_XR_GAP_LOG", false);
  if (static uint32_t n = 0; gapLog && ++n % 60 == 0)
    Log.info("gap check: tick {} (display frame {}), base {} (cur {}, before {}), per game frame {}: {}", thisTick,
             B.displayFrame, base.tick, R.epochCur.tick, R.epochBefore.tick, perGame, gap ? "gap" : "no gap");
  const bool havePrevViews = base.count != 0;
  // One game tick's worth of the head's motion in the vectors however many
  // ticks back `base` is. SpaceWarp extrapolates a frame's vectors as one
  // frame's motion; after frames that never went out, last-frame-out eyes put
  // several frames of head motion in, and the stage overshot the head and
  // snapped back. AURORA_XR_APPSW_HEAD_GAP=0: the whole gap's motion.
  static const bool headGap = env_flag("AURORA_XR_APPSW_HEAD_GAP", true);
  const uint64_t ticksBack =
      thisTick != 0 && base.tick != 0 && thisTick > base.tick ? (thisTick - base.tick) / static_cast<uint64_t>(perGame) : 1;
  const float share = headGap && ticksBack > 1 ? headShare / static_cast<float>(ticksBack) : headShare;
  for (int eye = 0; eye < 2; ++eye) {
    const Mat4 eyeToClip = mul(projection(views[eye].fov, 0.05f), view_from_pose(views[eye].pose));
    const Mat4 prevEyeToClip =
        havePrevViews ? mul(projection(views[eye].fov, 0.05f),
                            view_from_pose(blend_pose(views[eye].pose, base.views[eye].pose, share)))
                      : eyeToClip;
    for (size_t t = 0; t < transforms; ++t) {
      Mat4 world = cameraToWorld;
      Mat4 place{1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f}; // T
      // Clip plane, game world -> game camera space (the shader clips the
      // camera-space position): plane_cam = plane_world · V_game⁻¹.
      std::array<std::array<float, 4>, gfx::XrMaxClipPlanes> clipCam{};
      for (auto& c : clipCam)
        c = {0.f, 0.f, 0.f, 1.f};
      if (t > 0) {
        const auto& m = worldTransforms[t - 1];
        place = {m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], 0.f, 0.f, 0.f, 1.f};
        world = mul(place, cameraToWorld);
        for (int k = 0; k < gfx::XrMaxClipPlanes; ++k)
          for (int j = 0; j < 4; ++j) {
            clipCam[k][j] = 0.f;
            for (int i = 0; i < 4; ++i)
              clipCam[k][j] += m[12 + k * 4 + i] * cameraToWorld[i * 4 + j];
          }
      }
      struct {
        Mat4 m;
        uint32_t enabled[4];
      } u{mul(mul(eyeToClip, arena), world), {1, 0, 0, 0}};
      static_assert(sizeof(u) == gx::XrEyeUniformSize);
      if (g_multiview) {
        mv[t].m[eye] = u.m;
        // Previous positions in the room, seen with prevEyeToClip. prev: a
        // point fixed in the game world (V_game⁻¹ this frame's on both
        // sides); prevM: one at last frame's game camera space (a draw's own
        // last matrices).
        worldToRoom[t] = mul(arena, place);
        cameraToRoom[t] = mul(arena, world);
        mv[t].prev[eye] =
            mul(prevEyeToClip, mul(t < base.count ? base.worldToRoom[t] : worldToRoom[t], cameraToWorld));
        mv[t].prevM[eye] = mul(prevEyeToClip, t < base.count ? base.cameraToRoom[t] : cameraToRoom[t]);
        mv[t].enabled[0] = 1;
        // The clipped geometry's dissolve, as 1 - opacity (0: solid).
        // The entry's layout: move 3x4, planes, fades, opacity.
        constexpr int kFades = 12 + gfx::XrMaxClipPlanes * 4, kOpacity = kFades + gfx::XrMaxClipPlanes;
        mv[t].enabled[1] = std::bit_cast<uint32_t>(t > 0 ? 1.f - worldTransforms[t - 1][kOpacity] : 0.f);
        for (int k = 0; k < gfx::XrMaxClipPlanes; ++k) {
          std::copy(clipCam[k].begin(), clipCam[k].end(), mv[t].clip[k]);
          mv[t].fade[k] = t > 0 ? worldTransforms[t - 1][kFades + k] : 0.f;
        }
      } else {
        webgpu::g_queue.WriteBuffer(R.eyeUniforms[eye][t], 0, &u, sizeof(u));
      }
    }
  }
  Renderer3D::PrevEpoch newEpoch;
  if (g_multiview) {
    for (size_t t = 0; t < transforms; ++t)
      webgpu::g_queue.WriteBuffer(R.mvUniforms[t], 0, &mv[t], sizeof(mv[t]));
    newEpoch.worldToRoom = worldToRoom;
    newEpoch.cameraToRoom = cameraToRoom;
    newEpoch.views = views;
    newEpoch.count = transforms;
    newEpoch.tick = thisTick;
    if (!g_tweak.epoch)
      R.epochCur = newEpoch;
  }
  // This frame's eyes went out: it is the epoch the next one pairs with.
  const auto commitEpoch = [&] {
    if (!g_multiview || !g_tweak.epoch)
      return;
    if (R.epochCur.tick != thisTick || thisTick == 0)
      R.epochBefore = R.epochCur;
    R.epochCur = newEpoch;
  };

  // Direct path: the framebuffer's format matches the shared 3D image and
  // there is no MSAA to resolve, so both eyes render straight into the shared
  // image in one pass (no private eye images, no compose pass). Otherwise
  // each eye renders privately and a compose pass writes the shared image.
  auto& stereo = g_streams[kStereo];
  const auto stereoFormat = stereo.format == VK_FORMAT_B8G8R8A8_UNORM ? wgpu::TextureFormat::BGRA8Unorm
                                                                      : wgpu::TextureFormat::RGBA8Unorm;
  const bool direct = layout.sampleCount == 1 &&
                      layout.colorAttachments[gfx::SceneColorAttachmentIndex].format == stereoFormat &&
                      env_flag("AURORA_XR_DIRECT", true);
  if (!R.loggedPath) {
    Log.info("3D eyes render {}", direct ? "straight into the shared image" : "privately, then composed");
    R.loggedPath = true;
  }
  if (!direct && (stereoFormat != wgpu::TextureFormat::RGBA8Unorm || g_multiview)) {
    return false; // fallback compose writes side-by-side RGBA8 only (MSAA switched on mid-session)
  }
  if (direct) {
    if (!ensure_direct(layout))
      return false;
    // A frame released `latency` display frames after its tick needn't have
    // its image until one display frame before then: waiting only the usual
    // 10 ms, a stage that records quickly (Battlefield, done 5 ms after the
    // tick) asked before the runtime freed one at latency 4 and dropped every
    // other 3D frame.
    auto wait = std::chrono::milliseconds(10);
    if (const int64_t tickNs = R.renderedTick != 0 ? pace_wait_ns(R.renderedTick) : 0;
        tickNs != 0 && g_present.latency > 1) {
      // A display frame for the GPU at 120 Hz; with SpaceWarp's 60 Hz loop
      // the same 8.3 ms, not a whole loop frame: the image comes free only
      // as the next loop starts (the runtime keeps the shown one), and a
      // deadline right there missed every sixth frame.
      const float hz = g_displayHz > 0.f ? g_displayHz.load() : 120.f;
      const double frameNs = 1e9 / hz;
      const int64_t deadline = tickNs + static_cast<int64_t>(g_present.latency * frameNs - std::min(frameNs, 1e9 / 120.0));
      wait = std::max(wait, std::chrono::milliseconds((deadline - monotonic_ns()) / 1000000));
    }
    // SpaceWarp: a frame whose eyes start over a loop after its tick is
    // dropped (the runtime synthesizes from the last one). Drawn, it held
    // its images into the next frame's loop, that frame's eyes waited, and
    // four-CPU Brinstar settled at 33 3D frames a second for good.
    // AURORA_XR_APPSW_DROP_MS: how late (0: never drop).
    static const float dropMs = env_float("AURORA_XR_APPSW_DROP_MS", 16.7f);
    const int64_t tickNs = appsw_on() && dropMs > 0.f && R.renderedTick != 0 ? pace_wait_ns(R.renderedTick) : 0;
    // AURORA_XR_APPSW_LATE (tweak late): the frame is drawn although late,
    // and the next, if also late, is skipped to give its images back at once
    // (what a late frame held into the next loop used to sink Brinstar to 33
    // frames a second). One decision per tick: the resumed redraw of a frame
    // whose eyes went out early is not judged again.
    bool drop = false;
    if (tickNs != 0) {
      if (R.decidedTick == R.renderedTick) {
        drop = R.decidedDrop;
      } else {
        const bool late = monotonic_ns() - tickNs > static_cast<int64_t>(dropMs * 1e6);
        if (late && g_tweak.late && !R.skipNextLate) {
          R.skipNextLate = true;
          ++g_appSwLate;
        } else if (late) {
          drop = true;
          if (g_tweak.late)
            ++g_appSwPlannedSkip;
          R.skipNextLate = false;
        } else {
          R.skipNextLate = false;
        }
        R.decidedTick = R.renderedTick;
        R.decidedDrop = drop;
      }
    }
    if (drop) {
      R.missedStereo = true;
      ++g_appSwDropped;
      g_skipPresent = !env_flag("AURORA_XR_FIGHT_SCREEN", false);
      return false;
    }
    // After a tick that never went out: no draw pairs with its own previous
    // matrices (gap), for the early motion pass and the later one alike.
    struct ZeroPrev {
      explicit ZeroPrev(bool on) { gfx::set_xr_motion_zero_prev(on); }
      ~ZeroPrev() { gfx::set_xr_motion_zero_prev(false); }
    } zeroPrev(gap);
    if (gap)
      ++g_appSwGapCut;
    render_motion_early(frame, passes);
    if (auto dst = acquire_slot(stereo, wait, R.early && g_deferEyes)) {
      gfx::XrReplayTarget t;
      t.layout = layout;
      t.size = {stereo.width, stereo.height, 1};
      const wgpu::TextureViewDescriptor layered{.dimension = wgpu::TextureViewDimension::e2DArray,
                                                .arrayLayerCount = stereo.layers};
      const auto* viewDesc = g_multiview ? &layered : nullptr;
      const auto dstView = dst.CreateView(viewDesc);
      t.colorViews = R.directExtraViews;
      if (g_xrSamples > 1) {
        // Draw into the MSAA image, resolve into the shared one, and keep
        // none of the samples.
        t.resolveViews[gfx::SceneColorAttachmentIndex] = dstView;
        t.colorStore = wgpu::StoreOp::Discard;
      } else {
        t.colorViews[gfx::SceneColorAttachmentIndex] = dstView;
      }
      t.depthView = R.directDepth.CreateView(viewDesc);
      t.clearColor = {0, 0, 0, 0};
      t.clearDepth = gx::UseReversedZ ? 0.f : 1.f;
      t.depthStore = wgpu::StoreOp::Discard;
      if (g_multiview) {
        // One replay draws both eyes: view mask 0b11 over the image's two layers.
        float vw = static_cast<float>(stereo.width), vh = static_cast<float>(stereo.height);
        if (g_dynres.on) {
          update_dynamic_resolution(false);
          const auto even = [](float v) { return (static_cast<uint32_t>(v + 0.5f) + 1u) & ~1u; };
          const uint32_t rw = std::min(stereo.width, even(static_cast<float>(g_dynres.recWidth) * g_dynres.scale));
          const uint32_t rh = std::min(stereo.height, even(static_cast<float>(g_dynres.recHeight) * g_dynres.scale));
          vw = static_cast<float>(rw);
          vh = static_cast<float>(rh);
          t.renderAreaWidth = rw;
          t.renderAreaHeight = rh;
          R.renderedRect = {static_cast<int32_t>(rw), static_cast<int32_t>(rh)};
        }
        t.views[0] = {R.mvGroups, 0.f, 0.f, vw, vh};
        t.viewCount = 1;
        t.viewMask = 0b11;
      } else {
        const float eyeW = static_cast<float>(stereo.width / 2), eyeH = static_cast<float>(stereo.height);
        t.views[0] = {R.eyeGroups[0], 0.f, 0.f, eyeW, eyeH};
        t.views[1] = {R.eyeGroups[1], eyeW, 0.f, eyeW, eyeH};
        // AURORA_XR_ONE_EYE: replay the left eye only (measuring per-eye cost).
        t.viewCount = env_flag("AURORA_XR_ONE_EYE", false) ? 1 : 2;
      }
      uint32_t draws = 0;
      t.drawCount = &draws;
      t.finish = &draw_coverage;
      t.timestampWrites = zone_writes(kZone3D);
      t.passes = passes;
      gfx::encode_xr_replay(cmd, frame, gfx::XrCategory::World, t);
      R.timing.worldDraws += draws;
      ++R.timing.worldFrames;
      R.renderedViews = views;
      R.renderedStereo = true;
      commitEpoch();
      if (appsw_on())
        render_motion(cmd, frame, passes);
      g_skipPresent = !env_flag("AURORA_XR_FIGHT_SCREEN", false);
    } else {
      // No 3D image this frame: still a fight, so don't present the flat
      // frame nobody sees (the headset keeps showing the last 3D image).
      R.missedStereo = true;
      if (g_dynres.on)
        update_dynamic_resolution(true);
      g_skipPresent = !env_flag("AURORA_XR_FIGHT_SCREEN", false);
    }
  } else {
    for (int eye = 0; eye < 2; ++eye) {
      R.targets[eye].target.timestampWrites = zone_writes(eye == 0 ? kZone3D : kZone3DRight);
      R.targets[eye].target.passes = passes;
      gfx::encode_xr_replay(cmd, frame, gfx::XrCategory::World, R.targets[eye].target);
      R.targets[eye].target.passes = nullptr;
    }
    if (auto dst = acquire_slot(stereo)) {
      compose(cmd, dst, {0, 1}, kZoneCompose3D);
      R.renderedViews = views;
      R.renderedStereo = true;
      commitEpoch();
      g_skipPresent = !env_flag("AURORA_XR_FIGHT_SCREEN", false);
    } else {
      R.missedStereo = true;
      g_skipPresent = !env_flag("AURORA_XR_FIGHT_SCREEN", false);
    }
  }
  return true;
}

// Early eyes (gfx::run_xr_early): the eyes, once the frame's world is
// recorded, from the passes and transforms up to then. True if they were
// encoded into a 3D image; the frame then submits `cmd` at once and
// early_submitted hands the image over.
bool early_eyes(const wgpu::CommandEncoder& cmd, gfx::detail::FramePacket& frame,
                const gfx::detail::XrEarlyEyes& early) {
  if (g_phase != Phase::Imported || !g_sessionRunning || !frame.xrHasWorld || !env_flag("AURORA_XR_3D", true))
    return false;
  R.renderedStereo = false;
  R.missedStereo = false;
  R.renderedRect = {};
  R.renderedTick = frame.xrTickFrame;
  pace_frame(frame.xrRecordStartNs, frame.xrWorldEndNs, 0, 0); // recording goes on
  R.early = true;
  const bool drawn = render_eyes(cmd, frame, early.transforms, &early.passes) && R.renderedStereo;
  R.early = false;
  return drawn;
}

#ifdef __ANDROID__
// AURORA_XR_RENDERDOC_FRAME=N: with RenderDoc's Vulkan layer injected (the
// GPU debug layer settings, docs/quest-xr.md), the Nth early-eyes submit
// (eye and motion passes) is captured; RenderDoc can't find frames in an
// OpenXR app, which never presents. AURORA_XR_RENDERDOC_SPIKE=1: the first
// submit after a GPU frame spike instead (a KO's explosion, say).
RENDERDOC_API_1_6_0* renderdoc_api() {
  static RENDERDOC_API_1_6_0* api = []() -> RENDERDOC_API_1_6_0* {
    void* mod = dlopen("libVkLayer_GLES_RenderDoc.so", RTLD_NOW | RTLD_NOLOAD);
    if (mod == nullptr) {
      Log.warn("RenderDoc capture asked for, but its layer isn't loaded");
      return nullptr;
    }
    auto get = reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(mod, "RENDERDOC_GetAPI"));
    RENDERDOC_API_1_6_0* a = nullptr;
    if (get == nullptr || get(eRENDERDOC_API_Version_1_6_0, reinterpret_cast<void**>(&a)) != 1)
      return nullptr;
    return a;
  }();
  return api;
}

void renderdoc_around_submit(bool before) {
  static const int frame = [] {
    const char* v = std::getenv("AURORA_XR_RENDERDOC_FRAME");
    return v != nullptr ? std::atoi(v) : 0;
  }();
  static const bool onSpike = env_flag("AURORA_XR_RENDERDOC_SPIKE", false);
  static int count = 0;
  static bool capturing = false;
  static bool spikeTaken = false;
  if (frame <= 0 && !onSpike)
    return;
  if (before) {
    ++count;
    if (onSpike ? spikeTaken || !g_spikeSeen.load() : count != frame)
      return;
    spikeTaken = true;
    if (auto* api = renderdoc_api()) {
      api->StartFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(B.vkInstance), nullptr);
      capturing = true;
    }
  } else if (capturing) {
    capturing = false;
    auto* api = renderdoc_api();
    const bool ok = api->EndFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(B.vkInstance), nullptr) == 1;
    char path[512] = {};
    uint32_t len = sizeof path;
    if (ok && api->GetNumCaptures() > 0)
      api->GetCapture(api->GetNumCaptures() - 1, path, &len, nullptr);
    Log.info("RenderDoc capture of early eyes {}: {}", count, ok ? path : "failed");
  }
}
#endif

// Around the early eyes' submit: with their image taken before it was
// waited (g_deferEyes), Dawn translates them with its flush held, and the
// flush goes once the XR thread has waited the image.
void early_around_submit(bool before) {
#ifdef __ANDROID__
  struct Capture {
    bool before;
    Capture(bool b) : before(b) {
      if (before)
        renderdoc_around_submit(true);
    }
    ~Capture() {
      if (!before)
        renderdoc_around_submit(false);
    }
  } capture{before};
#endif
  if (!g_streams[kStereo].unwaited && !g_streams[kMotion].unwaited && !g_streams[kMotionDepth].unwaited)
    return;
  if (before) {
    dawn::native::vulkan::HoldQueueFlush(webgpu::g_device.Get(), true);
    return;
  }
  for (int i : {kStereo, kMotion, kMotionDepth})
    await_slot_waited(g_streams[i]);
  TraceSection trace{"Flush eyes"};
  if (!dawn::native::vulkan::HoldQueueFlush(webgpu::g_device.Get(), false))
    Log.error("Submitting the held eyes failed");
}

void early_submitted(gfx::detail::FramePacket&) {
  for (int i : {kStereo, kMotion, kMotionDepth})
    await_slot_waited(g_streams[i]); // not held after all: never release an image not waited
  release_slot(g_streams[kStereo], &R.renderedViews, R.renderedRect, R.renderedTick);
  release_motion(R.renderedTick);
  ++g_earlyEyes;
}

void render_3d_frame(const wgpu::CommandEncoder& cmd, gfx::detail::FramePacket& frame) {
  if (g_phase != Phase::Imported || !g_sessionRunning || !frame.xrHasWorld ||
      !env_flag("AURORA_XR_3D", true))
    return;
  // The early eyes already went (gfx::run_xr_early), unless world draws came
  // after them.
  if (!frame.xrEyesDone || frame.xrWorldResumed) {
    if (!render_eyes(cmd, frame, frame.xrTransforms, nullptr))
      return;
  }
  if (frame.xrHasHud) {
    R.targets[2].target.timestampWrites = zone_writes(kZoneHud);
    gfx::encode_xr_replay(cmd, frame, gfx::XrCategory::Hud, R.targets[2].target);
    // With SpaceWarp the HUD's image, like the 3D one, comes free only as a
    // loop starts: once frames ran late, waiting up to 10 ms for it held up
    // the render worker every frame, and four-CPU Brinstar never caught up
    // (33 3D frames a second). Missing it keeps the last HUD up a frame.
    const auto hudWait = std::chrono::milliseconds(appsw_on() ? 2 : 10);
    if (auto dst = acquire_slot(g_streams[kHud], hudWait)) {
      compose(cmd, dst, {2}, kZoneComposeHud);
    }
  }
  resolve_timing(cmd);
}

} // namespace

// ---------------------------------------------------------------- public API

bool wanted() noexcept {
#ifdef AURORA_XR_DEFAULT_ON
  static const bool want = env_flag("AURORA_XR", true);
#else
  static const bool want = env_flag("AURORA_XR", false);
#endif
  return want;
}

// Requested whether or not this process presents to XR: Dawn keys its blob
// cache on the device's enabled features, and the Quest launcher panel and
// the immersive game process share one cache. With different feature sets,
// every game pipeline missed what the panel built (about 85 ms each on the
// Quest instead of about 10) and the panel's prune deleted the game's entries.
void add_required_features(const wgpu::Adapter& adapter, std::vector<wgpu::FeatureName>& features) noexcept {
#ifdef __ANDROID__
  const std::array needed{wgpu::FeatureName::SharedTextureMemoryAHardwareBuffer, wgpu::FeatureName::SharedFenceSyncFD};
#else
  const std::array needed{wgpu::FeatureName::SharedTextureMemoryOpaqueFD,
                          wgpu::FeatureName::SharedFenceVkSemaphoreOpaqueFD};
#endif
  for (auto f : needed) {
    if (adapter.HasFeature(f) && std::find(features.begin(), features.end(), f) == features.end())
      features.push_back(f);
  }
  // Bindless batching's prerequisite (not requested yet): eye draws with
  // different textures could join if they named them in Dawn's resource table.
  Log.info("Dawn resource table (bindless textures): {}",
           adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSamplingResourceTable) ? "offered" : "not offered");
  // Eye batches sample through it (gx pipeline.cpp); AURORA_XR_BINDLESS=0
  // turns that off.
  if (env_flag("AURORA_XR_BINDLESS", true) &&
      adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalSamplingResourceTable) &&
      std::find(features.begin(), features.end(), wgpu::FeatureName::ChromiumExperimentalSamplingResourceTable) ==
          features.end())
    features.push_back(wgpu::FeatureName::ChromiumExperimentalSamplingResourceTable);
  // Dawn's API is only safe from several threads with implicit device
  // synchronization, and XR uses it from the render, XR and pipeline-compile
  // threads. Without it a compile on the worker raced the render thread:
  // Tint emitted invalid SPIR-V, Adreno's compiler crashed, a shader module's
  // Tint program was freed under it (2026-10-15, cold caches, bindless on).
  // gx build_pipeline then compiles asynchronously so the device lock isn't
  // held through a 30-700 ms compile. AURORA_XR_DEVICE_SYNC=0 turns it off.
  if (env_flag("AURORA_XR_DEVICE_SYNC", true) &&
      adapter.HasFeature(wgpu::FeatureName::ImplicitDeviceSynchronization) &&
      std::find(features.begin(), features.end(), wgpu::FeatureName::ImplicitDeviceSynchronization) == features.end())
    features.push_back(wgpu::FeatureName::ImplicitDeviceSynchronization);
  // Both eyes in one pass (Dawn fork multiview).
  if (adapter.HasFeature(wgpu::FeatureName::ChromiumExperimentalMultiview) &&
      std::find(features.begin(), features.end(), wgpu::FeatureName::ChromiumExperimentalMultiview) == features.end())
    features.push_back(wgpu::FeatureName::ChromiumExperimentalMultiview);
  // Dynamic resolution's partial render areas (AURORA_XR_DYNRES).
  if (adapter.HasFeature(wgpu::FeatureName::RenderPassRenderArea) &&
      std::find(features.begin(), features.end(), wgpu::FeatureName::RenderPassRenderArea) == features.end())
    features.push_back(wgpu::FeatureName::RenderPassRenderArea);
  // MSAA eye attachments that never leave tile memory (AURORA_XR_MSAA).
  if (adapter.HasFeature(wgpu::FeatureName::TransientAttachments) &&
      std::find(features.begin(), features.end(), wgpu::FeatureName::TransientAttachments) == features.end())
    features.push_back(wgpu::FeatureName::TransientAttachments);
  // XR GPU timing (AURORA_XR_TIMING).
  if (env_flag("AURORA_XR_TIMING", true) && adapter.HasFeature(wgpu::FeatureName::TimestampQuery) &&
      std::find(features.begin(), features.end(), wgpu::FeatureName::TimestampQuery) == features.end())
    features.push_back(wgpu::FeatureName::TimestampQuery);
}

bool active() noexcept { return g_phase == Phase::Imported && g_sessionRunning; }

bool skip_present() noexcept { return g_skipPresent; }

void screen_size(uint32_t contentWidth, uint32_t contentHeight, uint32_t& width, uint32_t& height) noexcept {
  static uint32_t fixedW = 0, fixedH = 0;
  if (fixedW == 0) {
    const auto vp = webgpu::calculate_present_viewport(8192, 8192, contentWidth, contentHeight);
    const float aspect = vp.height > 0.f ? vp.width / vp.height : 4.f / 3.f;
    fixedH = static_cast<uint32_t>(std::clamp(env_float("AURORA_XR_SCREEN_HEIGHT", 1080.f), 240.f, 4096.f));
    fixedW = (static_cast<uint32_t>(static_cast<float>(fixedH) * aspect + 0.5f) + 1u) & ~1u;
  }
  width = fixedW;
  height = fixedH;
  g_contentW = contentWidth;
  g_contentH = contentHeight;
}

wgpu::Texture begin_frame(uint32_t width, uint32_t height) noexcept {
  if (!wanted() || width == 0 || height == 0)
    return {};
  Phase phase = g_phase;
  if (phase == Phase::Idle) {
    if (!B.prepared) {
      Log.error("OpenXR was not set up before the GPU device (prepare_device); presenting to the window");
      g_phase = Phase::Failed;
      return {};
    }
    const auto h = dawn::native::vulkan::GetDeviceVkHandles(webgpu::g_device.Get());
    B.vkInstance = h.instance;
    B.phys = h.physicalDevice;
    B.dev = h.device;
    B.queueFamily = h.queueFamilyIndex;
    B.queueIndex = 1;
    B.queue = dawn::native::vulkan::GetExtraQueue(webgpu::g_device.Get(), 0);
    g_bindlessOk = webgpu::g_device.HasFeature(wgpu::FeatureName::ChromiumExperimentalSamplingResourceTable);
    gfx::set_xr_bindless(env_flag("AURORA_XR_BINDLESS", true) && g_bindlessOk);
    Log.info("Bindless eye batching: {}", gfx::xr_bindless() ? "on" : "off");
    {
      // What the device's descriptor indexing allows, for bindless batching
      // (Dawn's resource table asks 65536 + 1000 update-after-bind images
      // and samplers per stage before it offers itself).
      VkPhysicalDeviceDescriptorIndexingProperties dip{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};
      VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
      p2.pNext = &dip;
      VkPhysicalDeviceDescriptorIndexingFeatures dif{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
      VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
      f2.pNext = &dif;
      // Vulkan 1.1 entry points: through the instance (Android's libvulkan
      // exports only 1.0 at our API level).
      const auto getProps = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
          vkGetInstanceProcAddr(B.vkInstance, "vkGetPhysicalDeviceProperties2"));
      const auto getFeatures = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
          vkGetInstanceProcAddr(B.vkInstance, "vkGetPhysicalDeviceFeatures2"));
      if (getProps != nullptr && getFeatures != nullptr) {
        getProps(B.phys, &p2);
        getFeatures(B.phys, &f2);
      }
      Log.info("Descriptor indexing: non-uniform sampled images {}, samplers in arrays update-after-bind {} / "
               "images {} per stage, {} / {} per set, all pools {}; partially bound {}, variable count {}, "
               "update-after-bind images {}, robust buffer access with it {}",
               dif.shaderSampledImageArrayNonUniformIndexing != 0,
               dip.maxPerStageDescriptorUpdateAfterBindSamplers, dip.maxPerStageDescriptorUpdateAfterBindSampledImages,
               dip.maxDescriptorSetUpdateAfterBindSamplers, dip.maxDescriptorSetUpdateAfterBindSampledImages,
               dip.maxUpdateAfterBindDescriptorsInAllPools, dif.descriptorBindingPartiallyBound != 0,
               dif.descriptorBindingVariableDescriptorCount != 0,
               dif.descriptorBindingSampledImageUpdateAfterBind != 0, dip.robustBufferAccessUpdateAfterBind != 0);
    }
    g_streams[kScreen].name = "screen";
    g_streams[kStereo].name = "3d";
    g_streams[kHud].name = "hud";
    g_streams[kScreen].width = width;
    g_streams[kScreen].height = height;
    // The HUD is big flat text on a small plane: half the screen's
    // resolution by default (AURORA_XR_HUD_SCALE).
    const float hudScale = std::clamp(env_float("AURORA_XR_HUD_SCALE", 0.5f), 0.1f, 2.f);
    g_streams[kHud].width = (static_cast<uint32_t>(static_cast<float>(width) * hudScale + 0.5f) + 1u) & ~1u;
    g_streams[kHud].height = (static_cast<uint32_t>(static_cast<float>(height) * hudScale + 0.5f) + 1u) & ~1u;
    const auto layout = gfx::scene_render_target_layout();
    g_sceneFormat = layout.colorAttachments[gfx::SceneColorAttachmentIndex].format;
    g_sceneSamples = layout.sampleCount;
    // Multiview needs the direct path (single-sample framebuffer).
    g_multiview = webgpu::g_device.HasFeature(wgpu::FeatureName::ChromiumExperimentalMultiview) &&
                  g_sceneSamples == 1 && env_flag("AURORA_XR_MULTIVIEW", true);
    {
      const char* v = std::getenv("AURORA_XR_FIXED_LATENCY");
      g_present.forced = v != nullptr && *v != '\0' ? std::max(0, std::atoi(v)) : -1;
      if (g_present.forced > 0)
        g_present.latency = g_present.forced;
    }
    // AURORA_XR_MSAA=<1|2|4>: samples for the 3D eyes alone (the flat frame
    // keeps its own), resolved into the shared image at the end of the pass.
    // 2 by default on Android: every sample sits in tile memory, so 4x means
    // twice the bins of 2x, and every draw is replayed per bin. Four-CPU
    // Brinstar's eye pass at GPU level 2: 10.7 ms at 4x, 9.4 at 2x, 7.6 at 1x
    // (2x needs the Dawn fork's sample count 2, 051c687).
    {
#ifdef __ANDROID__
      constexpr float kDefaultSamples = 2.f;
#else
      constexpr float kDefaultSamples = 1.f;
#endif
      const int n = static_cast<int>(env_float("AURORA_XR_MSAA", kDefaultSamples));
      g_xrSamples = g_multiview && (n == 2 || n == 4) ? static_cast<uint32_t>(n) : 1u;
    }
    Log.info("3D eyes: {}, {}x MSAA", g_multiview ? "multiview, both in one pass" : "side by side, one pass each",
             g_xrSamples);
    g_phase = Phase::Starting;
    g_thread = std::thread(thread_main);
    return {};
  }
  if (phase == Phase::SlotsReady) {
    if (!wrap_swapchains_into_dawn()) {
      Log.error("Virtual screen disabled; presenting to the window instead");
      g_phase = Phase::Failed;
      return {};
    }
    gfx::set_xr_frame_hook(&render_3d);
    // The eyes go to the GPU as soon as the frame's world is recorded
    // (xr_replay.hpp), multiview only; on by default on Android, where it's
    // measured (AURORA_XR_EARLY_EYES=0|1 overrides). Four-CPU Mute City: GPU
    // done about 2.5 ms sooner after the tick.
#ifdef __ANDROID__
    constexpr bool kEarlyEyesDefault = true;
#else
    constexpr bool kEarlyEyesDefault = false;
#endif
    gfx::set_xr_early_hooks(&early_eyes, &early_submitted, &early_around_submit);
    const bool earlyEyes = g_multiview && env_flag("AURORA_XR_EARLY_EYES", kEarlyEyesDefault);
    gfx::set_xr_early_eyes(earlyEyes);
    g_deferEyes = earlyEyes && env_flag("AURORA_XR_DEFER_EYES", kEarlyEyesDefault);
    g_phase = Phase::Imported;
    phase = Phase::Imported;
  }
  if (phase != Phase::Imported)
    return {};
  if (g_multiview) {
    // World draws recorded from now on resolve their multiview twins.
    auto mvLayout = gfx::scene_render_target_layout();
    mvLayout.viewCount = 2;
    mvLayout.sampleCount = g_xrSamples;
    gfx::detail::finalize_render_target_layout(mvLayout);
    gfx::set_xr_multiview_layout(mvLayout);
    if (B.appSw) {
      // ... and, the opaque ones, their SpaceWarp motion twins.
      gfx::RenderTargetLayout motion{};
      motion.colorAttachmentCount = 1;
      motion.colorAttachments[gfx::SceneColorAttachmentIndex] = {
          .semantic = gfx::ColorAttachmentSemantic::SceneColor,
          .format = wgpu::TextureFormat::RGBA16Float,
          .width = g_streams[kMotion].width,
          .height = g_streams[kMotion].height,
      };
      motion.depthStencilFormat = g_streams[kMotionDepth].format == VK_FORMAT_D24_UNORM_S8_UINT
                                      ? wgpu::TextureFormat::Depth24PlusStencil8
                                      : wgpu::TextureFormat::Depth32Float;
      motion.viewCount = 2;
      motion.xrMotion = 1;
      gfx::detail::finalize_render_target_layout(motion);
      gfx::set_xr_motion_layout(motion);
    }
  }
  auto& screen = g_streams[kScreen];
  if (width != screen.width || height != screen.height) {
    static bool warned = false;
    if (!warned) {
      Log.warn("Frame size changed to {}x{}; the virtual screen stays {}x{} and the window takes over", width, height,
               screen.width, screen.height);
      warned = true;
    }
    return {};
  }
  auto texture = acquire_slot(screen);
  if (!texture) {
    // No image ready (the XR thread is a frame behind): drop this frame's
    // flat present rather than falling back to the window.
    g_skipPresent = true;
  }
  return texture;
}

void end_frame() noexcept {
  g_skipPresent = false;
  map_timing();
  release_slot(g_streams[kScreen], nullptr);
  release_slot(g_streams[kStereo], R.renderedStereo ? &R.renderedViews : nullptr, R.renderedRect, R.renderedTick);
  release_motion(R.renderedTick);
  R.renderedStereo = false;
  release_slot(g_streams[kHud], nullptr, {}, R.renderedTick);
}

bool prepare_device() noexcept {
  if (!wanted() || B.prepared)
    return B.prepared;
  if (!create_instance()) {
    Log.error("OpenXR unavailable; presenting to the window instead");
    g_phase = Phase::Failed;
    return false;
  }
  auto getReqs = xr_proc<PFN_xrGetVulkanGraphicsRequirements2KHR>("xrGetVulkanGraphicsRequirements2KHR");
  B.createVulkanInstance = xr_proc<PFN_xrCreateVulkanInstanceKHR>("xrCreateVulkanInstanceKHR");
  B.createVulkanDevice = xr_proc<PFN_xrCreateVulkanDeviceKHR>("xrCreateVulkanDeviceKHR");
  B.getVulkanGraphicsDevice = xr_proc<PFN_xrGetVulkanGraphicsDevice2KHR>("xrGetVulkanGraphicsDevice2KHR");
  if (!getReqs || !B.createVulkanInstance || !B.createVulkanDevice || !B.getVulkanGraphicsDevice) {
    Log.error("XR_KHR_vulkan_enable2 entry points missing; presenting to the window instead");
    g_phase = Phase::Failed;
    return false;
  }
  // Required before the runtime creates anything Vulkan for us.
  XrGraphicsRequirementsVulkan2KHR reqs{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
  if (XR_FAILED(getReqs(B.instance, B.systemId, &reqs))) {
    Log.error("xrGetVulkanGraphicsRequirements2KHR failed; presenting to the window instead");
    g_phase = Phase::Failed;
    return false;
  }
  dawn::native::vulkan::ExternalVulkanHooks hooks;
  hooks.createInstance = &hook_create_instance;
  hooks.getPhysicalDevice = &hook_physical_device;
  hooks.createDevice = &hook_create_device;
  hooks.extraQueueCount = 1; // the runtime's queue (graphics binding)
  dawn::native::vulkan::SetExternalVulkanHooks(&hooks);
  B.prepared = true;
  return true;
}

void release_instance() noexcept {
  if (B.instance) {
    xrDestroyInstance(B.instance);
    B.instance = XR_NULL_HANDLE;
  }
  dawn::native::vulkan::SetExternalVulkanHooks(nullptr);
}

void shutdown() noexcept {
  g_stop = true;
  if (g_thread.joinable() && g_thread.get_id() != std::this_thread::get_id())
    g_thread.join();
}

} // namespace aurora::xr

extern "C" bool aurora_xr_get_pad(PADStatus* out) {
  std::lock_guard lock{aurora::xr::g_padMutex};
  if (!aurora::xr::g_padValid)
    return false;
  *out = aurora::xr::g_pad;
  return true;
}

extern "C" bool aurora_xr_screen_pointer(float* x, float* y, bool* pressed) {
  std::lock_guard lock{aurora::xr::g_padMutex};
  const auto& p = aurora::xr::g_screenPointer;
  if (!p.valid)
    return false;
  *x = p.x;
  *y = p.y;
  *pressed = p.pressed;
  return true;
}

extern "C" bool aurora_xr_screen_back(void) {
  std::lock_guard lock{aurora::xr::g_padMutex};
  return aurora::xr::g_screenBack;
}

extern "C" void aurora_xr_set_paused(bool paused) { aurora::xr::g_fightPaused = paused; }

extern "C" void aurora_xr_set_placing(int mode) { aurora::xr::g_placing = mode; }

extern "C" void aurora_xr_reset_stage(void) {
  using namespace aurora::xr;
  std::lock_guard lock{g_arenaMutex};
  init_arena();
  g_stageArenas.erase(g_stage);
  g_arena = g_adoptedSet ? g_adoptedArena : g_defaultArena;
  g_arena.scale *= g_stageScale;
}

extern "C" void aurora_xr_adopt_stage(void) {
  using namespace aurora::xr;
  std::lock_guard lock{g_arenaMutex};
  init_arena();
  g_adoptedArena = g_arena;
  g_adoptedArena.scale /= g_stageScale;
  g_adoptedSet = true;
  g_stageArenas.clear();
  g_spot.createWanted = true; // kept in the room (the stage spot's anchor)
  Log.info("Arena adopted as every stage's start: {:.2f},{:.2f},{:.2f}, yaw {:.0f} deg, scale {:.5f}",
           g_adoptedArena.pos.x, g_adoptedArena.pos.y, g_adoptedArena.pos.z, g_adoptedArena.yaw * 57.2958f,
           g_adoptedArena.scale);
}

extern "C" void aurora_xr_set_board_proxy(const AuroraXrBoardProxy* proxy) {
  using namespace aurora::xr;
  if (!proxy || proxy->footHalfX <= 0.f || proxy->footHalfZ <= 0.f || proxy->depth <= 0.f)
    return;
  std::lock_guard lock{g_boardProxyMutex};
  BP.proxy = *proxy;
  BP.haveProxy = true;
}

extern "C" bool aurora_xr_board_placing(void) { return aurora::xr::g_boardPlacing; }

extern "C" float aurora_xr_stand_height(void) {
  using namespace aurora::xr;
  std::lock_guard lock{g_arenaMutex};
  const float s = g_arena.scale * g_stageZoom;
  if (!g_boardPlaced || s <= 0.f)
    return 0.f;
  return std::max(g_arena.pos.y - g_boardSurfaceY, 0.f) / s;
}

extern "C" bool aurora_xr_stage_adopted(void) {
  std::lock_guard lock{aurora::xr::g_arenaMutex};
  return aurora::xr::g_adoptedSet;
}

extern "C" void aurora_xr_set_placing_clip(int which, int width, int height, const unsigned char* rgba, int frames,
                                           int cols, int frameW, int frameH, float fps, int x, int y, int w, int h) {
  using namespace aurora::xr;
  if (which < 0 || which >= static_cast<int>(g_placingClips.size()) || width <= 0 || height <= 0 || !rgba ||
      frames <= 0 || cols <= 0 || frameW <= 0 || frameH <= 0 || cols * frameW > width ||
      (frames + cols - 1) / cols * frameH > height)
    return;
  std::lock_guard lock{g_placingMutex};
  PlacingClip& clip = g_placingClips[which];
  if (clip.atlas.swapchain)
    return; // already shown; clips are given once
  clip.atlas.width = static_cast<uint32_t>(width);
  clip.atlas.height = static_cast<uint32_t>(height);
  clip.atlas.rgba.assign(rgba, rgba + static_cast<size_t>(width) * height * 4);
  clip.frames = frames;
  clip.cols = cols;
  clip.frameW = frameW;
  clip.frameH = frameH;
  clip.fps = fps > 0.f ? fps : 12.f;
  clip.x = x;
  clip.y = y;
  clip.w = w;
  clip.h = h;
}

extern "C" void aurora_xr_set_placing_image(int which, int width, int height, const unsigned char* rgba) {
  using namespace aurora::xr;
  if (which < 0 || which >= static_cast<int>(g_placingImages.size()) || width <= 0 || height <= 0 || !rgba)
    return;
  std::lock_guard lock{g_placingMutex};
  PlacingImage& img = g_placingImages[which];
  if (img.swapchain)
    return; // already shown; pictures are given once
  img.width = static_cast<uint32_t>(width);
  img.height = static_cast<uint32_t>(height);
  img.rgba.assign(rgba, rgba + static_cast<size_t>(width) * height * 4);
}

extern "C" void aurora_xr_set_passthrough(bool on) { aurora::xr::g_passthroughWanted = on; }

extern "C" bool aurora_xr_active(void) { return aurora::xr::active(); }

extern "C" void aurora_xr_heavy_effect(void) { aurora::xr::g_heavyHint = true; }

extern "C" void aurora_xr_set_stage_top(float y) {
  std::lock_guard lock{aurora::xr::g_arenaMutex};
  aurora::xr::g_stageTop = y;
}

extern "C" void aurora_xr_set_stage_footprint(float xmin, float xmax, float zmin, float zmax, float ybottom) {
  using namespace aurora::xr;
  {
    std::lock_guard lock{g_arenaMutex};
    g_footprint = {xmin < xmax && zmin < zmax, xmin, xmax, zmin, zmax, ybottom};
  }
  // A stage ready for its fight: measured now, in case no hold comes.
  surface_request(1000, false);
}

extern "C" void aurora_xr_dev_audio_defaults(const AuroraXrDevAudio* audio) {
  using namespace aurora::xr;
  std::lock_guard lock{g_devAudio.mutex};
  g_devAudio.defaults = *audio;
  if (!g_devAudio.have)
    g_devAudio.value = *audio;
  g_devAudio.have = true;
}

extern "C" bool aurora_xr_dev_audio(AuroraXrDevAudio* audio) {
  using namespace aurora::xr;
  if (!dev_tools_wanted())
    return false;
  std::lock_guard lock{g_devAudio.mutex};
  if (!g_devAudio.have)
    return false;
  *audio = g_devAudio.value;
  return true;
}

namespace aurora::xr {
int add_dev_option(DevOption o) {
  if (!dev_tools_wanted())
    return -1;
  std::lock_guard lock{g_devOptions.mutex};
  const int option = static_cast<int>(g_devOptions.options.size());
  if (option >= AURORA_XR_DEV_OPTIONS)
    return -1;
  g_devOptionValue[option].store(o.initial, std::memory_order_relaxed);
  g_devOptions.options.push_back(std::move(o));
  g_devOptionsAdded.store(true, std::memory_order_relaxed);
  return option;
}
} // namespace aurora::xr

extern "C" int aurora_xr_dev_option_add(const char* name, int count, const char* const choices[], int initial) {
  using namespace aurora::xr;
  if (count < 1)
    return -1;
  DevOption o;
  o.name = name;
  for (int i = 0; i < std::min(count, static_cast<int>(AURORA_XR_DEV_CHOICES)); ++i)
    o.choices.emplace_back(choices[i]);
  o.initial = static_cast<float>(std::clamp(initial, 0, static_cast<int>(o.choices.size()) - 1));
  return add_dev_option(std::move(o));
}

extern "C" int aurora_xr_dev_slider_add(const char* name, float min, float max, float step, float initial) {
  using namespace aurora::xr;
  if (!(max > min))
    return -1;
  DevOption o;
  o.name = name;
  o.min = min;
  o.max = max;
  o.step = std::max(step, 0.f);
  o.initial = std::clamp(initial, min, max);
  return add_dev_option(std::move(o));
}

extern "C" int aurora_xr_dev_option(int option) {
  if (option < 0 || option >= AURORA_XR_DEV_OPTIONS)
    return -1;
  return static_cast<int>(aurora::xr::g_devOptionValue[option].load(std::memory_order_relaxed));
}

extern "C" float aurora_xr_dev_slider(int option, float fallback) {
  if (option < 0 || option >= AURORA_XR_DEV_OPTIONS)
    return fallback;
  return aurora::xr::g_devOptionValue[option].load(std::memory_order_relaxed);
}

extern "C" void aurora_xr_set_grab_box(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax) {
  using namespace aurora::xr;
  std::lock_guard lock{g_arenaMutex};
  // From the stage's game coordinates to the box's, around the arena center.
  const XrVector3f c{g_arenaCenter[0], g_arenaCenter[1], g_arenaCenter[2]};
  g_grabBoxMin = XrVector3f{xmin, ymin, zmin} - c;
  g_grabBoxMax = XrVector3f{xmax, ymax, zmax} - c;
}

extern "C" void aurora_xr_dev_mix_stage(int stage, int count, const char* const names[],
                                        const AuroraXrMixChannel defaults[]) {
  using namespace aurora::xr;
  if (!dev_tools_wanted())
    return;
  count = std::clamp(count, 0, static_cast<int>(AURORA_XR_MIX_CHANNELS));
  std::lock_guard lock{g_devMix.mutex};
  g_devMix.stage = stage;
  const auto [it, fresh] = g_devMix.stages.try_emplace(stage);
  if (!fresh)
    return; // kept from the stage's last fight
  DevMixStage& st = it->second;
  st.count = count;
  for (int c = 0; c < count; ++c) {
    st.names[c] = names[c];
    st.defaults[c] = st.value[c] = defaults[c];
  }
}

extern "C" bool aurora_xr_dev_mix_listening(void) {
  return aurora::xr::g_devMixListening.load(std::memory_order_relaxed);
}

extern "C" bool aurora_xr_dev_mix(int stage, int channel, AuroraXrMixChannel* out) {
  using namespace aurora::xr;
  if (!dev_tools_wanted() || channel < 0 || channel >= AURORA_XR_MIX_CHANNELS)
    return false;
  std::lock_guard lock{g_devMix.mutex};
  const auto it = g_devMix.stages.find(stage);
  if (it == g_devMix.stages.end() || channel >= it->second.count)
    return false;
  *out = it->second.value[channel];
  return true;
}

extern "C" void aurora_xr_set_room_grade(float r, float g, float b, float saturation) {
  using namespace aurora::xr;
  std::lock_guard lock{g_grade.mutex};
  g_grade.game = {std::max(r, 0.f), std::max(g, 0.f), std::max(b, 0.f), std::max(saturation, 0.f)};
  // The grade panel's, while it has one for this stage.
  const auto it = g_grade.panel.find(g_grade.stage);
  set_grade_locked(it != g_grade.panel.end() ? it->second : g_grade.game);
}

extern "C" void aurora_xr_set_stage(int stage, float x, float y, float z, float scale) {
  using namespace aurora::xr;
  surface_request(0, true); // another stage: what was under the last one isn't under it
  {
    std::lock_guard lock{g_grade.mutex};
    g_grade.stage = stage;
  }
  aurora_xr_set_room_grade(1.f, 1.f, 1.f, 1.f);
  std::lock_guard lock{g_arenaMutex};
  g_stage = stage;
  g_arenaCenter = {x, y, z};
  g_stageScale = scale > 0.f ? scale : 1.f;
  g_stageZoom = 1.f;
  g_hudOnTop = false;
  g_footprint = {};
  g_grabBoxMin = kGrabBoxMin;
  g_grabBoxMax = kGrabBoxMax;
  init_arena();
  if (const auto it = g_stageArenas.find(stage); it != g_stageArenas.end()) {
    g_arena = it->second;
  } else {
    // Not placed yet this session: in front of where the head is and looks,
    // like the placing cards, not of the tracking origin (which is wherever
    // the player last recentered).
    if (g_adoptedSet) {
      g_arena = g_adoptedArena;
      g_arena.scale *= g_stageScale;
    } else {
      g_arena = g_defaultArena;
      g_arena.scale *= g_stageScale;
      g_arenaToHead = true;
    }
  }
  static const bool atHead = env_flag("AURORA_XR_ARENA_AT_HEAD", false);
  if (atHead)
    g_arenaToHead = true;
}

extern "C" void aurora_xr_set_stage_center(float x, float y, float z) {
  using namespace aurora::xr;
  std::lock_guard lock{g_arenaMutex};
  g_arenaCenter = {x, y, z};
}

extern "C" void aurora_xr_set_hud_on_top(bool on) { aurora::xr::g_hudOnTop = on; }

extern "C" bool aurora_xr_audio_listener(float m[3][4]) {
  using namespace aurora::xr;
  if (!active())
    return false;
  XrPosef head;
  {
    std::lock_guard lock{g_viewMutex};
    if (!g_viewsValid)
      return false;
    head.orientation = B.headOrientation;
    head.position = B.head;
  }
  // Head <- room <- game: the inverse of the head's pose, after the arena's
  // placement (what the eyes see, A in render_eyes, without the eye offsets).
  const Mat4 t = mul(view_from_pose(head), arena_transform());
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c)
      m[r][c] = t[r * 4 + c];
  return true;
}

extern "C" bool aurora_xr_audio_screen(float center[3], float right[3], float* width) {
  using namespace aurora::xr;
  if (!active())
    return false;
  XrPosef head, screen;
  {
    std::lock_guard lock{g_viewMutex};
    if (!g_viewsValid || !g_audioScreen.shown)
      return false;
    head.orientation = B.headOrientation;
    head.position = B.head;
    screen = g_audioScreen.pose;
    *width = g_audioScreen.width;
  }
  const Mat4 v = view_from_pose(head);
  const XrVector3f c = screen.position, x = qrot(screen.orientation, {1.f, 0.f, 0.f});
  for (int r = 0; r < 3; ++r) {
    center[r] = v[r * 4] * c.x + v[r * 4 + 1] * c.y + v[r * 4 + 2] * c.z + v[r * 4 + 3];
    right[r] = v[r * 4] * x.x + v[r * 4 + 1] * x.y + v[r * 4 + 2] * x.z;
  }
  return true;
}

extern "C" void aurora_xr_set_stage_zoom(float zoom) {
  using namespace aurora::xr;
  std::lock_guard lock{g_arenaMutex};
  g_stageZoom = zoom > 0.f ? zoom : 1.f;
}

extern "C" bool aurora_xr_pace(void) {
  using namespace aurora::xr;
  if (g_displayPerGameFrame <= 0 || !g_sessionRunning)
    return false;
  std::unique_lock lock{g_paceMutex};
  const uint64_t tick = g_paceTick;
  // Bounded: a paused or stopped session falls back to the game's own timer.
  const bool ticked = g_paceCv.wait_for(lock, std::chrono::milliseconds(50), [&] {
    return g_paceTick != tick || g_displayPerGameFrame <= 0 || !g_sessionRunning;
  }) && g_paceTick != tick;
  if (ticked)
    pace_wake(g_lastTickFrame);
  // The frame the game records next carries this tick (fixed-latency
  // presentation; 0 without one).
  aurora::gfx::set_xr_game_frame_tick(ticked ? g_lastTickFrame.load() : 0);
  return ticked;
}

extern "C" uint64_t aurora_xr_pace_count(void) {
  using namespace aurora::xr;
  if (g_displayPerGameFrame <= 0 || !g_sessionRunning) {
    // Nothing to pace to: the frame recorded next carries no tick, rather
    // than whichever one paced the last.
    aurora::gfx::set_xr_game_frame_tick(0);
    return 0;
  }
  std::lock_guard lock{g_paceMutex};
  return g_paceTick;
}

extern "C" bool aurora_xr_pace_until(uint64_t count) {
  using namespace aurora::xr;
  std::unique_lock lock{g_paceMutex};
  const bool waits = g_paceTick < count;
  // Bounded like aurora_xr_pace: a session that stops ticking hands the
  // caller back its own timer.
  const bool due = g_paceCv.wait_for(lock, std::chrono::milliseconds(100), [&] {
    return g_paceTick >= count || g_displayPerGameFrame <= 0 || !g_sessionRunning;
  }) && g_paceTick >= count && g_displayPerGameFrame > 0 && g_sessionRunning;
  const uint64_t frame = g_lastTickFrame.load();
  lock.unlock();
  if (due && waits)
    pace_wake(frame);
  aurora::gfx::set_xr_game_frame_tick(due ? frame : 0);
  return due;
}
