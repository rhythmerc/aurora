#include "xr.hpp"

#include <aurora/xr.h>

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "../gfx/frame.hpp"
#include "../gfx/recording.hpp"
#include "../gfx/xr_replay.hpp"
#include "../gx/gx.hpp"
#include "../dolphin/pad/fight.hpp"

#include <aurora/gfx.hpp>
#include <dolphin/pad.h>

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
#include <roomtone/record.hpp>
#include <roomtone/roomtone.hpp>
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
// The head's last few poses (publish_views; XR thread): a recenter's report
// is read against where the head was just before it (spot_recentered).
struct HeadSample {
  XrVector3f pos{};
  XrQuaternionf orientation{0.f, 0.f, 0.f, 1.f};
};
std::array<HeadSample, 12> g_headHistory{};
size_t g_headHistoryN = 0;
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
// And the match itself stopped (aurora_xr_set_match_stopped): a pause, not
// Slippi's overlay.
std::atomic<bool> g_matchStopped{false};
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

// Set by the game (aurora_xr_set_passthrough): mixed reality shows the room,
// full VR doesn't. The XR thread pauses or starts passthrough to match.
std::atomic<bool> g_passthroughWanted{true};

// The game's rumble motor (aurora_xr_set_rumble); the XR thread drives the
// controllers' haptics to match.
std::atomic<bool> g_rumbleWanted{false};

// Controller input, written by the XR thread, read by the game thread: the
// controllers' own inputs, made a GameCube pad as the game reads it
// (aurora_xr_get_pad), through the fight mapping in a fight.
std::mutex g_padMutex;
PADFightRaw g_touchRaw{};
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
  bool hasEnvDepthExt = false, envDepth = false; // XR_META_environment_depth: offered and wanted; the system can
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
  XrAction btnStart = XR_NULL_HANDLE;
  std::array<XrAction, 2> gripBtn{}, stickClick{}; // left, right
  XrAction trigL = XR_NULL_HANDLE, trigR = XR_NULL_HANDLE;
  // Pointing and grabbing the arena while a fight is paused.
  std::array<XrAction, 2> aim{}, grab{};
  std::array<XrSpace, 2> aimSpace{};
  // The game's rumble, on both controllers' haptics (aurora_xr_set_rumble).
  XrAction rumble = XR_NULL_HANDLE;
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
#ifdef XR_META_environment_depth
    B.hasEnvDepthExt |= !std::strcmp(p.extensionName, XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME);
#endif
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
#ifdef XR_META_environment_depth
  // The room capture's depth (AURORA_XR_ROOM_DEPTH=0 leaves it out: room_depth_frame).
  B.hasEnvDepthExt &= env_flag("AURORA_XR_ROOM_DEPTH", true);
  if (B.hasEnvDepthExt)
    exts.push_back(XR_META_ENVIRONMENT_DEPTH_EXTENSION_NAME);
#endif
  B.hasLocalFloorExt &= env_flag("AURORA_XR_LOCAL_FLOOR", true);
  if (B.hasLocalFloorExt)
    exts.push_back(XR_EXT_LOCAL_FLOOR_EXTENSION_NAME);
  // The room's grade by colour LUT; without it, brightness, contrast and
  // saturation (AURORA_XR_GRADE_LUT=0 forces those, AURORA_XR_GRADE=0 none).
  B.hasColorLutExt &= B.hasPassthroughExt && grade_wanted() &&
                      env_float("AURORA_XR_GRADE_LUT", 16.f) >= 2.f;
  if (B.hasColorLutExt)
    exts.push_back(XR_META_PASSTHROUGH_COLOR_LUT_EXTENSION_NAME);
  // The stage spot kept in the room across launches (offered to apps
  // declaring USE_ANCHOR_API): off, AURORA_XR_ANCHORS=1 turns it on. Off, a
  // placement lasts the launch, and a recenter's own report keeps the stage
  // in its place in the room (spot_recentered).
  const bool anchorExts = B.hasSpatialEntityExt && B.hasSpacePersistenceExt && B.hasSpaceDiscoveryExt &&
                          env_flag("AURORA_XR_ANCHORS", false);
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
  std::strncpy(ci.applicationInfo.applicationName, g_config.appName != nullptr ? g_config.appName : "aurora",
               XR_MAX_APPLICATION_NAME_SIZE - 1);
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
#ifdef XR_META_environment_depth
    XrSystemEnvironmentDepthPropertiesMETA depth{XR_TYPE_SYSTEM_ENVIRONMENT_DEPTH_PROPERTIES_META};
    if (B.hasEnvDepthExt) {
      depth.next = sp.next;
      sp.next = &depth;
    }
#endif
    if (B.hasColorLutExt) {
      lut.next = sp.next;
      sp.next = &lut;
    }
    if (XR_SUCCEEDED(xrGetSystemProperties(B.instance, B.systemId, &sp))) {
      B.maxLayers = std::max<uint32_t>(sp.graphicsProperties.maxLayerCount, 1);
      B.envRaycast = B.hasEnvRaycastExt && raycast.supportsEnvironmentRaycast;
      B.maxLutResolution = B.hasColorLutExt ? lut.maxColorLutResolution : 0;
#ifdef XR_META_environment_depth
      B.envDepth = B.hasEnvDepthExt && depth.supportsEnvironmentDepth;
      if (B.hasEnvDepthExt)
        Log.info("Environment depth: {}, hand removal {}", depth.supportsEnvironmentDepth ? "supported" : "unsupported",
                 depth.supportsHandRemoval ? "supported" : "unsupported");
#endif
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

// Fixed foveated rendering (AuroraConfig::xrFoveation, AURORA_XR_FOVEATION
// 0-3): the eyes' pass gets a fragment density map of ours
// (VK_EXT_fragment_density_map, through the Dawn fork's
// RenderPassMultiview::vkFragmentDensityMapView), full density around each
// lens' centre and less towards the edges, where the tiles then hold fewer,
// bigger fragments. A map is the whole image's size in 32-pixel density
// texels, and with dynamic resolution the eye is drawn in the image's corner,
// so there is one map per eye size: the XR thread makes them on its queue
// (density_frame), the eye sizes dynamic resolution can pick ahead of the
// fights and any other size the render worker asks for; until a size's map
// is ready the nearest one stands in. Level 0 leaves the pass and its
// pipelines as they were.
struct DensityMaps {
  int level = 0;              // for the session, decided with the device (hook_create_device)
  std::atomic_bool on{false}; // the eyes' pass has a map (multiview, the extension on: begin_frame)
  VkExtent2D texel{32, 32};   // the density texel size the maps are made for
  struct Map {
    uint32_t w = 0, h = 0;    // the eye size (render area) it's for
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    bool ready = false;       // filled, and its upload's fence signalled
  };
  std::mutex mutex;           // maps, asked
  std::deque<Map> maps;       // never freed before shutdown (a few kB each)
  std::vector<std::pair<uint32_t, uint32_t>> asked; // sizes the render worker found missing
  bool primed = false;        // dynamic resolution's sizes asked for (XR thread)
  bool failed = false;        // a map couldn't be made: no more tries (XR thread)
  // The upload in flight (XR thread): its staging buffer, and the maps it fills.
  Bridge::Upload pending{};
  std::vector<size_t> pendingMaps;
} g_density;

// The instance the runtime made for Dawn (hook_create_instance), for the
// device hook's queries.
VkInstance g_hookInstance = VK_NULL_HANDLE;

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
  if (vr == VK_SUCCESS)
    g_hookInstance = *out;
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

// Foveation's needs: the extension, its feature, and RG8 density maps.
// (Vulkan 1.1 queries through the instance: Android's libvulkan exports 1.0.)
bool density_supported(PFN_vkGetInstanceProcAddr getProcAddr, VkPhysicalDevice phys) {
  const auto features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
      getProcAddr(g_hookInstance, "vkGetPhysicalDeviceFeatures2"));
  const auto properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
      getProcAddr(g_hookInstance, "vkGetPhysicalDeviceProperties2"));
  if (!features2 || !properties2) {
    Log.info("Foveation: no vkGetPhysicalDeviceFeatures2/Properties2");
    return false;
  }
  uint32_t n = 0;
  vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, nullptr);
  std::vector<VkExtensionProperties> exts(n);
  vkEnumerateDeviceExtensionProperties(phys, nullptr, &n, exts.data());
  if (std::none_of(exts.begin(), exts.end(), [](const VkExtensionProperties& e) {
        return !std::strcmp(e.extensionName, VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME);
      })) {
    Log.info("Foveation: no {}", VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME);
    return false;
  }
  VkPhysicalDeviceFragmentDensityMapFeaturesEXT ff{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT};
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  f2.pNext = &ff;
  features2(phys, &f2);
  VkPhysicalDeviceFragmentDensityMapPropertiesEXT fp{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_PROPERTIES_EXT};
  VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  p2.pNext = &fp;
  properties2(phys, &p2);
  VkFormatProperties format{};
  vkGetPhysicalDeviceFormatProperties(phys, VK_FORMAT_R8G8_UNORM, &format);
  const bool formatOk = (format.optimalTilingFeatures & VK_FORMAT_FEATURE_FRAGMENT_DENSITY_MAP_BIT_EXT) != 0;
  // Dawn's attachments and the swapchain's images aren't subsampled.
  Log.info("Foveation: fragment density map {}, with plain attachments {}, texels {}x{} to {}x{}, extra "
           "invocations {}, RG8 maps {}",
           ff.fragmentDensityMap ? "available" : "unavailable",
           ff.fragmentDensityMapNonSubsampledImages ? "yes" : "no", fp.minFragmentDensityTexelSize.width,
           fp.minFragmentDensityTexelSize.height, fp.maxFragmentDensityTexelSize.width,
           fp.maxFragmentDensityTexelSize.height, fp.fragmentDensityInvocations != 0, formatOk ? "yes" : "no");
  if (!ff.fragmentDensityMap || !ff.fragmentDensityMapNonSubsampledImages || !formatOk)
    return false;
  // 32-pixel density texels, or the nearest the GPU takes.
  g_density.texel = {std::clamp(32u, fp.minFragmentDensityTexelSize.width, fp.maxFragmentDensityTexelSize.width),
                     std::clamp(32u, fp.minFragmentDensityTexelSize.height, fp.maxFragmentDensityTexelSize.height)};
  return true;
}

VkResult hook_create_device(void*, PFN_vkGetInstanceProcAddr getProcAddr, VkPhysicalDevice phys,
                            const VkDeviceCreateInfo* info, VkDevice* out) {
  // Foveation: VK_EXT_fragment_density_map on Dawn's device as well.
  auto& fd = g_density;
  fd.level = std::clamp(
      static_cast<int>(env_float("AURORA_XR_FOVEATION", static_cast<float>(g_config.xrFoveation))), 0, 3);
  VkDeviceCreateInfo dci = *info;
  std::vector<const char*> exts(info->ppEnabledExtensionNames,
                                info->ppEnabledExtensionNames + info->enabledExtensionCount);
  VkPhysicalDeviceFragmentDensityMapFeaturesEXT densityFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_DENSITY_MAP_FEATURES_EXT};
  if (fd.level > 0 && density_supported(getProcAddr, phys)) {
    exts.push_back(VK_EXT_FRAGMENT_DENSITY_MAP_EXTENSION_NAME);
    dci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    dci.ppEnabledExtensionNames = exts.data();
    densityFeatures.fragmentDensityMap = VK_TRUE;
    densityFeatures.fragmentDensityMapNonSubsampledImages = VK_TRUE;
    densityFeatures.pNext = const_cast<void*>(dci.pNext);
    dci.pNext = &densityFeatures;
  } else {
    fd.level = 0;
  }
  XrVulkanDeviceCreateInfoKHR xdci{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
  xdci.systemId = B.systemId;
  xdci.pfnGetInstanceProcAddr = getProcAddr;
  xdci.vulkanPhysicalDevice = phys;
  xdci.vulkanCreateInfo = &dci;
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

// The eye size (each eye's render area) at a share of the recommended size,
// as render_eyes picks it for dynamic resolution.
std::pair<uint32_t, uint32_t> dynres_eye_size(float scale) {
  const auto even = [](float v) { return (static_cast<uint32_t>(v + 0.5f) + 1u) & ~1u; };
  const auto& stereo = g_streams[kStereo];
  return {std::min(stereo.width, even(static_cast<float>(g_dynres.recWidth) * scale)),
          std::min(stereo.height, even(static_cast<float>(g_dynres.recHeight) * scale))};
}

// Foveation levels: full density inside r0, `mid` density out to r1, `out`
// beyond (per axis: 1, 1/2 or 1/4). r is the distance from the lens' centre
// in the eye's tangent space, 1 at the image's edge in each direction (the
// corners are ~1.4). Low was half density across only (2x1) out to 1.0: it
// measured no cheaper than off (2026-10-23), so the Quest 3 likely clamps
// 2x1 to 1x1; now it's Medium's 2x2 from further out, and no 4x2.
struct FoveationLevel {
  float r0, r1;
  float midX, midY, outX, outY;
};
constexpr std::array<FoveationLevel, 3> kFoveation{{
    {0.65f, 1.00f, 0.5f, 0.5f, 0.5f, 0.5f},
    {0.55f, 0.85f, 0.5f, 0.5f, 0.25f, 0.5f},
    {0.45f, 0.75f, 0.5f, 0.5f, 0.25f, 0.25f},
}};

// A density as the map's UNORM8: the fragment area is 1/density, rounded
// down to a size the GPU has, so 1/2 is stored just under (127 is 2.008 px).
uint8_t density_byte(float d) { return d >= 1.f ? 255 : d >= 0.5f ? 127 : 63; }

// The map for an eye size (the render area, w x h): one RG8 texel per
// density texel of the whole image, one layer per eye, for that eye's fov.
std::vector<uint8_t> density_texels(uint32_t w, uint32_t h, uint32_t mapW, uint32_t mapH, uint32_t layers,
                                    const std::array<XrView, 2>& views) {
  const auto& lv = kFoveation[static_cast<size_t>(std::clamp(g_density.level, 1, 3) - 1)];
  const auto& T = g_density.texel;
  std::vector<uint8_t> out(static_cast<size_t>(mapW) * mapH * 2 * layers);
  for (uint32_t layer = 0; layer < layers; ++layer) {
    const XrFovf& fov = views[std::min(layer, 1u)].fov;
    const float l = std::tan(fov.angleLeft), r = std::tan(fov.angleRight);
    const float u = std::tan(fov.angleUp), d = std::tan(fov.angleDown);
    for (uint32_t y = 0; y < mapH; ++y) {
      for (uint32_t x = 0; x < mapW; ++x) {
        // The texel's centre in the eye's tangent space (image top is up).
        const float fx = std::min((static_cast<float>(x) + 0.5f) * T.width / static_cast<float>(w), 1.f);
        const float fy = std::min((static_cast<float>(y) + 0.5f) * T.height / static_cast<float>(h), 1.f);
        const float tx = l + fx * (r - l), ty = u - fy * (u - d);
        const float nx = tx >= 0.f ? tx / r : tx / -l, ny = ty >= 0.f ? ty / u : ty / -d;
        const float dist = std::sqrt(nx * nx + ny * ny);
        const float dx = dist < lv.r0 ? 1.f : dist < lv.r1 ? lv.midX : lv.outX;
        const float dy = dist < lv.r0 ? 1.f : dist < lv.r1 ? lv.midY : lv.outY;
        uint8_t* p = &out[((static_cast<size_t>(layer) * mapH + y) * mapW + x) * 2];
        p[0] = density_byte(dx);
        p[1] = density_byte(dy);
      }
    }
  }
  return out;
}

// XR thread: makes the maps for `sizes` and starts their upload (on the XR
// queue, never waited on: density_frame picks them up once it's done).
bool density_build(const std::vector<std::pair<uint32_t, uint32_t>>& sizes, const std::array<XrView, 2>& views) {
  auto& fd = g_density;
  const auto& stereo = g_streams[kStereo];
  const uint32_t layers = stereo.layers;
  struct Part {
    DensityMaps::Map map;
    uint32_t mapW, mapH;
    size_t offset;
  };
  std::vector<Part> parts;
  std::vector<uint8_t> staging;
  const auto destroyParts = [&] {
    for (auto& p : parts) {
      if (p.map.view)
        vkDestroyImageView(B.dev, p.map.view, nullptr);
      if (p.map.image)
        vkDestroyImage(B.dev, p.map.image, nullptr);
      if (p.map.memory)
        vkFreeMemory(B.dev, p.map.memory, nullptr);
    }
  };
  for (const auto& [w, h] : sizes) {
    // Every map covers the whole image, not only the render area: the spec
    // sizes a density texel from the render area (and a map that size), but
    // the Quest 3's driver went by the image, so a corner-sized map was read
    // at twice the texel size, stretched right and down (2026-10-23). With
    // the whole image's size either reading gives 32-pixel texels while the
    // render area is over half the image (dynamic resolution's least is 0.6).
    Part p{};
    p.map.w = w;
    p.map.h = h;
    p.mapW = (stereo.width + fd.texel.width - 1) / fd.texel.width;
    p.mapH = (stereo.height + fd.texel.height - 1) / fd.texel.height;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8_UNORM;
    ici.extent = {p.mapW, p.mapH, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = layers;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(B.dev, &ici, nullptr, &p.map.image) != VK_SUCCESS) {
      destroyParts();
      return false;
    }
    parts.push_back(p);
    auto& m = parts.back().map;
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(B.dev, m.image, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = find_memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(B.dev, &mai, nullptr, &m.memory) != VK_SUCCESS ||
        vkBindImageMemory(B.dev, m.image, m.memory, 0) != VK_SUCCESS) {
      destroyParts();
      return false;
    }
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = m.image;
    vci.viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8_UNORM;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    if (vkCreateImageView(B.dev, &vci, nullptr, &m.view) != VK_SUCCESS) {
      destroyParts();
      return false;
    }
    parts.back().offset = staging.size();
    const auto texels = density_texels(w, h, p.mapW, p.mapH, layers, views);
    staging.insert(staging.end(), texels.begin(), texels.end());
  }
  if (parts.empty())
    return true;

  auto& up = fd.pending;
  const auto fail = [&] {
    destroyParts();
    if (up.fence)
      vkDestroyFence(B.dev, up.fence, nullptr);
    if (up.cmd)
      vkFreeCommandBuffers(B.dev, B.pool, 1, &up.cmd);
    if (up.buf)
      vkDestroyBuffer(B.dev, up.buf, nullptr);
    if (up.mem)
      vkFreeMemory(B.dev, up.mem, nullptr);
    up = {};
    return false;
  };
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = staging.size();
  bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (vkCreateBuffer(B.dev, &bci, nullptr, &up.buf) != VK_SUCCESS)
    return fail();
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(B.dev, up.buf, &req);
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = req.size;
  mai.memoryTypeIndex =
      find_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(B.dev, &mai, nullptr, &up.mem) != VK_SUCCESS ||
      vkBindBufferMemory(B.dev, up.buf, up.mem, 0) != VK_SUCCESS)
    return fail();
  void* mapped = nullptr;
  if (vkMapMemory(B.dev, up.mem, 0, staging.size(), 0, &mapped) != VK_SUCCESS)
    return fail();
  std::memcpy(mapped, staging.data(), staging.size());
  vkUnmapMemory(B.dev, up.mem);

  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = B.pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  if (vkAllocateCommandBuffers(B.dev, &cai, &up.cmd) != VK_SUCCESS)
    return fail();
  VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(up.cmd, &cbi);
  for (const auto& p : parts) {
    VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    bar.srcQueueFamilyIndex = bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image = p.map.image;
    bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(up.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &bar);
    VkBufferImageCopy region{};
    region.bufferOffset = p.offset;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers};
    region.imageExtent = {p.mapW, p.mapH, 1};
    vkCmdCopyBufferToImage(up.cmd, up.buf, p.map.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    // The layout the eyes' pass reads it in (Dawn's RenderPassMultiview).
    bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar.newLayout = VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT;
    bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_FRAGMENT_DENSITY_MAP_READ_BIT_EXT;
    vkCmdPipelineBarrier(up.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT,
                         0, 0, nullptr, 0, nullptr, 1, &bar);
  }
  if (vkEndCommandBuffer(up.cmd) != VK_SUCCESS)
    return fail();
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  if (vkCreateFence(B.dev, &fci, nullptr, &up.fence) != VK_SUCCESS)
    return fail();
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &up.cmd;
  if (vkQueueSubmit(B.queue, 1, &si, up.fence) != VK_SUCCESS)
    return fail();
  std::lock_guard lock{fd.mutex};
  for (const auto& p : parts) {
    fd.pendingMaps.push_back(fd.maps.size());
    fd.maps.push_back(p.map);
  }
  return true;
}

// XR thread, every frame while foveation is on: finishes the upload in
// flight, then makes the maps asked for (dynamic resolution's eye sizes
// first) once the views' fovs are known.
void density_frame() {
  auto& fd = g_density;
  if (!fd.on || fd.failed)
    return;
  auto& up = fd.pending;
  if (up.fence) {
    if (vkGetFenceStatus(B.dev, up.fence) != VK_SUCCESS)
      return;
    {
      std::lock_guard lock{fd.mutex};
      for (size_t i : fd.pendingMaps)
        fd.maps[i].ready = true;
      fd.pendingMaps.clear();
    }
    vkDestroyFence(B.dev, up.fence, nullptr);
    vkFreeCommandBuffers(B.dev, B.pool, 1, &up.cmd);
    vkDestroyBuffer(B.dev, up.buf, nullptr);
    vkFreeMemory(B.dev, up.mem, nullptr);
    up = {};
  }
  std::array<XrView, 2> views;
  {
    std::lock_guard lock{g_viewMutex};
    if (!g_viewsValid)
      return;
    views = g_latestViews;
  }
  std::vector<std::pair<uint32_t, uint32_t>> sizes;
  {
    std::lock_guard lock{fd.mutex};
    sizes.swap(fd.asked);
  }
  if (!fd.primed) {
    fd.primed = true;
    const auto& stereo = g_streams[kStereo];
    if (g_dynres.on) {
      const int levels = std::clamp(
          static_cast<int>((g_dynres.maxScale - g_dynres.minScale) / kDynresStep + 0.5f) + 1, 1, kDynresLevels);
      for (int i = 0; i < levels; ++i)
        sizes.push_back(dynres_eye_size(
            std::min(g_dynres.minScale + kDynresStep * static_cast<float>(i), g_dynres.maxScale)));
    } else {
      sizes.emplace_back(stereo.width, stereo.height);
    }
  }
  {
    std::lock_guard lock{fd.mutex};
    std::erase_if(sizes, [&](const auto& s) {
      return std::any_of(fd.maps.begin(), fd.maps.end(), [&](const auto& m) { return m.w == s.first && m.h == s.second; });
    });
  }
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  if (sizes.empty())
    return;
  if (!density_build(sizes, views)) {
    Log.error("Foveation: couldn't make {} density maps", sizes.size());
    fd.failed = true; // the nearest made stands in; with none, render_eyes skips the eyes
    return;
  }
  static bool logged = false;
  if (!logged) {
    logged = true;
    const XrFovf& f = views[0].fov;
    Log.info("Foveation level {}: {} density maps of {}x{} texels (left eye fov {:.1f} {:.1f} {:.1f} {:.1f} deg)",
             fd.level, sizes.size(), fd.texel.width, fd.texel.height, f.angleLeft * 57.2958f,
             f.angleRight * 57.2958f, f.angleUp * 57.2958f, f.angleDown * 57.2958f);
  }
}

// Render worker: whether any map is ready.
bool density_ready() {
  std::lock_guard lock{g_density.mutex};
  return std::any_of(g_density.maps.begin(), g_density.maps.end(), [](const auto& m) { return m.ready; });
}

// Render worker: the map for an eye size, as a VkImageView for Dawn; one of
// another size while it's being made (asked for here), 0 before any is ready.
uint64_t density_view(uint32_t w, uint32_t h) {
  auto& fd = g_density;
  std::lock_guard lock{fd.mutex};
  const DensityMaps::Map* best = nullptr;
  int64_t bestCost = INT64_MAX;
  bool known = false;
  for (const auto& m : fd.maps) {
    if (m.w == w && m.h == h) {
      known = true;
      if (m.ready)
        return reinterpret_cast<uintptr_t>(m.view);
    }
    if (!m.ready)
      continue;
    // The nearest, one at least as big preferred (it covers the framebuffer).
    const int64_t dw = static_cast<int64_t>(m.w) - w, dh = static_cast<int64_t>(m.h) - h;
    const int64_t cost = std::abs(dw) + std::abs(dh) + (dw < 0 || dh < 0 ? 100000 : 0);
    if (cost < bestCost) {
      bestCost = cost;
      best = &m;
    }
  }
  if (!known && std::find(fd.asked.begin(), fd.asked.end(), std::pair{w, h}) == fd.asked.end())
    fd.asked.emplace_back(w, h);
  return best ? reinterpret_cast<uintptr_t>(best->view) : 0;
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
// In a fight the game's own mapping (PADSetFightMapping) takes each input,
// the grips and stick clicks apart, instead (aurora_xr_get_pad).
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

bool suggest(const char* profile, const std::vector<std::pair<XrAction, const char*>>& bindings) {
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
      !create_action(B.gripBtn[0], "z_left", "Left grip", XR_ACTION_TYPE_BOOLEAN_INPUT) ||
      !create_action(B.gripBtn[1], "z_right", "Right grip", XR_ACTION_TYPE_BOOLEAN_INPUT) ||
      !create_action(B.stickClick[0], "stick_click_left", "Left stick click", XR_ACTION_TYPE_BOOLEAN_INPUT) ||
      !create_action(B.stickClick[1], "stick_click_right", "Right stick click", XR_ACTION_TYPE_BOOLEAN_INPUT) ||
      !create_action(B.btnStart, "start", "Start", XR_ACTION_TYPE_BOOLEAN_INPUT) ||
      !create_action(B.trigL, "l", "L", XR_ACTION_TYPE_FLOAT_INPUT) ||
      !create_action(B.trigR, "r", "R", XR_ACTION_TYPE_FLOAT_INPUT) ||
      !create_action(B.aim[0], "aim_left", "Left pointer", XR_ACTION_TYPE_POSE_INPUT) ||
      !create_action(B.aim[1], "aim_right", "Right pointer", XR_ACTION_TYPE_POSE_INPUT) ||
      !create_action(B.grab[0], "grab_left", "Left grab", XR_ACTION_TYPE_FLOAT_INPUT) ||
      !create_action(B.grab[1], "grab_right", "Right grab", XR_ACTION_TYPE_FLOAT_INPUT))
    return false;

  std::vector<std::pair<XrAction, const char*>> touchBindings{
      {B.stickMain, "/user/hand/left/input/thumbstick"},
      {B.stickC, "/user/hand/right/input/thumbstick"},
      {B.btnA, "/user/hand/right/input/a/click"},
      {B.btnB, "/user/hand/right/input/b/click"},
      {B.btnX, "/user/hand/left/input/x/click"},
      {B.btnY, "/user/hand/left/input/y/click"},
      {B.gripBtn[0], "/user/hand/left/input/squeeze/value"},
      {B.gripBtn[1], "/user/hand/right/input/squeeze/value"},
      {B.stickClick[0], "/user/hand/left/input/thumbstick/click"},
      {B.stickClick[1], "/user/hand/right/input/thumbstick/click"},
      {B.btnStart, "/user/hand/left/input/menu/click"},
      {B.trigL, "/user/hand/left/input/trigger/value"},
      {B.trigR, "/user/hand/right/input/trigger/value"},
      {B.aim[0], "/user/hand/left/input/aim/pose"},
      {B.aim[1], "/user/hand/right/input/aim/pose"},
      {B.grab[0], "/user/hand/left/input/squeeze/value"},
      {B.grab[1], "/user/hand/right/input/squeeze/value"},
  };
  // The game's rumble on both controllers' haptics. Suggested with the pad's
  // bindings (a later suggestion for a profile replaces the earlier one), and
  // the pad alone again if the runtime turns that down.
  bool touch = false;
  if (create_action(B.rumble, "rumble", "Rumble", XR_ACTION_TYPE_VIBRATION_OUTPUT)) {
    auto withHaptics = touchBindings;
    withHaptics.push_back({B.rumble, "/user/hand/left/output/haptic"});
    withHaptics.push_back({B.rumble, "/user/hand/right/output/haptic"});
    touch = suggest("/interaction_profiles/oculus/touch_controller", withHaptics);
  }
  if (!touch) {
    B.rumble = XR_NULL_HANDLE;
    Log.warn("Touch controller haptics not bound: no rumble");
    touch = suggest("/interaction_profiles/oculus/touch_controller", touchBindings);
  }
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

// The game's rumble on the controllers (XR thread, after xrSyncActions). A
// GameCube motor is on or off, with no length, so while it is on each
// controller gets a vibration that outlasts the next display frame and is
// renewed before it ends; off stops it at once. AURORA_XR_RUMBLE is the
// strength, 0..1 (0 turns it off).
void update_rumble(bool focused) {
  using namespace std::chrono;
  static const float amplitude = std::clamp(env_float("AURORA_XR_RUMBLE", 0.7f), 0.f, 1.f);
  static bool on = false;
  static steady_clock::time_point renewed{};
  // The runtime stops haptics itself when the session loses focus.
  if (!focused) {
    on = false;
    return;
  }
  if (B.rumble == XR_NULL_HANDLE)
    return;
  XrHapticActionInfo hi{XR_TYPE_HAPTIC_ACTION_INFO};
  hi.action = B.rumble;
  const bool want = g_rumbleWanted.load(std::memory_order_relaxed) && amplitude > 0.f;
  const auto now = steady_clock::now();
  if (want && (!on || now - renewed >= 100ms)) {
    XrHapticVibration v{XR_TYPE_HAPTIC_VIBRATION};
    v.duration = duration_cast<nanoseconds>(200ms).count();
    v.frequency = XR_FREQUENCY_UNSPECIFIED;
    v.amplitude = amplitude;
    xrApplyHapticFeedback(B.session, &hi, reinterpret_cast<const XrHapticBaseHeader*>(&v));
    on = true;
    renewed = now;
  } else if (!want && on) {
    xrStopHapticFeedback(B.session, &hi);
    on = false;
  }
}

void update_input() {
  if (!B.actionSet || !B.focused) {
    update_rumble(false);
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
  update_rumble(true);
  PADFightRaw raw{};
  const XrVector2f main = action_vec2(B.stickMain);
  const XrVector2f c = action_vec2(B.stickC);
  B.stickY = {main.y, c.y};
  // A controller dragging the screen reels it in and out with its stick
  // instead of steering the menu.
  if (B.reelHand != 0)
    raw.stick[0][0] = main.x, raw.stick[0][1] = main.y;
  if (B.reelHand != 1)
    raw.stick[1][0] = c.x, raw.stick[1][1] = c.y;
  const float l = B.triggersClick ? 0.f : action_float(B.trigL);
  const float r = B.triggersClick ? 0.f : action_float(B.trigR);
  raw.analog[0] = static_cast<u8>(std::lround(std::clamp(l, 0.f, 1.f) * 255.f));
  raw.analog[1] = static_cast<u8>(std::lround(std::clamp(r, 0.f, 1.f) * 255.f));
  const auto press = [&raw](bool on, int input) { raw.pressed |= on ? 1u << input : 0u; };
  press(action_bool(B.btnA), PAD_FIGHT_TOUCH_A);
  press(action_bool(B.btnB), PAD_FIGHT_TOUCH_B);
  press(action_bool(B.btnX), PAD_FIGHT_TOUCH_X);
  press(action_bool(B.btnY), PAD_FIGHT_TOUCH_Y);
  press(!B.pointerMode && action_bool(B.gripBtn[0]), PAD_FIGHT_TOUCH_LEFT_GRIP);
  press(!B.pointerMode && action_bool(B.gripBtn[1]), PAD_FIGHT_TOUCH_RIGHT_GRIP);
  press(action_bool(B.stickClick[0]), PAD_FIGHT_LEFT_STICK_CLICK);
  press(action_bool(B.stickClick[1]), PAD_FIGHT_RIGHT_STICK_CLICK);
  press(l > 0.9f, PAD_FIGHT_LEFT_TRIGGER);
  press(r > 0.9f, PAD_FIGHT_RIGHT_TRIGGER);
  raw.start = action_bool(B.btnStart);
  std::lock_guard lock{g_padMutex};
  g_touchRaw = raw;
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
    g_headHistory[g_headHistoryN++ % g_headHistory.size()] = {B.head, B.headOrientation};
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
// point placed at the arena position, aurora_xr_set_stage): wider and
// taller than any stage's main platform, so pointing near it is enough.
constexpr XrVector3f kGrabBoxMin{-120.f, -80.f, -60.f};
constexpr XrVector3f kGrabBoxMax{120.f, 100.f, 60.f};
// Meters per game unit: Final Destination from about 25 cm to 8.5 m wide.
constexpr float kMinArenaScale = 0.0015f, kMaxArenaScale = 0.05f;

// A room point (meters) in game units from the arena center. Not from the
// world origin: on Corneria, centered on the Great Fox far above it, the box
// sat well below the ship.
XrVector3f to_game(const ArenaPose& a, XrVector3f p) { return rot_y(p - a.pos, -a.yaw) * (1.f / a.scale); }

bool in_grab_box(const ArenaPose& a, XrVector3f p) {
  const XrVector3f g = to_game(a, p);
  return g.x >= kGrabBoxMin.x && g.x <= kGrabBoxMax.x && g.y >= kGrabBoxMin.y && g.y <= kGrabBoxMax.y &&
         g.z >= kGrabBoxMin.z && g.z <= kGrabBoxMax.z;
}

// Distance along the ray (meters) to the arena's grab box, or -1 for a miss.
float hit_arena(const ArenaPose& a, XrVector3f origin, XrVector3f dir) {
  const XrVector3f o = to_game(a, origin);
  const XrVector3f d = rot_y(dir, -a.yaw) * (1.f / a.scale);
  const float os[3] = {o.x, o.y, o.z}, ds[3] = {d.x, d.y, d.z};
  const float lo[3] = {kGrabBoxMin.x, kGrabBoxMin.y, kGrabBoxMin.z};
  const float hi[3] = {kGrabBoxMax.x, kGrabBoxMax.y, kGrabBoxMax.z};
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
    if (hand.valid && !hand.inside && hit_screen_plane(screen, hand.origin, hand.dir, t, l)) {
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

// The player watches from behind the stage (the side away from the HUD's
// face). XR thread.
std::atomic<bool> g_viewBehind{false};

// Per frame: whether the head is behind the stage, by the side of the arena
// it's on (with ~15 deg either side of the stage's left-right line before it
// switches). While a fight shows from behind, every pad's left and right are
// swapped, so left on the stick runs to the player's left
// (AURORA_XR_BEHIND=0: never).
void behind_frame(bool fight) {
  static const bool enabled = env_flag("AURORA_XR_BEHIND", true);
  const ArenaPose arena = arena_pose();
  const XrVector3f local = rot_y(B.head - arena.pos, -arena.yaw);
  const float flat = std::sqrt(local.x * local.x + local.z * local.z);
  if (flat > 0.01f) {
    const float front = local.z / flat; // cos of the angle from the stage's front
    const bool behind = g_viewBehind ? front < 0.26f : front < -0.26f;
    if (behind != g_viewBehind) {
      g_viewBehind = behind;
      Log.info("Viewer {} the stage", behind ? "behind" : "in front of");
    }
  }
  PADSetMirrorX(enabled && fight && g_viewBehind);
}

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
  // From behind the stage it turns around to face the player, over the
  // stage's front edge (the far one from there).
  const float yaw = g_viewBehind ? arena.yaw + 3.14159265f : arena.yaw;
  pose.orientation = yaw_quat(yaw);
  pose.position = arena.pos + rot_y({0.f, height, -0.15f * k}, yaw);
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

// a beam and a dot per hand, and the screen's bar and Back button
constexpr size_t kMaxPointerLayers = 6;

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
// it. Adds to what `out` has.
void build_pointer_layers(PointerLayers& out, bool onTargetOnly = false) {
  if (!B.beamSwapchain || !B.dotSwapchain)
    return;
  for (int h = 0; h < 2; ++h) {
    const Hand& hand = G.hands[h];
    if (!hand.valid || (onTargetOnly && !hand.engaged && !hand.clicking && !hand.inside && hand.hit < 0.f))
      continue;
    const XrColor4f color = hand.engaged                     ? XrColor4f{1.f, 0.8f, 0.3f, 1.f}
                            : hand.clicking                  ? XrColor4f{1.f, 1.f, 1.f, 1.f}
                            : hand.inside || hand.hit >= 0.f ? XrColor4f{0.45f, 0.85f, 1.f, 1.f}
                                                             : XrColor4f{0.85f, 0.9f, 1.f, 0.5f};
    if (hand.engaged ? hand.direct : hand.inside) {
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

// The room survey (raycasts) choosing where stages and the screen start: off
// for now, AURORA_XR_ROOM_SURVEY=1 turns it on. Its fits were often awkward to
// work with, and Place Stage now sets where every stage starts; off, stages
// start at the arena's default (0.7 m ahead, 0.25 m under the eyes, 0.0035 m
// a unit: between the placing cards and their legend) and the screen at its
// own. The stage shadow's surface casts don't depend on it.
constexpr bool kRoomSurveyDefault = false;

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
  int captureCasting = 0;  // capture_casting's, last loop (for the log)
  int64_t lightCastNs = 0; // light_cast's last
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
// Ambient room capture's pace: a frame (RoomCapture skips the rest) and a
// light_cast each 200 ms.
constexpr int kCapAmbientMs = 200;
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

// Room capture's (below): whether one is running, and a depth hit along a
// ray from the head while it does.
int capture_casting();
void capture_ambient(bool ok);
void capture_moved(XrVector3f p0, float y0, XrVector3f p1, float y1);
void capture_note_cast();
void capture_note_hit(const XrVector3f& origin, const XrVector3f& point);

// While a room capture runs: a grid of rays across the view (5 across, 4
// down, within 25 and 18 degrees of its middle), each hit noted by its
// direction from the head, so the capture's lights get a place in the room
// as well as a direction (a lamp or window seen straight on, which a
// capture asks for).
void light_cast(SurfaceWorker& w, const XrPosef& head, XrTime time) {
  XrEnvironmentRaycastFilterDistanceMETA reach{XR_TYPE_ENVIRONMENT_RAYCAST_FILTER_DISTANCE_META};
  reach.maxDistance = kRoomReach;
  const XrEnvironmentRaycastFilterBaseHeaderMETA* filters[] = {
      reinterpret_cast<const XrEnvironmentRaycastFilterBaseHeaderMETA*>(&reach)};
  capture_note_cast();
  for (const float x : {-0.47f, -0.23f, 0.f, 0.23f, 0.47f})
    for (const float y : {-0.32f, -0.11f, 0.11f, 0.32f}) {
      const float n = std::sqrt(x * x + y * y + 1.f);
      XrEnvironmentRaycastHitGetInfoMETA gi{XR_TYPE_ENVIRONMENT_RAYCAST_HIT_GET_INFO_META};
      gi.baseSpace = B.space;
      gi.time = time;
      gi.origin = head.position;
      gi.direction = qrot(head.orientation, {x / n, y / n, -1.f / n});
      gi.filterCount = 1;
      gi.filters = filters;
      XrEnvironmentRaycastHitMETA hit{XR_TYPE_ENVIRONMENT_RAYCAST_HIT_META};
      if (XR_SUCCEEDED(B.performRaycast(w.raycaster, &gi, &hit)) &&
          (hit.status == XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_META ||
           hit.status == XR_ENVIRONMENT_RAYCAST_HIT_STATUS_HIT_INVALID_ORIENTATION_META))
        capture_note_hit(head.position, hit.pose.position);
    }
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
    {
      std::unique_lock lock{g_surface.mutex};
      // 15 casts a second while casting, the raycaster's future polled every
      // 50 ms while it's created; otherwise asleep until asked (waking each
      // second to let an idle raycaster go).
      const bool surveyDue = (g_surface.surveyWanted && !g_surface.survey.done) || capture_casting() != 0;
      if (!g_surface.stop && !g_surface.reset)
        g_surface.cv.wait_for(lock, w.caster == Caster::Creating ? std::chrono::milliseconds(50)
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
    }
    const int64_t now = monotonic_ns();
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
    const int casting = capture_casting();
    if (casting != w.captureCasting) {
      w.captureCasting = casting;
      if (casting == 1)
        Log.info("Room capture: casting (raycaster {}, head {}, time {})", static_cast<int>(w.caster), headValid,
                 time != 0);
    }
    // Ambient capture's casts at its frames' pace.
    if (casting != 0 && (casting == 1 || now - w.lightCastNs >= kCapAmbientMs * kMsNs) && headValid && time != 0 &&
        B.envRaycast && (w.caster == Caster::Off || w.caster == Caster::Creating || w.caster == Caster::Ready)) {
      raycaster_step(w);
      if (w.caster == Caster::Ready) {
        light_cast(w, head, time);
        w.lastCastNs = w.lightCastNs = now;
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
  static const bool survey = env_flag("AURORA_XR_ROOM_SURVEY", kRoomSurveyDefault);
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
  static const bool survey = env_flag("AURORA_XR_ROOM_SURVEY", kRoomSurveyDefault);
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
  capture_moved(p0, y0, p1, y1); // the room's captured light keeps its place too
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

XrQuaternionf qmul(const XrQuaternionf& a, const XrQuaternionf& b);

// A recenter (LOCAL) puts the origin at the head, facing where it looks. The
// stage and the screen keep their places in the room: everything placed
// moves the other way. The report's pose is meant to be the new origin in
// the old space; Quest's (Oculus 209) is the old origin in the new space
// instead (2026-10-23: eight recenters, each the inverse of the head's
// jump), which turned and threw the stage across the room. Which reading
// holds comes from the head's last poses, since the new origin is where the
// head just was. The move waits for the head to show in the new space
// (spot_recenter_frame: 0-2 frames after the report on Quest), so nothing is
// drawn displaced in between.
struct {
  bool pending = false;
  XrVector3f origin{}; // the new origin in the old space
  float yaw = 0.f;
  XrVector3f oldHead{}; // where the head was, in the old space
  int frames = 0;
} g_recenter;

void spot_recenter_frame();

void spot_recentered(const XrEventDataReferenceSpaceChangePending& e) {
  if (e.referenceSpaceType != XR_REFERENCE_SPACE_TYPE_LOCAL || !e.poseValid)
    return;
  if (g_recenter.pending) { // another one already: that one first
    g_recenter.frames = 10;
    spot_recenter_frame();
  }
  const XrVector3f t = e.poseInPreviousSpace.position;
  const float th = pose_yaw(e.poseInPreviousSpace.orientation);
  const XrVector3f inv = rot_y(XrVector3f{-t.x, -t.y, -t.z}, -th); // the inverse's translation
  // How far a head pose is from an origin (meters, with half a meter a radian).
  auto off = [](const HeadSample& h, XrVector3f o, float yaw) {
    float turn = pose_yaw(h.orientation) - yaw;
    while (turn > 3.14159265f)
      turn -= 6.2831853f;
    while (turn < -3.14159265f)
      turn += 6.2831853f;
    return vlen(h.pos - o) + 0.5f * std::fabs(turn);
  };
  float asSpec = std::numeric_limits<float>::infinity(), asInverse = asSpec;
  XrVector3f specHead = t, inverseHead = inv;
  const size_t n = std::min(g_headHistoryN, g_headHistory.size());
  for (size_t i = 0; i < n; ++i) {
    const HeadSample& h = g_headHistory[i];
    if (const float d = off(h, t, th); d < asSpec)
      asSpec = d, specHead = h.pos;
    if (const float d = off(h, inv, -th); d < asInverse)
      asInverse = d, inverseHead = h.pos;
  }
  const bool spec = asSpec <= asInverse;
  g_recenter = {true, spec ? t : inv, spec ? th : -th, spec ? specHead : inverseHead, 0};
  Log.info("Recentered: the runtime's pose read {} ({:.2f} off the head before it, {:.2f} the other way)",
           spec ? "as the new origin" : "as the old origin", spec ? asSpec : asInverse, spec ? asInverse : asSpec);
}

// XR thread, each frame once the head is located: a recenter's move, once
// the head shows in the new space (or after 10 frames).
void spot_recenter_frame() {
  if (!g_recenter.pending)
    return;
  const XrVector3f o = g_recenter.origin;
  const float yaw = g_recenter.yaw;
  const XrVector3f moved = rot_y(g_recenter.oldHead - o, -yaw); // the old head, in the new space
  if (vlen(B.head - moved) >= vlen(B.head - g_recenter.oldHead) && ++g_recenter.frames < 10)
    return;
  g_recenter.pending = false;
  // p_new = R^-1 (p_old - o): the old origin goes to rot_y(-o, -yaw).
  {
    std::lock_guard lock{g_arenaMutex};
    if (g_arenaInit)
      move_arenas(o, yaw, {0.f, 0.f, 0.f}, 0.f);
    if (g_spot.haveLast) {
      g_spot.lastPos = rot_y(g_spot.lastPos - o, -yaw);
      g_spot.lastYaw -= yaw;
    }
  }
  const XrQuaternionf turn = yaw_quat(-yaw);
  auto move = [&](ScreenPose& s) {
    s.pos = rot_y(s.pos - o, -yaw);
    s.orientation = qmul(turn, s.orientation);
  };
  if (g_screenInit)
    move(g_screen);
  move(g_screenGlide.from);
  move(g_screenGlide.to);
  Log.info("Recentered: the stage and the screen kept their places in the room ({:.2f} m, {:.0f} deg)", vlen(o),
           yaw * 57.2958f);
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

// Whether a point (in B.space) is inside either eye's view: 1 yes, 0 no, -1
// no views yet.
int in_view(XrVector3f p) {
  std::array<XrView, 2> views;
  {
    std::lock_guard lock{g_viewMutex};
    if (!g_viewsValid)
      return -1;
    views = g_latestViews;
  }
  for (const XrView& v : views) {
    const XrVector3f l = qrot(qconj(v.pose.orientation), p - v.pose.position);
    if (l.z > -0.05f)
      continue;
    const float x = l.x / -l.z, y = l.y / -l.z;
    if (x >= std::tan(v.fov.angleLeft) && x <= std::tan(v.fov.angleRight) && y >= std::tan(v.fov.angleDown) &&
        y <= std::tan(v.fov.angleUp))
      return 1;
  }
  return 0;
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
  // One counts only if it's in view the first time it's located: a spot
  // saved in another room can still be located through the walls (the same
  // space map), and took the stage there.
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
      const int seen = in_view(pos);
      if (seen < 0)
        continue;
      if (seen == 0) {
        Log.info("Stage spot: saved anchor {} is out of view ({:.1f} m away); not used this session",
                 uuid_hex(g_spot.candidates[i].uuid), vlen(pos - B.head));
        xrDestroySpace(g_spot.candidates[i].space);
        g_spot.candidates.erase(g_spot.candidates.begin() + static_cast<long>(i));
        --i;
        continue;
      }
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

// When LOCAL was last recentered (monotonic_ns): the placing cards start over
// about the head, which the new space puts somewhere else.
std::atomic<int64_t> g_recenteredNs{0};

struct PlacingLayers {
  std::array<XrCompositionLayerQuad, kMaxPlacingLayers> quads;
  uint32_t count = 0;
};

void build_placing_layers(PlacingLayers& out, int mode) {
  out.count = 0;
  // Where each sits against the head (x right, y up, z back, meters, turned
  // with the layout's yaw), and its width: the cards an arm and a half
  // ahead, about 35 degrees wide, above the middle of the view, and the
  // legend well under it, so the two frame the stage being placed between
  // them. AURORA_XR_PLACE_CARDS / _LEGEND: "dx,dy,dz,width".
  struct Spot {
    const char* env;
    XrVector3f offset;
    float width;
  };
  const std::array<Spot, 2> spots{{{"AURORA_XR_PLACE_CARDS", {0.f, 0.1f, -1.2f}, 0.75f},
                                    {"AURORA_XR_PLACE_LEGEND", {0.f, -0.6f, -1.f}, 0.3f}}};
  // Soft head lock, sideways only: the layout turns about where the head
  // stood when the hold began, lagging behind once the head has turned past
  // a dead zone, and settles when it catches up; inside the dead zone it
  // creeps back to center once the head has held still for a second and a
  // half, unless it's looking at the cards (reading one at the side mustn't
  // slide it away). Height and pitch stay put.
  // The layout keeps to the head, not to any stage: the default arena isn't
  // moved by a recenter and a placed stage is elsewhere, and measured from
  // them the cards could end up at the player's face. A recenter, which
  // moves the head in LOCAL, starts the turning over about the head, for the
  // half second the new head pose takes to come.
  static auto last = std::chrono::steady_clock::time_point{};
  static float yaw = 0.f;
  static bool following = false;
  static float still = 0.f; // seconds the head has turned under 10 deg/s
  static float prevHeadYaw = 0.f;
  static XrVector3f pivot{};
  const auto now = std::chrono::steady_clock::now();
  const float dt = std::chrono::duration<float>(now - last).count();
  last = now;
  const XrVector3f fwd = qrot(B.headOrientation, {0.f, 0.f, -1.f});
  const float headYaw = std::atan2(-fwd.x, -fwd.z); // rot_y's sense
  const bool recentered = monotonic_ns() - g_recenteredNs.load(std::memory_order_relaxed) < 500'000'000;
  constexpr float kPi = 3.14159265f;
  const auto wrap = [&](float a) { return a - 2.f * kPi * std::floor((a + kPi) / (2.f * kPi)); };
  // Whether the head's forward ray meets a picture where the layout puts it
  // now, with a 10 cm margin (its height taken as half its width at most).
  const auto looking_at = [&](const Spot& sp) {
    const XrVector3f p = pivot + rot_y(sp.offset, yaw);
    const XrVector3f n = vnorm(pivot - p);
    const float facingRay = vdot(fwd, n);
    if (facingRay > -1e-3f)
      return false;
    const XrVector3f hit = B.head + fwd * (vdot(p - B.head, n) / facingRay);
    const XrVector3f local = hit - p;
    const float across = vdot(local, rot_y({1.f, 0.f, 0.f}, yaw));
    return std::fabs(across) < sp.width * 0.5f + 0.1f && std::fabs(local.y) < sp.width * 0.25f + 0.1f;
  };
  if (dt > 0.25f || recentered) { // a new hold, or a recenter: start where the head looks
    yaw = headYaw;
    following = false;
    still = 0.f;
    pivot = B.head;
  } else {
    const float turn = std::fabs(wrap(headYaw - prevHeadYaw)) / std::max(dt, 1e-3f);
    still = turn < 10.f * kPi / 180.f ? still + dt : 0.f;
    const float diff = wrap(headYaw - yaw);
    if (std::fabs(diff) > 20.f * kPi / 180.f)
      following = true;
    if (following) {
      yaw += diff * std::min(1.f, dt * 4.f);
      if (std::fabs(diff) < 1.f * kPi / 180.f)
        following = false;
    } else if (still > 1.5f && !(mode >= 2 && looking_at(spots[0]))) {
      yaw += diff * std::min(1.f, dt * 2.f); // the creep back
    }
  }
  prevHeadYaw = headYaw;
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
    const XrVector3f pos = pivot + rot_y(spot.offset, yaw);
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

// ---------------------------------------------------------------- Room lighting (dev tools)
//
// An optional stand-in for the game's lighting in the 3D view: light from
// the room around the stage, blended with the game's by a weight for the
// fighters (and their items), one for the stage's lit draws and one for its
// unlit ones (baked lighting), each 0 (the game's) to 1 (the room's). The
// eye shaders take it (ShaderConfig::xrRoom*, set per draw from the game's
// GX link markers; render_eyes fills the uniform). Until rooms are captured,
// the room is one of a few presets: a sky gradient (up, horizon, down) in
// SH L2, and a key and a fill light set against the stage (they turn with
// it; YAW turns them about its up axis). Lighting is per vertex, as GX's
// is, or per pixel for the room's part (a comparison, and its cost), or per
// pixel with facet normals: the stage's unlit draws, which have no normals,
// lit by the room as their triangles face (a flat-shaded relight of baked
// stages), or with smooth normals the CPU works out from their triangles
// where it decodes their vertices (the Quest's default; AURORA_VTX_DECODE=1
// on a desktop), facets elsewhere. Unlit draws lit by normals are lit
// relative to the room's light from above: their baked light stays, and the
// room adds its shading and colour.
//
// AURORA_XR_ROOM_LIGHT "source,fighters,stage,unlit,strength,yaw,shading,view,delight,gloss,keys,probes"
// (any leading part; source 0 off, 1 lamp, 2 day, 3 office, 4 the room as a
// capture saw it (room capture, below); shading 0 per
// vertex, 1 per pixel, 2 facet, 3 smooth; view 0 off, 1 lit/unlit, 2 fighters/stage,
// 3 normals; delight 0 to 1, how much of a fit of the stage's baked light
// SMOOTH divides out of its vertex colours; gloss 0 to 1; keys 0 to 2;
// probes 0 a grid over the stage, 1 a line across it) sets it at launch;
// the dev tools panel's LIGHT page sets it live and logs each change as
// that value.

#ifndef __ANDROID__
// AURORA_XR_ROOM_REPLAY's capture poll (init_room_light).
std::thread g_replayPoll;
std::atomic<bool> g_replayPollStop{false};
#endif

struct RoomPreset {
  const char* name;
  std::array<float, 3> up, horizon, down; // the sky gradient's colours
  std::array<float, 3> keyDir, key, fillDir, fill; // directions in the stage's world (y up, z toward the front)
};
constexpr std::array<RoomPreset, 3> kRoomPresets{{
    // An evening room lit by a lamp to the front left: warm, a dim floor.
    {"LAMP", {0.42f, 0.36f, 0.30f}, {0.36f, 0.30f, 0.24f}, {0.18f, 0.15f, 0.12f},
     {-0.55f, 0.70f, 0.45f}, {0.70f, 0.52f, 0.34f}, {0.60f, 0.20f, -0.50f}, {0.10f, 0.10f, 0.13f}},
    // Daylight through a window to the right, low.
    {"DAY", {0.50f, 0.54f, 0.60f}, {0.44f, 0.46f, 0.48f}, {0.26f, 0.24f, 0.21f},
     {0.85f, 0.35f, 0.40f}, {0.62f, 0.60f, 0.55f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 0.f}},
    // Overhead fluorescent light: bright, flat, cool.
    {"OFFICE", {0.62f, 0.64f, 0.66f}, {0.48f, 0.50f, 0.51f}, {0.28f, 0.28f, 0.28f},
     {0.10f, 1.00f, 0.25f}, {0.38f, 0.40f, 0.42f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 0.f}},
}};

struct RoomLight {
  int source = 0; // 0 off, else 1 + a kRoomPresets index
  float fighters = 1.f, stage = 1.f, unlit = 0.f, strength = 1.f, yaw = 0.f; // yaw in degrees
  int shading = 0; // 0 per vertex, 1 per pixel, 2 per pixel with facet normals where meshes have none, 3 smooth ones
  int view = 0;
  float delight = 1.f;
  float gloss = 1.f; // the room's specular highlight: fighters by kFighterGloss, the stage by stage_gloss
  float keys = 1.f;  // the key and fill lights' strength, on top of STRENGTH (0-2)
  int probes = 0;    // where the light is made (ProbeGrid): 0 a grid over the stage, 1 a line across it
};

// The fighters' share of GLOSS (tuned on the headset: at 1 they looked too
// shiny, the stages' metal didn't; 0.5 still far too heavy, 2026-10-23).
constexpr float kFighterGloss = 0.1f;

// The current stage (GrKind, aurora_xr_set_stage), for stage_gloss.
std::atomic<int> g_roomStage{-1};

// How glossy a stage's own surfaces are (times GLOSS): the metal ones, and
// the cars of the F-Zero stages; 0 elsewhere (no highlight pipelines). At
// most 0.1, like the fighters (1 and 0.6 were far too heavy, 2026-10-23).
float stage_gloss(int grkind) {
  switch (grkind) {
  case 0x0E: // Corneria: the Great Fox
  case 0x0F: // Venom
  case 0x12: // Mute City
  case 0x13: // Big Blue
  case 0x22: // Big Blue's route
    return 0.1f;
  default:
    return 0.f;
  }
}

struct {
  std::mutex mutex;
  RoomLight value, defaults;
  std::array<std::array<float, 3>, 9> sh[kRoomPresets.size()]; // per preset, unscaled (room_sh)
} g_room;

// Room capture (source ROOM): the room's light from a passthrough camera
// (the platform's capture thread, aurora_xr_room_capture_*), estimated by
// Roomtone (github.com/rhythmerc/roomtone): each frame's grid of linear
// colour with its exposure and the head's pose, and the raycasts' depth
// hits, become SH L2, up to two lights with a place in the room, and a
// level that matches the brightness passthrough shows the room at, at the
// stage. A capture runs 20 s from choosing ROOM, the head looking around,
// then ambiently while ROOM stays the source (in the menus, while the stage
// is placed, in a pause), at kCapAmbientMs a frame. In room_frame the
// stage's unlit draws take the level from above (their tops the baked
// colour times the level, times STRENGTH), and the fighters as the room's
// mean light over every direction (kCapFighters times the level): a
// backlit room shades their front, but on average they keep the game's
// brightness. AURORA_XR_ROOM_RECORD=path records the capture's inputs for
// roomtone_replay.
constexpr int64_t kCaptureNs = 20'000'000'000;
// The fighters' mean light at level 1.
constexpr float kCapFighters = 0.7f;
constexpr int kRoomSourceCaptured = static_cast<int>(kRoomPresets.size()) + 1;

// The points the light is made for (room_frame's ProbeGrid), each lifted
// to about a fighter's middle (kCapLift, game units, so it scales with the
// stage: 4 cm at the default 0.0035; the stage's origin is on the table,
// whose own surface Roomtone leaves out within 15 cm). GRID (PROBES, the
// default): a grid over the stage's footprint and up to its highest floor,
// a point about every kCapSpacing (3-6 across, 1-3 up, 1-2 deep). LINE:
// three across its width (its footprint's 20%, 50% and 80%). The shader
// blends their SH by where a vertex or pixel is among them; a fighter takes
// the blend at its middle (aurora_xr_light_probe). The reference point
// (the middle, low: the level's exposure and the key lights) is made first.
constexpr int kCapPoints = gfx::XrRoomProbes;
constexpr float kCapLift = 11.4f;
constexpr float kCapSpacing = 0.6f; // metres
constexpr std::array<int, 3> kCapGridMin{3, 1, 1}, kCapGridMax{6, 3, 2};

// The probe points: n[0] x n[1] x n[2] of them, x fastest, from `origin`
// along each axis (the whole span; zero where n is 1), in the room.
struct ProbeGrid {
  std::array<int, 3> n{1, 1, 1};
  XrVector3f origin{};
  std::array<XrVector3f, 3> axis{};
  int ref = 0; // the reference point's index
  int count() const { return n[0] * n[1] * n[2]; }
  XrVector3f at(int i) const {
    const int ix = i % n[0], iy = i / n[0] % n[1], iz = i / (n[0] * n[1]);
    const auto f = [](int k, int m) { return m > 1 ? static_cast<float>(k) / static_cast<float>(m - 1) : 0.f; };
    return origin + axis[0] * f(ix, n[0]) + axis[1] * f(iy, n[1]) + axis[2] * f(iz, n[2]);
  }
};

// The captured light for room_frame (tracking space): the SH at each point,
// the two lights at the middle one, with their places and the distance at
// which their colours apply (the shader's point lights), the reference's
// mean light and the level.
struct CapLight {
  std::array<std::array<std::array<float, 3>, 9>, kCapPoints> sh{};
  ProbeGrid grid;
  std::array<std::array<float, 3>, 2> dir{}, col{};
  std::array<bool, 2> placed{};
  std::array<XrVector3f, 2> pos{};
  std::array<float, 2> range{};
  float mean = 1.f, level = 1.f;
};

struct Capture {
  // When a capture runs (quick: the XR thread's, every frame).
  std::mutex mutex;
  bool requested = false;
  int64_t untilNs = 0; // steady clock; 0: not started (no head pose yet)
  // Ambient capture: while ROOM is the source, in the menus, while the stage
  // is placed and in a pause (the head looks around the room there, past
  // lights the camera's narrower view misses while it looks ahead).
  bool room = false, ambientOk = false;
  uint32_t ambientFrames = 0; // since the last log
  uint32_t casts = 0;
  bool any = false; // the estimate has a frame (capture_resume)
  // The latest depth image (room_depth_feed, the XR thread), for the capture
  // thread's next frame.
  bool depthPending = false;
  roomtone::DepthImage depth;
  std::vector<uint16_t> depthPx;
  // The estimate: a make takes milliseconds (a build per point, with
  // voxels), on the capture thread; the XR thread only try-locks it.
  std::mutex estMutex;
  roomtone::Room est;
  roomtone::Recorder rec;
  bool recTried = false;
  // capture_light's last answer, for frames the capture thread holds it.
  bool haveLast = false;
  CapLight last;
} g_capture;

int64_t steady_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

roomtone::Vec3 rt(const XrVector3f& v) { return {v.x, v.y, v.z}; }
roomtone::Quat rt(const XrQuaternionf& q) { return {q.x, q.y, q.z, q.w}; }

// AURORA_XR_ROOM_RECORD: opened with the first capture. Under estMutex.
void capture_record_open_locked() {
  auto& c = g_capture;
  if (std::exchange(c.recTried, true))
    return;
  if (const char* path = std::getenv("AURORA_XR_ROOM_RECORD"); path && *path)
    Log.info("Room capture: recording to {} ({})", path, c.rec.open(path) ? "open" : "failed");
}

// Starts a capture (choosing ROOM), what was seen dropped.
void capture_start() {
  auto& c = g_capture;
  {
    std::lock_guard lock{c.mutex};
    c.requested = true;
    c.room = true;
    c.ambientFrames = 0;
    c.untilNs = 0;
    c.casts = 0;
    c.any = false;
    c.depthPending = false;
  }
  std::lock_guard lock{c.estMutex};
  capture_record_open_locked();
  c.est.reset();
  c.rec.reset();
  c.haveLast = false;
}

// ROOM chosen again: the capture so far kept (ambient capture goes on from
// it), so ROOM and another source compare back to back; a capture only
// when there is none yet (RECAPTURE starts over).
void capture_resume() {
  {
    std::lock_guard lock{g_capture.mutex};
    if (g_capture.any) {
      g_capture.room = true;
      return;
    }
  }
  capture_start();
}

// Another source chosen: no ambient capture.
void capture_stop() {
  std::lock_guard lock{g_capture.mutex};
  g_capture.requested = g_capture.room = false;
}

// Under g_arenaMutex (move_arenas): the room moved in LOCAL, (p0, y0) to
// (p1, y1); the capture and its light move with it.
void capture_moved(XrVector3f p0, float y0, XrVector3f p1, float y1) {
  auto& c = g_capture;
  std::lock_guard lock{c.estMutex};
  c.rec.moved(rt(p0), y0, rt(p1), y1);
  if (c.est.moved(rt(p0), y0, rt(p1), y1))
    Log.info("Room capture: the room turned {:.0f} deg in tracking space; its lights kept their place",
             (y1 - y0) * 57.2958f);
}

// A capture's frames wanted now: 1 (the 20 s after choosing ROOM), 2
// (ambient), 0 (none). Under g_capture.mutex.
int capture_mode_locked() {
  const auto& c = g_capture;
  return c.requested ? 1 : c.room && c.ambientOk ? 2 : 0;
}

// light_cast's: 1 while a capture's first frames are in, 2 ambient, 0 not.
int capture_casting() {
  std::lock_guard lock{g_capture.mutex};
  const int mode = capture_mode_locked();
  return mode == 1 ? (g_capture.untilNs != 0 ? 1 : 0) : mode == 2 && g_capture.any ? 2 : 0;
}

void capture_note_cast() {
  std::lock_guard lock{g_capture.mutex};
  ++g_capture.casts;
}

// The surface worker's: dropped while a make runs (a grid's takes long
// enough to stall the placing surface; there are hits to spare).
void capture_note_hit(const XrVector3f& origin, const XrVector3f& point) {
  std::unique_lock lock{g_capture.estMutex, std::try_to_lock};
  if (!lock.owns_lock())
    return;
  g_capture.est.addHit(rt(origin), rt(point));
  g_capture.rec.hit(rt(origin), rt(point));
}

// The captured light at the points (room_frame, the XR thread): false while
// none is made; while the capture thread makes it, the last answer.
// Roomtone takes the grid's reference point first, then the rest in order.
bool capture_light(const ProbeGrid& grid, CapLight& out) {
  auto& c = g_capture;
  if (std::unique_lock lock{c.estMutex, std::try_to_lock}; lock.owns_lock()) {
    const int n = grid.count();
    std::array<int, kCapPoints> order; // Roomtone's point i is the grid's order[i]
    order[0] = grid.ref;
    for (int i = 0, k = 1; i < n; ++i)
      if (i != grid.ref)
        order[k++] = i;
    std::array<roomtone::Vec3, kCapPoints> pts;
    for (int i = 0; i < n; ++i)
      pts[i] = rt(grid.at(order[i]));
    if (c.rec.isOpen())
      c.rec.points(pts.data(), static_cast<size_t>(n), steady_ns());
    c.est.setPoints(pts.data(), static_cast<size_t>(n));
    std::array<roomtone::Lighting, kCapPoints> l;
    for (int i = 0; i < n; ++i)
      if (!c.est.lightAt(static_cast<size_t>(i), l[i])) {
        // Made for another grid (PROBES changed, or the stage's size): the
        // last answer until the next make.
        if (c.haveLast && c.last.grid.count() != n)
          break;
        c.haveLast = false;
        return false;
      } else if (i + 1 == n) {
        CapLight& r = c.last;
        r.grid = grid;
        for (int k = 0; k < n; ++k)
          r.sh[order[k]] = l[k].sh;
        for (int k = 0; k < 2; ++k) {
          const roomtone::Light& li = l[0].lights[k];
          r.dir[k] = {li.dir.x, li.dir.y, li.dir.z};
          r.col[k] = li.color;
          r.placed[k] = li.placed && li.range > 0.f;
          r.pos[k] = {li.pos.x, li.pos.y, li.pos.z};
          r.range[k] = li.range;
        }
        r.mean = l[0].mean;
        r.level = l[0].level;
        c.haveLast = true;
      }
  }
  if (!c.haveLast)
    return false;
  out = c.last;
  return true;
}

#ifdef XR_META_environment_depth
// The room capture's depth (AURORA_XR_ROOM_DEPTH, on by default where the
// runtime offers XR_META_environment_depth): while a capture runs, the
// provider runs, and up to kRoomDepthHz images a second are read back (a
// copy into a host buffer) and handed to Roomtone, view 0 (the left eye,
// by the left camera) at 80x80, so the room's surfaces get a place
// (docs/xr-room-depth-plan.md). Measured on a Quest 3 (e7df6af): 320x320 a
// view, D16, nearZ / (1 - d) with rows from the bottom, 94x99 degrees; the
// first image 128 ms after a 9 ms start; 0.25-0.3 ms an acquire; no
// measurable cost in a four-CPU fight.
constexpr int64_t kRoomDepthNs = 100'000'000;
constexpr uint32_t kRoomDepthGrid = 80;

struct RoomDepth {
  bool tried = false, ok = false, running = false, first = false;
  XrEnvironmentDepthProviderMETA provider = XR_NULL_HANDLE;
  XrEnvironmentDepthSwapchainMETA swapchain = XR_NULL_HANDLE;
  uint32_t w = 0, h = 0;
  std::vector<VkImage> images;
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  const uint16_t* mapped = nullptr;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  bool pending = false; // a read-back in flight
  XrEnvironmentDepthImageMETA copied{XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_META};
  int64_t copiedNs = 0, startNs = 0, acquireAtNs = 0, logNs = 0;
  uint32_t acquired = 0, unavailable = 0, failed = 0, fed = 0;
  int64_t acquireNs = 0, ageNs = 0;
  std::vector<uint16_t> small;
  PFN_xrCreateEnvironmentDepthProviderMETA createProvider = nullptr;
  PFN_xrStartEnvironmentDepthProviderMETA start = nullptr;
  PFN_xrStopEnvironmentDepthProviderMETA stop = nullptr;
  PFN_xrCreateEnvironmentDepthSwapchainMETA createSwapchain = nullptr;
  PFN_xrEnumerateEnvironmentDepthSwapchainImagesMETA enumerateImages = nullptr;
  PFN_xrGetEnvironmentDepthSwapchainStateMETA getState = nullptr;
  PFN_xrAcquireEnvironmentDepthImageMETA acquire = nullptr;
  PFN_xrSetEnvironmentDepthHandRemovalMETA setHandRemoval = nullptr;
} g_depth;

bool room_depth_init() {
  auto& d = g_depth;
  d.createProvider = xr_proc<PFN_xrCreateEnvironmentDepthProviderMETA>("xrCreateEnvironmentDepthProviderMETA");
  d.start = xr_proc<PFN_xrStartEnvironmentDepthProviderMETA>("xrStartEnvironmentDepthProviderMETA");
  d.stop = xr_proc<PFN_xrStopEnvironmentDepthProviderMETA>("xrStopEnvironmentDepthProviderMETA");
  d.createSwapchain = xr_proc<PFN_xrCreateEnvironmentDepthSwapchainMETA>("xrCreateEnvironmentDepthSwapchainMETA");
  d.enumerateImages =
      xr_proc<PFN_xrEnumerateEnvironmentDepthSwapchainImagesMETA>("xrEnumerateEnvironmentDepthSwapchainImagesMETA");
  d.getState = xr_proc<PFN_xrGetEnvironmentDepthSwapchainStateMETA>("xrGetEnvironmentDepthSwapchainStateMETA");
  d.acquire = xr_proc<PFN_xrAcquireEnvironmentDepthImageMETA>("xrAcquireEnvironmentDepthImageMETA");
  d.setHandRemoval = xr_proc<PFN_xrSetEnvironmentDepthHandRemovalMETA>("xrSetEnvironmentDepthHandRemovalMETA");
  if (!d.createProvider || !d.start || !d.stop || !d.createSwapchain || !d.enumerateImages || !d.getState ||
      !d.acquire) {
    Log.warn("Room depth: the extension's functions are missing");
    return false;
  }
  XrEnvironmentDepthProviderCreateInfoMETA pci{XR_TYPE_ENVIRONMENT_DEPTH_PROVIDER_CREATE_INFO_META};
  if (const XrResult r = d.createProvider(B.session, &pci, &d.provider); XR_FAILED(r)) {
    Log.warn("Room depth: xrCreateEnvironmentDepthProviderMETA failed: {}", static_cast<int>(r));
    return false;
  }
  XrEnvironmentDepthSwapchainCreateInfoMETA sci{XR_TYPE_ENVIRONMENT_DEPTH_SWAPCHAIN_CREATE_INFO_META};
  if (const XrResult r = d.createSwapchain(d.provider, &sci, &d.swapchain); XR_FAILED(r)) {
    Log.warn("Room depth: xrCreateEnvironmentDepthSwapchainMETA failed: {}", static_cast<int>(r));
    return false;
  }
  XrEnvironmentDepthSwapchainStateMETA st{XR_TYPE_ENVIRONMENT_DEPTH_SWAPCHAIN_STATE_META};
  if (XR_FAILED(d.getState(d.swapchain, &st)) || st.width < kRoomDepthGrid || st.height < kRoomDepthGrid)
    return false;
  d.w = st.width, d.h = st.height;
  uint32_t n = 0;
  if (XR_FAILED(d.enumerateImages(d.swapchain, 0, &n, nullptr)) || n == 0)
    return false;
  std::vector<XrSwapchainImageVulkan2KHR> imgs(n, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR});
  if (XR_FAILED(d.enumerateImages(d.swapchain, n, &n, reinterpret_cast<XrSwapchainImageBaseHeader*>(imgs.data()))))
    return false;
  for (const auto& i : imgs)
    d.images.push_back(i.image);
  // Both views, D16.
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = static_cast<VkDeviceSize>(d.w) * d.h * 2 * 2;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VK_TRY(vkCreateBuffer(B.dev, &bci, nullptr, &d.buf));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(B.dev, d.buf, &mr);
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = mr.size;
  mai.memoryTypeIndex =
      find_memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (mai.memoryTypeIndex == UINT32_MAX)
    return false;
  VK_TRY(vkAllocateMemory(B.dev, &mai, nullptr, &d.mem));
  VK_TRY(vkBindBufferMemory(B.dev, d.buf, d.mem, 0));
  void* p = nullptr;
  VK_TRY(vkMapMemory(B.dev, d.mem, 0, VK_WHOLE_SIZE, 0, &p));
  d.mapped = static_cast<const uint16_t*>(p);
  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = B.pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  VK_TRY(vkAllocateCommandBuffers(B.dev, &cai, &d.cmd));
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VK_TRY(vkCreateFence(B.dev, &fci, nullptr, &d.fence));
  // The hands left out (the room behind them), as the camera's samples of
  // a hand land on whatever depth is behind it anyway.
  bool handRemoval = false;
  if (d.setHandRemoval) {
    XrEnvironmentDepthHandRemovalSetInfoMETA hi{XR_TYPE_ENVIRONMENT_DEPTH_HAND_REMOVAL_SET_INFO_META};
    hi.enabled = XR_TRUE;
    handRemoval = XR_SUCCEEDED(d.setHandRemoval(d.provider, &hi));
  }
  Log.info("Room depth: {}x{} depth, {} images, hand removal {}", d.w, d.h, n, handRemoval ? "on" : "off");
  return true;
}

// The read-back's view 0, at kRoomDepthGrid a side, to the capture.
void room_depth_feed() {
  auto& d = g_depth;
  const auto& im = d.copied;
  const XrEnvironmentDepthImageViewMETA& v = im.views[0];
  const uint32_t step = d.w / kRoomDepthGrid, stepY = d.h / kRoomDepthGrid;
  d.small.resize(kRoomDepthGrid * kRoomDepthGrid);
  for (uint32_t y = 0; y < kRoomDepthGrid; ++y)
    for (uint32_t x = 0; x < kRoomDepthGrid; ++x)
      d.small[y * kRoomDepthGrid + x] = d.mapped[(y * stepY + stepY / 2) * d.w + x * step + step / 2];
  roomtone::DepthImage di;
  di.depth = d.small.data();
  di.width = di.height = di.stride = static_cast<int>(kRoomDepthGrid);
  di.nearZ = im.nearZ;
  di.farZ = im.farZ;
  di.tanLeft = std::tan(v.fov.angleLeft), di.tanRight = std::tan(v.fov.angleRight);
  di.tanUp = std::tan(v.fov.angleUp), di.tanDown = std::tan(v.fov.angleDown);
  di.view = {rt(v.pose.orientation), rt(v.pose.position)};
  di.timeNs = d.copiedNs;
  // For the capture thread's next frame (the estimate may be mid-make).
  std::lock_guard lock{g_capture.mutex};
  g_capture.depthPx = d.small;
  g_capture.depth = di;
  g_capture.depth.depth = nullptr;
  g_capture.depthPending = true;
  ++d.fed;
}

#define VK_TRY_VOID(x)                                                                                                 \
  do {                                                                                                                 \
    VkResult r_ = (x);                                                                                                 \
    if (r_ != VK_SUCCESS) {                                                                                            \
      Log.error("{} failed: VkResult {}", #x, static_cast<int>(r_));                                                   \
      d.ok = false;                                                                                                    \
      return;                                                                                                          \
    }                                                                                                                  \
  } while (0)

// Each frame between xrBeginFrame and xrEndFrame.
void room_depth_frame(XrTime displayTime) {
  auto& d = g_depth;
  if (!B.envDepth)
    return;
  if (!d.tried) {
    d.tried = true;
    d.ok = room_depth_init();
  }
  if (!d.ok)
    return;
  int captureMode;
  {
    std::lock_guard lock{g_capture.mutex};
    captureMode = capture_mode_locked();
  }
  const bool want = captureMode != 0;
  const int64_t now = steady_ns();
  if (want != d.running) {
    const int64_t t0 = steady_ns();
    const XrResult r = want ? d.start(d.provider) : d.stop(d.provider);
    Log.info("Room depth: provider {} ({}, {:.2f} ms)", want ? "started" : "stopped", static_cast<int>(r),
             (steady_ns() - t0) / 1e6);
    if (XR_FAILED(r))
      return;
    d.running = want;
    d.first = true;
    d.startNs = now;
  }
  if (d.pending && vkGetFenceStatus(B.dev, d.fence) == VK_SUCCESS) {
    d.pending = false;
    room_depth_feed();
  }
  if (!d.running || d.pending || now - d.acquireAtNs < kRoomDepthNs)
    return;
  d.acquireAtNs = now;
  XrEnvironmentDepthImageAcquireInfoMETA ai{XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_ACQUIRE_INFO_META};
  ai.space = B.space;
  ai.displayTime = displayTime;
  XrEnvironmentDepthImageTimestampMETA ts{XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_TIMESTAMP_META};
  XrEnvironmentDepthImageMETA im{XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_META};
  im.next = &ts;
  im.views[0].type = im.views[1].type = XR_TYPE_ENVIRONMENT_DEPTH_IMAGE_VIEW_META;
  const XrResult r = d.acquire(d.provider, &ai, &im);
  d.acquireNs += steady_ns() - now;
  if (r == XR_ENVIRONMENT_DEPTH_NOT_AVAILABLE_META) {
    ++d.unavailable;
  } else if (XR_FAILED(r)) {
    ++d.failed;
  } else if (im.swapchainIndex < d.images.size()) {
    ++d.acquired;
    if (ts.captureTime > 0)
      d.ageNs += displayTime - ts.captureTime;
    if (d.first) {
      d.first = false;
      Log.info("Room depth: first image {:.0f} ms after the start", (now - d.startNs) / 1e6);
    }
    VK_TRY_VOID(vkResetFences(B.dev, 1, &d.fence));
    VK_TRY_VOID(vkResetCommandBuffer(d.cmd, 0));
    VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_TRY_VOID(vkBeginCommandBuffer(d.cmd, &cbi));
    const VkImage img = d.images[im.swapchainIndex];
    VkImageMemoryBarrier b = barrier(img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                     VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_QUEUE_FAMILY_IGNORED,
                                     VK_QUEUE_FAMILY_IGNORED, VK_IMAGE_ASPECT_DEPTH_BIT);
    vkCmdPipelineBarrier(d.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 2};
    region.imageExtent = {d.w, d.h, 1};
    vkCmdCopyImageToBuffer(d.cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, d.buf, 1, &region);
    b = barrier(img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                VK_IMAGE_ASPECT_DEPTH_BIT);
    vkCmdPipelineBarrier(d.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    VkBufferMemoryBarrier bb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bb.srcQueueFamilyIndex = bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bb.buffer = d.buf;
    bb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(d.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &bb, 0,
                         nullptr);
    VK_TRY_VOID(vkEndCommandBuffer(d.cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &d.cmd;
    VK_TRY_VOID(vkQueueSubmit(B.queue, 1, &si, d.fence));
    d.copied = im;
    d.copiedNs = now;
    d.pending = true;
  }
  if (now - d.logNs >= 30'000'000'000) {
    const uint32_t n = d.acquired + d.unavailable + d.failed;
    if (d.logNs != 0 && n > 0)
      Log.info("Room depth: {} acquires (new {}, not available {}, failed {}), {:.3f} ms each, images {:.1f} ms old, "
               "{} fed",
               n, d.acquired, d.unavailable, d.failed, d.acquireNs / 1e6 / n,
               d.acquired > 0 ? d.ageNs / 1e6 / d.acquired : -1.0, d.fed);
    d.logNs = now;
    d.acquired = d.unavailable = d.failed = d.fed = 0;
    d.acquireNs = d.ageNs = 0;
  }
}
#undef VK_TRY_VOID
#else
void room_depth_frame(XrTime) {}
#endif


// The Light page's ROOM segment and RECAPTURE while a capture runs: the
// seconds left (-1: none running).
int capture_seconds_left() {
  std::lock_guard lock{g_capture.mutex};
  if (!g_capture.requested)
    return -1;
  const int64_t left = g_capture.untilNs == 0 ? kCaptureNs : std::max<int64_t>(g_capture.untilNs - steady_ns(), 0);
  return static_cast<int>((left + 999'999'999) / 1'000'000'000);
}

// The sky gradient's SH L2 irradiance, divided by pi (so a uniform sky of
// colour c lights a surface c, as a GX ambient does), each coefficient
// times its basis constant: the shader evaluates the bare polynomials
// (shader.cpp, xr_room_diffuse), in this order.
std::array<std::array<float, 3>, 9> room_sh(const RoomPreset& p) {
  std::array<std::array<double, 3>, 9> acc{};
  constexpr int kTheta = 64, kPhi = 128;
  constexpr double kPi = 3.14159265358979323846;
  for (int i = 0; i < kTheta; ++i) {
    const double theta = (i + 0.5) * kPi / kTheta; // from up
    const double y = std::cos(theta), r = std::sin(theta);
    const double dOmega = r * (kPi / kTheta) * (2.0 * kPi / kPhi);
    std::array<double, 3> col;
    for (int c = 0; c < 3; ++c)
      col[c] = y >= 0.0 ? p.horizon[c] + (p.up[c] - p.horizon[c]) * y : p.horizon[c] + (p.down[c] - p.horizon[c]) * -y;
    for (int j = 0; j < kPhi; ++j) {
      const double phi = (j + 0.5) * 2.0 * kPi / kPhi;
      const double x = r * std::cos(phi), z = r * std::sin(phi);
      const std::array<double, 9> poly{1.0, y, z, x, x * y, y * z, 3.0 * z * z - 1.0, x * z, x * x - y * y};
      constexpr std::array<double, 9> basis{0.282095, 0.488603, 0.488603, 0.488603, 1.092548,
                                            1.092548, 0.315392, 1.092548, 0.546274};
      for (int k = 0; k < 9; ++k)
        for (int c = 0; c < 3; ++c)
          acc[k][c] += col[c] * basis[k] * poly[k] * dOmega;
    }
  }
  // Cosine lobe per band over pi: 1, 2/3, 1/4; and the basis constant again.
  constexpr std::array<double, 9> lobe{1.0, 2.0 / 3, 2.0 / 3, 2.0 / 3, 0.25, 0.25, 0.25, 0.25, 0.25};
  constexpr std::array<double, 9> basis{0.282095, 0.488603, 0.488603, 0.488603, 1.092548,
                                        1.092548, 0.315392, 1.092548, 0.546274};
  std::array<std::array<float, 3>, 9> out;
  for (int k = 0; k < 9; ++k)
    for (int c = 0; c < 3; ++c)
      out[k][c] = static_cast<float>(acc[k][c] * lobe[k] * basis[k]);
  return out;
}

std::string room_light_value(const RoomLight& r) {
  return fmt::format("{},{:.2f},{:.2f},{:.2f},{:.2f},{:.0f},{},{},{:.2f},{:.2f},{:.2f},{}", r.source, r.fighters,
                     r.stage, r.unlit, r.strength, r.yaw, r.shading, r.view, r.delight, r.gloss, r.keys, r.probes);
}

// The eye pipelines' variant bits (gfx::set_xr_room_light).
void room_light_apply(const RoomLight& r) {
  gfx::set_xr_room_light(r.source == 0 ? 0u
                                         : 1u | (r.shading == 1 ? 2u : 0u) | (static_cast<uint32_t>(r.view) & 3u) << 2 |
                                               (r.shading >= 2 ? 16u : 0u) | (r.shading == 3 ? 32u : 0u) |
                                               (r.gloss > 0.f && stage_gloss(g_roomStage) > 0.f ? 64u : 0u));
}

void init_room_light() {
  std::lock_guard lock{g_room.mutex};
  for (size_t i = 0; i < kRoomPresets.size(); ++i)
    g_room.sh[i] = room_sh(kRoomPresets[i]);
  RoomLight r;
  if (const char* v = std::getenv("AURORA_XR_ROOM_LIGHT")) {
    float f[12] = {0.f, r.fighters, r.stage, r.unlit, r.strength, r.yaw, 0.f, 0.f, r.delight, r.gloss, r.keys, 0.f};
    std::sscanf(v, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", &f[0], &f[1], &f[2], &f[3], &f[4], &f[5], &f[6], &f[7],
                &f[8], &f[9], &f[10], &f[11]);
    r.source = std::clamp(static_cast<int>(f[0]), 0, kRoomSourceCaptured);
    r.fighters = std::clamp(f[1], 0.f, 1.f);
    r.stage = std::clamp(f[2], 0.f, 1.f);
    r.unlit = std::clamp(f[3], 0.f, 1.f);
    r.strength = std::clamp(f[4], 0.f, 2.f);
    r.yaw = std::fmod(std::fmod(f[5], 360.f) + 360.f, 360.f);
    r.shading = std::clamp(static_cast<int>(f[6]), 0, 3);
    r.view = std::clamp(static_cast<int>(f[7]), 0, 3);
    r.delight = std::clamp(f[8], 0.f, 1.f);
    r.gloss = std::clamp(f[9], 0.f, 1.f);
    r.keys = std::clamp(f[10], 0.f, 2.f);
    r.probes = std::clamp(static_cast<int>(f[11]), 0, 1);
    Log.info("Room light: AURORA_XR_ROOM_LIGHT={}", room_light_value(r));
  }
  if (r.source == kRoomSourceCaptured)
    capture_start();
  // AURORA_XR_ROOM_REPLAY=recording (AURORA_XR_ROOM_RECORD's, a desktop's
  // test of a headset's room): the capture made from it, no camera needed.
  if (const char* path = std::getenv("AURORA_XR_ROOM_REPLAY"); path && *path && r.source == kRoomSourceCaptured) {
    auto& c = g_capture;
    uint32_t frames = 0;
    {
      std::lock_guard lock{c.estMutex};
      roomtone::replay(path, c.est, [&](const roomtone::ReplayEvent& e) { frames += e.kind == roomtone::ReplayEvent::Frame; });
      c.est.make();
      Log.info("Room capture: replayed {} ({} frames): {}", path, frames, c.est.summary());
    }
    std::lock_guard lock{c.mutex};
    c.requested = false;
    c.any = c.est.made();
#ifndef __ANDROID__
    // The capture poll Android's RoomCapture runs, so the light is made
    // again as the stage moves (or its probe grid changes), as on a headset.
    // Until the XR thread exits.
    g_replayPoll = std::thread([] {
      while (!g_replayPollStop.load()) {
        aurora_xr_room_capture_wanted();
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
      }
    });
#endif
  }
  g_room.value = g_room.defaults = r;
  room_light_apply(r);
}

// What render_eyes puts in the eye uniform's room fields, for a frame.
struct RoomFrame {
  bool on = false;
  float w[4]{}, up[4]{};
  // The SH at each point of the grid (the room), for the shader to blend
  // between by a vertex's or pixel's place.
  float sh[kCapPoints][9][4]{};
  ProbeGrid grid;
  std::array<float, 3> dir[2]{}; // in the room, from the reference point
  float col[2][4]{};
  // A light with a place (the room) and the distance from the reference
  // point its colour applies at: the shader's point light (range 0: directional).
  std::array<XrVector3f, 2> pos{};
  float range[2]{};
  float delight = 0.f;
  float gloss[2]{}; // fighters', the stage's
};

// The points the light is made at for the stage's placement (`arena`):
// game units into the room (kCapPoints' comment).
ProbeGrid probe_grid(const std::array<float, 16>& arena, bool line) {
  std::array<float, 3> ctr;
  Footprint fp;
  float top;
  {
    std::lock_guard lock{g_arenaMutex};
    ctr = g_arenaCenter;
    fp = g_footprint;
    top = g_stageTop;
  }
  const auto room = [&](float x, float y, float z) {
    return XrVector3f{arena[0] * x + arena[1] * y + arena[2] * z + arena[3],
                      arena[4] * x + arena[5] * y + arena[6] * z + arena[7],
                      arena[8] * x + arena[9] * y + arena[10] * z + arena[11]};
  };
  const auto along = [&](float x, float y, float z) { return room(x, y, z) - room(0.f, 0.f, 0.f); };
  // 75 units either side without a footprint.
  const float x0 = fp.valid ? fp.xmin : ctr[0] - 75.f, x1 = fp.valid ? fp.xmax : ctr[0] + 75.f;
  const float y0 = ctr[1] + kCapLift;
  ProbeGrid g;
  if (line) {
    g.n = {3, 1, 1};
    g.origin = room(x0 + 0.2f * (x1 - x0), y0, ctr[2]);
    g.axis[0] = along(0.6f * (x1 - x0), 0.f, 0.f);
    g.ref = 1;
    return g;
  }
  const float y1 = std::isnan(top) ? y0 : std::max(top + kCapLift, y0);
  const float z0 = fp.valid ? fp.zmin : ctr[2], z1 = fp.valid ? fp.zmax : ctr[2];
  const std::array<float, 3> span{x1 - x0, y1 - y0, z1 - z0};
  const float scale = vlen(along(1.f, 0.f, 0.f)); // metres a game unit
  for (int a = 0; a < 3; ++a) {
    const int n = static_cast<int>(std::lround(span[a] * scale / kCapSpacing)) + 1;
    g.n[a] = std::clamp(n, kCapGridMin[a], kCapGridMax[a]);
    if (span[a] <= 0.f)
      g.n[a] = 1;
  }
  g.origin = room(x0, y0, z0);
  g.axis = {g.n[0] > 1 ? along(span[0], 0.f, 0.f) : XrVector3f{}, g.n[1] > 1 ? along(0.f, span[1], 0.f) : XrVector3f{},
            g.n[2] > 1 ? along(0.f, 0.f, span[2]) : XrVector3f{}};
  // The middle across, the lowest, the nearest the middle deep.
  g.ref = (g.n[0] - 1) / 2 + g.n[0] * g.n[1] * ((g.n[2] - 1) / 2);
  return g;
}

// The SH blended at `p` (the room) among the grid's points' `sh`, as the
// shader blends them (xr_room_sh_eo): trilinear in the grid's cell.
void probe_blend(const ProbeGrid& g, const float (*sh)[9][4], const XrVector3f& p, float out[9][4]) {
  std::array<int, 3> c0{};
  std::array<float, 3> f{};
  for (int a = 0; a < 3; ++a) {
    const float l2 = vdot(g.axis[a], g.axis[a]);
    const float t = g.n[a] > 1 && l2 > 1e-12f ? std::clamp(vdot(p - g.origin, g.axis[a]) / l2, 0.f, 1.f) : 0.f;
    const float u = t * static_cast<float>(g.n[a] - 1);
    c0[a] = std::min(static_cast<int>(u), g.n[a] - 1);
    f[a] = u - static_cast<float>(c0[a]);
  }
  for (int k = 0; k < 9; ++k)
    for (int c = 0; c < 4; ++c)
      out[k][c] = 0.f;
  for (int corner = 0; corner < 8; ++corner) {
    float w = 1.f;
    std::array<int, 3> ci{};
    for (int a = 0; a < 3; ++a) {
      const bool up = (corner >> a) & 1;
      w *= up ? f[a] : 1.f - f[a];
      ci[a] = c0[a] + (up ? 1 : 0);
    }
    if (w <= 0.f)
      continue;
    const int i = (ci[2] * g.n[1] + ci[1]) * g.n[0] + ci[0];
    for (int k = 0; k < 9; ++k)
      for (int c = 0; c < 3; ++c)
        out[k][c] += w * sh[i][k][c];
  }
}

// `arena`: the stage's placement (its rotation about y and scale).
RoomFrame room_frame(const std::array<float, 16>& arena) {
  RoomFrame f;
  RoomLight r;
  CapLight cap;
  {
    std::lock_guard lock{g_room.mutex};
    r = g_room.value;
    if (r.source == 0)
      return f;
    if (r.source != kRoomSourceCaptured)
      cap.sh.fill(g_room.sh[r.source - 1]);
  }
  cap.grid = probe_grid(arena, r.probes == 1);
  if (static std::array<int, 3> logged{}; cap.grid.n != logged) {
    logged = cap.grid.n;
    Log.info("Room light probes: {} {}x{}x{} ({} points, {:.2f} m across)", r.probes == 1 ? "LINE" : "GRID",
             logged[0], logged[1], logged[2], cap.grid.count(), vlen(cap.grid.axis[0]));
  }
  const bool captured = r.source == kRoomSourceCaptured;
  if (captured && !capture_light(cap.grid, cap))
    return f;
  const float capMean = cap.mean, capLevel = captured ? cap.level : 1.f;
  // The captured room's level (ABL, by STRENGTH), and the fighters' scale:
  // the room's mean light to kCapFighters times it. Roomtone's light is in
  // the units of its reference (the room as the head saw it: up-facing 1,
  // capMean its mean), so content nearer a light is lit brighter than the
  // reference and farther dimmer.
  const float level = r.strength * capLevel;
  const float gain = captured ? level * kCapFighters / std::max(capMean, 1e-3f) : r.strength;
  const RoomPreset& p = kRoomPresets[captured ? 0 : r.source - 1];
  f.on = true;
  f.w[0] = r.fighters, f.w[1] = r.stage, f.w[2] = r.unlit;
  f.delight = r.delight;
  f.gloss[0] = r.gloss * kFighterGloss;
  f.gloss[1] = r.gloss * stage_gloss(g_roomStage);
  for (int i = 0; i < cap.grid.count(); ++i)
    for (int k = 0; k < 9; ++k)
      for (int c = 0; c < 3; ++c)
        f.sh[i][k][c] = cap.sh[i][k][c] * gain;
  f.grid = cap.grid;
  // The lights' directions: turned by YAW about the stage's up, then by the
  // stage's placement into the room.
  const float yaw = r.yaw * 0.017453292f, cy = std::cos(yaw), sy = std::sin(yaw);
  const std::array<float, 3>* dirs[2] = {&p.keyDir, &p.fillDir};
  const std::array<float, 3>* cols[2] = {&p.key, &p.fill};
  for (int l = 0; l < 2; ++l) {
    const auto& d = *dirs[l];
    const float gx = cy * d[0] + sy * d[2], gy = d[1], gz = -sy * d[0] + cy * d[2];
    std::array<float, 3> v{arena[0] * gx + arena[1] * gy + arena[2] * gz, arena[4] * gx + arena[5] * gy + arena[6] * gz,
                           arena[8] * gx + arena[9] * gy + arena[10] * gz};
    const float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    for (float& c : v)
      c = len > 1e-12f ? c / len : 0.f;
    f.dir[l] = v;
    for (int c = 0; c < 3; ++c)
      f.col[l][c] = (*cols[l])[c] * r.strength * r.keys;
  }
  if (captured)
    for (int l = 0; l < 2; ++l) {
      f.dir[l] = cap.dir[l];
      for (int c = 0; c < 3; ++c)
        f.col[l][c] = cap.col[l][c] * gain * r.keys;
      f.pos[l] = cap.pos[l];
      f.range[l] = cap.placed[l] ? cap.range[l] : 0.f;
    }
  // An up-facing surface's light at the reference point, for unlit draws
  // (as xr_room_diffuse at up).
  const auto& ref = f.sh[f.grid.ref];
  for (int c = 0; c < 3; ++c) {
    float e = ref[0][c] + ref[1][c] - ref[6][c] - ref[8][c];
    for (int l = 0; l < 2; ++l)
      e += std::max(f.dir[l][1], 0.f) * f.col[l][c];
    f.up[c] = std::clamp(e, 0.f, 1.f);
  }
  // Unlit draws lit by normals: relative to an up-facing surface's light
  // (its brightness, as STRENGTH leaves it at 1), as baked stages are lit
  // from above: their tops keep the baked brightness, taking the room's
  // colour, and the rest the room's shading. (The sphere's average took in
  // the dark floor: tops came out twice as bright.)
  {
    const float s = std::max(r.strength, 1e-3f);
    const float upLum = 0.2126f * f.up[0] + 0.7152f * f.up[1] + 0.0722f * f.up[2];
    // The captured room's: its tops the baked colour times the level, times
    // the light from above at the stage over the reference's (Roomtone's 1;
    // divided by the stage's own, a light's falloff cancelled on the tops).
    f.w[3] = captured ? level / std::max(gain, 1e-3f) : 1.f / std::max(upLum / s, 0.05f);
  }
  return f;
}

// ---------------------------------------------------------------- XR thread: the dev tools panel
//
// For tuning and A/Bs on the headset (AURORA_XR_DEV_TOOLS=1, a debug aid):
// while a fight is paused, a panel floats to the right of where the head
// looked as the pause began, with a page per tab. A laser on the panel and
// the trigger (a pinch for a tracked hand) pick a tab or a choice and drag
// a slider; Reset gives the page's settings back as they were without it.
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
// Light, which the eye shaders apply (room lighting, above): the room's
// source (off or a preset), the blend toward it for fighters, stage and
// unlit draws, its strength, its key lights' strength (KEY, on top), the
// turn of its lights about the stage, the
// room's part per vertex or per pixel, and a debug view. Reset puts back
// the launch settings; each change is logged as an AURORA_XR_ROOM_LIGHT
// value.
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

constexpr int kPanelW = 768, kPanelH = 520;
constexpr int kTrackX0 = 170, kTrackX1 = 590, kTrackH = 30;
constexpr int kRowY0 = 76, kRowStep = 86;
constexpr int kResetX0 = 604, kResetX1 = 750, kResetY0 = 10, kResetY1 = 52;
enum { kHue, kTint, kLight, kSat, kSliders };
constexpr std::array<const char*, kSliders> kSliderNames{"HUE", "TINT", "LIGHT", "SAT"};
constexpr std::array<float, kSliders> kSliderMax{360.f, 1.f, 1.f, 1.5f};

// The pages, as tabs left of Reset.
enum { kPageTint, kPageAudio, kPageMix, kPageLight, kPages };
constexpr std::array<const char*, kPages> kPageNames{"TINT", "AUDIO", "STAGE", "LIGHT"};
constexpr int kTabX0 = 10, kTabW = 118, kTabGap = 8;

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

// The Light page (room lighting): a row each, choices or sliders.
enum {
  kRlSource,
  kRlFighters,
  kRlStage,
  kRlUnlit,
  kRlDelight,
  kRlGloss,
  kRlStrength,
  kRlKeys,
  kRlYaw,
  kRlShading,
  kRlView,
  kRlProbes,
  kRlRows
};
constexpr int kLightRowY0 = 70, kLightRowStep = 34;
constexpr std::array<const char*, kRlRows> kLightNames{"SOURCE", "FIGHTERS", "STAGE",   "UNLIT", "DELIGHT", "GLOSS",
                                                       "STRENGTH", "KEY",    "YAW",     "SHADING", "VIEW", "PROBES"};
constexpr std::array<const char*, 4> kShadingNames{"VERTEX", "PIXEL", "FACET", "SMOOTH"};
constexpr std::array<const char*, 4> kViewNames{"OFF", "LIT", "CLASS", "NORMALS"};
constexpr std::array<const char*, 2> kProbeNames{"GRID", "LINE"};

// What's under a laser: a tint slider (kHue..kSat), an Audio row (kAudio +
// row), a tab (kTab + page), Reset, a Stage channel (kMixChan + channel),
// its level, its graph or its Q, Recapture, a Light row (kLightRow + row),
// or -1.
constexpr int kAudio = 10, kTab = 20, kReset = 30, kMixChan = 40, kMixLevel = 50, kMixGraph = 51, kMixQ = 52,
              kRecapture = 55, kLightRow = 60;
// Recapture, under the Light page's rows: a new room capture (ROOM chosen).
constexpr int kRecaptureY = kLightRowY0 + kRlRows * kLightRowStep;

struct DevPanel {
  bool shown = false;
  XrPosef pose{{0.f, 0.f, 0.f, 1.f}, {}};
  float width = 0.36f;
  int page = -1; // kPageTint or kPageAudio; -1 until first opened
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

void draw_header(Canvas& c) {
  const Rgba text{232, 236, 242, 255}, dim{150, 156, 168, 255};
  for (int i = 0; i < kPages; ++i) {
    const int x0 = kTabX0 + i * (kTabW + kTabGap);
    const bool on = P.page == i;
    c.fill(x0, kResetY0, x0 + kTabW, kResetY1,
           on ? Rgba{70, 110, 150, 255} : P.hover == kTab + i ? Rgba{60, 66, 80, 255} : Rgba{36, 40, 50, 255});
    const int w = static_cast<int>(std::strlen(kPageNames[i])) * 18 - 3;
    c.text(x0 + (kTabW - w) / 2, kResetY0 + 11, 3, kPageNames[i], on ? text : dim);
  }
  if (P.page == kPageTint)
    c.text(kTabX0 + kPages * (kTabW + kTabGap) + 8, kResetY0 + 11, 3, fmt::format("0x{:02X}", P.stage), dim);
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

// The Light page's rows: a choice's names (none for a slider), a slider's
// range and its value.
std::vector<const char*> light_choices(int row) {
  if (row == kRlSource) {
    std::vector<const char*> v{"OFF"};
    for (const auto& p : kRoomPresets)
      v.push_back(p.name);
    v.push_back("ROOM"); // kRoomSourceCaptured
    return v;
  }
  if (row == kRlShading)
    return {kShadingNames.begin(), kShadingNames.end()};
  if (row == kRlView)
    return {kViewNames.begin(), kViewNames.end()};
  if (row == kRlProbes)
    return {kProbeNames.begin(), kProbeNames.end()};
  return {};
}
float light_max(int row) { return row == kRlStrength || row == kRlKeys ? 2.f : row == kRlYaw ? 360.f : 1.f; }
float& light_slider(RoomLight& r, int row) {
  return row == kRlFighters ? r.fighters : row == kRlStage ? r.stage : row == kRlUnlit ? r.unlit
                                         : row == kRlDelight  ? r.delight
                                         : row == kRlGloss    ? r.gloss
                                         : row == kRlStrength ? r.strength
                                         : row == kRlKeys     ? r.keys : r.yaw;
}

void light_log() {
  RoomLight r;
  {
    std::lock_guard lock{g_room.mutex};
    r = g_room.value;
  }
  Log.info("Dev tools: stage 0x{:02X} room light: AURORA_XR_ROOM_LIGHT={}", P.stage, room_light_value(r));
}

void draw_light(Canvas& c) {
  const Rgba text{232, 236, 242, 255}, dim{150, 156, 168, 255}, lit{120, 200, 255, 255};
  RoomLight r;
  {
    std::lock_guard lock{g_room.mutex};
    r = g_room.value;
  }
  for (int row = 0; row < kRlRows; ++row) {
    const int y = kLightRowY0 + row * kLightRowStep;
    const int target = kLightRow + row;
    // The rest needs a source.
    const bool usable = row == kRlSource || r.source != 0;
    c.text(18, y + 4, 3, kLightNames[row],
           !usable ? Rgba{90, 94, 104, 255} : target == P.active ? lit : target == P.hover ? text : dim);
    const auto names = light_choices(row);
    if (names.empty()) {
      float& v = light_slider(r, row);
      const float t = std::clamp(v / light_max(row), 0.f, 1.f);
      const int kx = kTrackX0 + static_cast<int>(std::lround(t * (kTrackX1 - kTrackX0)));
      c.fill(kTrackX0, y + 10, kTrackX1, y + kTrackH - 10, {52, 56, 68, 255});
      c.fill(kTrackX0, y + 10, kx, y + kTrackH - 10, usable ? Rgba{70, 110, 150, 255} : Rgba{60, 64, 74, 255});
      if (row == kRlStrength || row == kRlKeys) { // 1x
        const int k1 = kTrackX0 + (kTrackX1 - kTrackX0) / 2;
        c.fill(k1 - 1, y + 4, k1 + 1, y + kTrackH - 4, {150, 156, 168, 255});
      }
      c.fill(kx - 6, y - 7, kx + 6, y + kTrackH + 7, {10, 10, 12, 255});
      c.fill(kx - 3, y - 4, kx + 3, y + kTrackH + 4, target == P.active ? lit : Rgba{250, 250, 250, 255});
      c.text(kTrackX1 + 22, y + 4, 3, row == kRlYaw ? fmt::format("{:.0f}", v) : fmt::format("{:.2f}", v), text);
      continue;
    }
    const int n = static_cast<int>(names.size());
    const int chosen = row == kRlSource ? r.source : row == kRlShading ? r.shading : row == kRlProbes ? r.probes : r.view;
    const int w = (kChoiceX1 - kTrackX0) / n;
    for (int i = 0; i < n; ++i) {
      const int x0 = kTrackX0 + i * w;
      const bool on = i == chosen;
      c.fill(x0 + 2, y - 2, x0 + w - 2, y + kTrackH + 2,
             on ? (usable ? Rgba{70, 110, 150, 255} : Rgba{60, 64, 74, 255}) : Rgba{36, 40, 50, 255});
      // ROOM while it captures: the seconds left.
      std::string label = names[i];
      if (row == kRlSource && i == kRoomSourceCaptured)
        if (const int left = capture_seconds_left(); left >= 0) {
          label = fmt::format("{}S", left);
          P.dirty = true; // drawn again next frame
        }
      const int tw = static_cast<int>(label.size()) * 18 - 3;
      c.text(x0 + (w - tw) / 2, y + 4, 3, label.c_str(), on ? text : dim);
    }
  }
  const int left = capture_seconds_left();
  const std::string label = left >= 0 ? fmt::format("CAPTURING {}S", left) : "RECAPTURE";
  c.fill(kTrackX0 + 2, kRecaptureY - 4, kChoiceX1 - 2, kRecaptureY + kTrackH + 4,
         P.hover == kRecapture || left >= 0 ? Rgba{70, 110, 150, 255} : Rgba{52, 56, 68, 255});
  const int tw = static_cast<int>(label.size()) * 18 - 3;
  c.text(kTrackX0 + (kChoiceX1 - kTrackX0 - tw) / 2, kRecaptureY + 4, 3, label.c_str(), text);
}

void draw_panel(Canvas& c) {
  c.fill(0, 0, kPanelW, kPanelH, {18, 20, 26, 225});
  c.fill(0, 0, kPanelW, 3, {90, 96, 110, 255});
  draw_header(c);
  if (P.page == kPageTint)
    draw_tint(c);
  else if (P.page == kPageAudio)
    draw_audio(c);
  else if (P.page == kPageLight)
    draw_light(c);
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
    for (int i = 0; i < kPages; ++i) {
      const float x0 = static_cast<float>(kTabX0 + i * (kTabW + kTabGap));
      if (px >= x0 && px < x0 + kTabW)
        return kTab + i;
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
  if (P.page == kPageLight) {
    bool on;
    {
      std::lock_guard lock{g_room.mutex};
      on = g_room.value.source != 0;
    }
    for (int row = 0; row < kRlRows; ++row) {
      // Each row's band, the rows' step tall (they're close since PROBES).
      const float y = static_cast<float>(kLightRowY0 + row * kLightRowStep);
      constexpr float pad = static_cast<float>(kLightRowStep - kTrackH) / 2.f;
      if ((row == kRlSource || on) && px >= kTrackX0 - 24 && px < kChoiceX1 && py >= y - pad && py < y + kTrackH + pad)
        return kLightRow + row;
    }
    if (px >= kTrackX0 && px < kChoiceX1 && py >= kRecaptureY - 10 && py < kRecaptureY + kTrackH + 10)
      return kRecapture;
    return -1;
  }
  for (int row = 0; row < kAudioRows; ++row) {
    const float y = static_cast<float>(kAudioRowY0 + row * kAudioRowStep);
    if (px >= kTrackX0 - 24 && px < kChoiceX1 && py >= y - 18 && py < y + kTrackH + 18)
      return kAudio + row;
  }
  return -1;
}

bool light_is_slider(int target) {
  return target >= kLightRow && target < kLightRow + kRlRows && light_choices(target - kLightRow).empty();
}

// A press on a Light choice row at px: picks the segment under it.
void light_choose(int row, float px) {
  const int n = static_cast<int>(light_choices(row).size()), w = (kChoiceX1 - kTrackX0) / n;
  const int i = std::clamp(static_cast<int>((px - kTrackX0) / static_cast<float>(w)), 0, n - 1);
  RoomLight r;
  {
    std::lock_guard lock{g_room.mutex};
    RoomLight& v = g_room.value;
    if (row == kRlSource)
      v.source = i;
    else if (row == kRlShading)
      v.shading = i;
    else if (row == kRlProbes)
      v.probes = i;
    else
      v.view = i;
    r = v;
  }
  if (row == kRlSource && i == kRoomSourceCaptured)
    capture_resume();
  else if (row == kRlSource)
    capture_stop();
  room_light_apply(r);
  P.dirty = true;
  light_log();
}

// Recapture: ROOM chosen, its capture started over.
void light_recapture() {
  RoomLight r;
  {
    std::lock_guard lock{g_room.mutex};
    g_room.value.source = kRoomSourceCaptured;
    r = g_room.value;
  }
  capture_start();
  room_light_apply(r);
  P.dirty = true;
  light_log();
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
  const float t = std::clamp((px - kTrackX0) / static_cast<float>(kTrackX1 - kTrackX0), 0.f, 1.f);
  if (light_is_slider(target)) {
    const int row = target - kLightRow;
    const float step = row == kRlYaw ? 5.f : 0.05f;
    const float v = std::round(t * light_max(row) / step) * step;
    std::lock_guard lock{g_room.mutex};
    float& slot = light_slider(g_room.value, row);
    if (v != slot) {
      slot = v;
      P.dirty = true;
    }
    return;
  }
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
  if (P.page == kPageLight) {
    RoomLight r;
    {
      std::lock_guard lock{g_room.mutex};
      r = g_room.value = g_room.defaults;
    }
    room_light_apply(r);
    P.dirty = true;
    light_log();
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
  else if (light_is_slider(P.active))
    light_log();
}

// Each display frame, before the arena's grab: shows the panel while `show`,
// and points, presses and drags on it. Marks the hands on it (Hand::onPanel).
// `tintUsable`: the room's grade is on and the room shows.
void update_dev_panel(bool show, bool tintUsable) {
  for (Hand& hand : G.hands) {
    hand.onPanel = false;
    hand.panelHit = -1.f;
  }
  if (tintUsable != P.tintUsable) {
    P.tintUsable = tintUsable;
    P.dirty = true;
  }
  if (show && !P.shown) {
    // Opened: to the right of where the head looks, facing it.
    const XrVector3f fwd = qrot(B.headOrientation, {0.f, 0.f, -1.f});
    XrVector3f offset{0.3f, -0.15f, -0.55f};
    P.width = 0.36f;
    if (const char* v = std::getenv("AURORA_XR_DEV_TOOLS_POS"))
      std::sscanf(v, "%f,%f,%f,%f", &offset.x, &offset.y, &offset.z, &P.width);
    P.pose.position = B.head + rot_y(offset, std::atan2(-fwd.x, -fwd.z));
    P.pose.orientation = facing(P.pose.position, B.head);
    if (P.page < 0) {
      std::lock_guard lock{g_room.mutex};
      P.page = g_room.defaults.source != 0 ? kPageLight : tintUsable ? kPageTint : kPageAudio;
    }
    P.dirty = true;
    std::lock_guard lock{g_grade.mutex};
    P.stage = g_grade.stage;
    panel_from_grade(g_grade.grade);
  }
  if (!show && P.active >= 0)
    panel_released();
  if (!show)
    P.active = P.activeHand = -1;
  P.shown = show;
  g_devMixListening.store(show && P.page == kPageMix, std::memory_order_relaxed);
  if (!show)
    return;
  int hover = -1;
  for (int h = 0; h < 2; ++h) {
    Hand& hand = G.hands[h];
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
    } else if (panel_pressed(hand) && target == kRecapture) {
      light_recapture();
    } else if (panel_pressed(hand) && target >= kLightRow && target < kLightRow + kRlRows && !light_is_slider(target)) {
      light_choose(target - kLightRow, px);
    } else if (panel_pressed(hand) && target >= kMixChan && target < kMixChan + AURORA_XR_MIX_CHANNELS) {
      P.mixChannel = target - kMixChan;
      P.dirty = true;
    } else if (panel_pressed(hand) &&
               (target == kAudio + kMusic || target == kMixLevel || target == kMixGraph || target == kMixQ ||
                light_is_slider(target) || (target >= 0 && target < kSliders)) &&
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
    P.dirty = false; // before drawing: a page may ask to be drawn again
    Canvas c{kPanelW, kPanelH};
    draw_panel(c);
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
  spot_recenter_frame();
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
    // and looks, up or down too (a headset on a desk may face anywhere). =2:
    // straight ahead at the default distance, as a player looking at the
    // stage sees it (the default's 25 cm under the eyes put it at the bottom
    // of a lying headset's view: 2% of the eye, the per-pixel costs a
    // player's view has hidden; 2026-10-23).
    static const bool ahead = env_float("AURORA_XR_ARENA_AT_HEAD", 0.f) >= 2.f;
    const XrVector3f& d = g_defaultArena.pos;
    g_arena.pos = !atHead ? XrVector3f{B.head.x, 0.f, B.head.z} + rot_y(d, yaw)
                  : ahead ? B.head + qrot(B.headOrientation, {0.f, 0.f, -std::sqrt(vdot(d, d))})
                          : B.head + qrot(B.headOrientation, d);
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
  // The dev tools panel, in a paused fight.
  const bool roomWanted = B.passthroughLayer && B.passthroughRunning && g_passthroughWanted;
  update_dev_panel(dev_tools_wanted() && pointing && g_fightPaused && placing == 0,
                   roomWanted && B.hasColorLutExt);
  update_grab(fight ? Target::Arena : Target::Screen, pointing || screenPointing);
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
  B.pointerMode = pointing || onScreen;
  // The triggers click the screen or the panel instead of pressing L and R.
  B.triggersClick = onScreen || onPanel;
  // The surface under the stage is measured while it's being placed: the
  // hold before a fight, or a grab in a pause.
  const bool placingNow = fight && (placing > 0 || (G.target == Target::Arena && (G.oneHand >= 0 || G.twoHands)));
  surface_frame(fight, placingNow, fs.predictedDisplayTime);
  // Ambient capture in the menus, while the stage is placed and in a pause
  // (where the stage is placed again; not Slippi's, the match runs on).
  capture_ambient(!fight || placing > 0 || (g_fightPaused && g_matchStopped));
  room_survey_apply();
  spot_frame(fs.predictedDisplayTime);
  room_depth_frame(fs.predictedDisplayTime);
  density_frame();
  behind_frame(fight && placing == 0);

  // The passthrough, the shadow, the HUD and the 3D view, the HUD on top in
  // full VR, then the placing cards and the lasers.
  std::array<const XrCompositionLayerBaseHeader*, 5 + kMaxPlacingLayers + kMaxPointerLayers> layers{};
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
  if (fs.shouldRender && fight) {
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
    if (panel)
      layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&panelQuad);
    if (pointing) {
      build_pointer_layers(pointers);
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
      // stage and the screen keep their places in the room.
      const auto& e = reinterpret_cast<const XrEventDataReferenceSpaceChangePending&>(ev);
      Log.info("Reference space {} recentered", static_cast<int>(e.referenceSpaceType));
      g_recenteredNs.store(monotonic_ns(), std::memory_order_relaxed);
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
  init_room_light();
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
#ifndef __ANDROID__
  g_replayPollStop = true;
  if (g_replayPoll.joinable())
    g_replayPoll.join();
#endif
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
// Then the room light's fields (shader.cpp's XrEye: room_m to room_probe).
constexpr uint64_t kMultiviewEyeSize = 2 * 64 + 16 + gfx::XrMaxClipPlanes * 16 + gfx::XrMaxClipPlanes * 4 + 4 * 64 +
                                       64 + 4 * 16 + kCapPoints * 9 * 16 + 4 * 16 + 2 * 16 + 2 * 16 + 4 * 16;

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
    multiview.fragmentDensityMap = g_density.on.load(); // drawn in the eyes' pass
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
        }
      } else if (dr.spanNs == 0) {
        // Until margins come in: what a game frame leaves us on the GPU.
        const float hz = g_displayHz > 0.f ? g_displayHz.load() : 120.f;
        const float compositorMs = g_compositorGpuMs > 0.f ? g_compositorGpuMs.load() : 2.3f;
        dr.targetNs = std::clamp(1.0e9 / 60.0 - compositorMs * 1.0e6 * hz / 60.0 - dr.headroomNs, 5.0e6, 16.0e6);
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
      } else if (++dr.sinceStep >= kDynresUpDwell && dr.level + 1 < levels && (dr.haveMargin || dr.fixedTarget) &&
                 predicted[dr.level + 1] <= dr.targetNs) {
        next = dr.level + 1;
      }
    }
    if (next > dr.level && !back)
      next = std::min(next, std::max(ceiling, dr.level));
    // Below 1.0 only while SpaceWarp is on (see minScale).
    const int floor = appsw_on() ? 0 : std::max(static_cast<int>((1.f - dr.minScale) / kDynresStep + 0.5f), 0);
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
                 const std::vector<std::array<float, gfx::XrTransformFloats>>& worldTransforms,
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
    // Room lighting (room_frame): game camera space to the room, the
    // weights (fighters, stage, unlit), an up-facing surface's light, the
    // key light's direction and half vector in camera space, the SH, the
    // two lights' directions in the room and their colours.
    Mat4 roomM;
    float roomW[4], roomUp[4], roomL[4], roomH[4];
    float roomSh[kCapPoints][9][4];
    float roomDir[2][4], roomCol[2][4];
    // Each eye in game camera space (the highlight's view vector); w: the
    // fighters' gloss, then the stage's.
    float roomEye[2][4];
    // The point lights in game camera space, w the squared distance (game
    // units) their colour applies at (0: directional); the probe grid in
    // game camera space: its first point, then its three axes (each its
    // whole span), w the grid's size along each (a fighter's own probe: one
    // point, its blend).
    float roomLp[2][4];
    float roomProbe[4][4];
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
    if (const RoomFrame room = room_frame(arena); room.on)
      for (size_t t = 0; t < transforms; ++t) {
        auto& u = mv[t];
        const Mat4& m = cameraToRoom[t];
        u.roomM = m;
        std::copy_n(room.w, 4, u.roomW);
        std::copy_n(room.up, 4, u.roomUp);
        std::memcpy(u.roomSh, room.sh, sizeof(u.roomSh));
        std::memcpy(u.roomCol, room.col, sizeof(u.roomCol));
        for (int l = 0; l < 2; ++l)
          std::copy_n(room.dir[l].data(), 3, u.roomDir[l]);
        // The key light into camera space (the transpose of the rotation:
        // the placement scales evenly), and its half vector with the view
        // direction (+z, toward the camera), as GX's specular light takes.
        const auto& d = room.dir[0];
        float l[3], h[3];
        for (int c = 0; c < 3; ++c)
          l[c] = m[0 * 4 + c] * d[0] + m[1 * 4 + c] * d[1] + m[2 * 4 + c] * d[2];
        const float ll = std::sqrt(l[0] * l[0] + l[1] * l[1] + l[2] * l[2]);
        for (float& c : l)
          c = ll > 1e-12f ? c / ll : 0.f;
        h[0] = l[0], h[1] = l[1], h[2] = l[2] + 1.f;
        const float hl = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
        for (int c = 0; c < 3; ++c) {
          u.roomL[c] = l[c];
          u.roomH[c] = hl > 1e-12f ? h[c] / hl : 0.f;
        }
        u.roomH[3] = room.delight;
        const Mat4 toCamera = inverse_affine({m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11]});
        const auto to_camera = [&](const XrVector3f& p) {
          std::array<float, 3> o;
          for (int c = 0; c < 3; ++c)
            o[c] = toCamera[c * 4] * p.x + toCamera[c * 4 + 1] * p.y + toCamera[c * 4 + 2] * p.z + toCamera[c * 4 + 3];
          return o;
        };
        // The grid, or a fighter's probe (its entry's last floats): the SH
        // blended at its middle, as one point.
        constexpr int kProbe = 12 + gfx::XrClipFloats;
        const XrVector3f* probe = nullptr;
        XrVector3f probeRoom;
        if (t > 0 && worldTransforms[t - 1][kProbe + 3] > 0.f) {
          const auto& e = worldTransforms[t - 1];
          const Mat4& w = worldToRoom[t];
          probeRoom = {w[0] * e[kProbe] + w[1] * e[kProbe + 1] + w[2] * e[kProbe + 2] + w[3],
                       w[4] * e[kProbe] + w[5] * e[kProbe + 1] + w[6] * e[kProbe + 2] + w[7],
                       w[8] * e[kProbe] + w[9] * e[kProbe + 1] + w[10] * e[kProbe + 2] + w[11]};
          probe = &probeRoom;
          probe_blend(room.grid, room.sh, probeRoom, u.roomSh[0]);
          if (static bool logged = false; !std::exchange(logged, true))
            Log.info("Room light: a draw's own probe at {:.2f},{:.2f},{:.2f} (a fighter's middle)", probeRoom.x,
                     probeRoom.y, probeRoom.z);
        }
        const ProbeGrid& g = room.grid;
        {
          const auto o = to_camera(probe ? *probe : g.origin);
          std::copy(o.begin(), o.end(), u.roomProbe[0]);
          for (int a = 0; a < 3; ++a) {
            const auto e = to_camera(g.origin + g.axis[a]), b = to_camera(g.origin);
            for (int c = 0; c < 3; ++c)
              u.roomProbe[a + 1][c] = probe ? 0.f : e[c] - b[c];
          }
          for (int a = 0; a < 3; ++a)
            u.roomProbe[a][3] = probe ? 1.f : static_cast<float>(g.n[a]);
          u.roomProbe[3][3] = 0.f;
        }
        const auto refCam = to_camera(g.at(g.ref));
        // A point light's place, w the squared distance its colour applies
        // at; a directional one's direction (unit), w 0.
        for (int l = 0; l < 2; ++l) {
          if (room.range[l] > 0.f) {
            const auto q = to_camera(room.pos[l]);
            std::copy(q.begin(), q.end(), u.roomLp[l]);
            float d2 = 0.f;
            for (int c = 0; c < 3; ++c)
              d2 += (q[c] - refCam[c]) * (q[c] - refCam[c]);
            u.roomLp[l][3] = d2;
          } else {
            const auto& d = room.dir[l];
            float v[3], n2 = 0.f;
            for (int c = 0; c < 3; ++c) {
              v[c] = toCamera[c * 4] * d[0] + toCamera[c * 4 + 1] * d[1] + toCamera[c * 4 + 2] * d[2];
              n2 += v[c] * v[c];
            }
            const float k = n2 > 1e-24f ? 1.f / std::sqrt(n2) : 0.f;
            for (int c = 0; c < 3; ++c)
              u.roomLp[l][c] = v[c] * k;
            u.roomLp[l][3] = 0.f;
          }
        }
        for (int eye = 0; eye < 2; ++eye) {
          const XrVector3f p = views[eye].pose.position;
          for (int c = 0; c < 3; ++c)
            u.roomEye[eye][c] = toCamera[c * 4] * p.x + toCamera[c * 4 + 1] * p.y + toCamera[c * 4 + 2] * p.z +
                                toCamera[c * 4 + 3];
          u.roomEye[eye][3] = room.gloss[eye];
        }
      }
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
    // Foveation: the eyes' pipelines need a density map, and none is ready
    // yet (they're made once the headset has views, long before a fight).
    if (g_density.on && !density_ready()) {
      R.missedStereo = true;
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
        if (g_density.on)
          t.densityMap = density_view(static_cast<uint32_t>(vw), static_cast<uint32_t>(vh));
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
    // Foveation: the eyes' pass has a density map, its pipelines are made for one.
    mvLayout.xrDensityMap = g_density.level > 0 ? 1 : 0;
    if (mvLayout.xrDensityMap != 0 && !g_density.on.exchange(true))
      Log.info("Foveation: level {}", g_density.level);
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

extern "C" void aurora_xr_set_rumble(bool on) {
  aurora::xr::g_rumbleWanted.store(on, std::memory_order_relaxed);
}

extern "C" bool aurora_xr_get_pad(PADStatus* out) {
  PADFightRaw raw;
  {
    std::lock_guard lock{aurora::xr::g_padMutex};
    if (!aurora::xr::g_padValid)
      return false;
    raw = aurora::xr::g_touchRaw;
  }
  // Through the fight mapping in a fight (PADSetFightMappingActive), the
  // usual one otherwise: the sticks at full scale, the triggers' analog
  // values as they are.
  static aurora::pad::FightReader reader; // the game thread's
  aurora::pad::fight_filter(PAD_FIGHT_DEVICE_TOUCH, reader, raw);
  *out = {};
  aurora::pad::fight_compose(PAD_FIGHT_DEVICE_TOUCH, raw,
                             PADFightMappingActive() ? nullptr : aurora::pad::fight_standard(PAD_FIGHT_DEVICE_TOUCH),
                             127.f, 127.f, 0, *out);
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
extern "C" void aurora_xr_set_match_stopped(bool stopped) { aurora::xr::g_matchStopped = stopped; }

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

namespace aurora::xr {
namespace {
// The capture's log line, as just made (under estMutex; casts from the
// policy's).
void capture_log(const char* what, uint32_t frames, uint32_t casts) {
  Log.info("Room capture{}: {} frames, {} casts; {}", what, frames, casts, g_capture.est.summary());
}

// Asked of the capture thread's next poll (aurora_xr_room_capture_wanted):
// a make and its log, off the XR thread.
bool g_captureMakePending = false, g_captureLogPending = false;
uint32_t g_captureLogFrames = 0;

// XR thread, every frame: whether ambient capture may run (the menus, the
// stage being placed, a pause). Leaving them, the light is made again (the
// stage may have moved since the last frame) and what ambient capture
// gathered logged, by the capture thread.
void capture_ambient(bool ok) {
  std::lock_guard lock{g_capture.mutex};
  auto& c = g_capture;
  if (c.ambientOk && !ok) {
    g_captureMakePending = true;
    if (c.ambientFrames > 0 && !c.requested) {
      g_captureLogPending = true;
      g_captureLogFrames = c.ambientFrames;
      c.ambientFrames = 0;
    }
  }
  c.ambientOk = ok;
}
} // namespace
} // namespace aurora::xr

// The capture thread's poll (every 250 ms): the mode, and the makes and logs
// the other threads asked for.
extern "C" int aurora_xr_room_capture_wanted(void) {
  using namespace aurora::xr;
  auto& c = g_capture;
  bool make, log;
  const char* what = "";
  uint32_t frames = 0, casts;
  int mode;
  {
    std::lock_guard lock{c.mutex};
    make = std::exchange(g_captureMakePending, false);
    log = std::exchange(g_captureLogPending, false);
    if (log)
      what = " (ambient)", frames = g_captureLogFrames;
    if (c.requested && c.untilNs != 0 && steady_ns() >= c.untilNs) {
      c.requested = false;
      make = log = true;
      what = "";
      frames = 0; // the estimate's own count
      c.ambientFrames = 0;
    }
    casts = c.casts;
    mode = capture_mode_locked();
  }
  {
    // Made again whenever the stage moved since the last make (Roomtone's
    // stale: a point moved 5 cm), capture or not: a recenter mid-fight
    // moved the stage across the room with no capture running, and it kept
    // the old spot's light, nearer a window, until the next pause
    // (2026-10-23). A make costs ~15 ms here, at most every poll.
    std::lock_guard lock{c.estMutex};
    if ((make && log) || c.est.stale()) {
      const int64_t t0 = steady_ns();
      c.est.make();
      // Logged at most every 10 s: how many makes, for how many points, and
      // what they cost (PROBES' GRID makes many more points than LINE).
      static uint32_t makes = 0;
      static double total = 0.0, most = 0.0;
      static int64_t since = t0;
      static int points = 0;
      const double ms = static_cast<double>(steady_ns() - t0) * 1e-6;
      ++makes, total += ms, most = std::max(most, ms);
      if (steady_ns() - since >= 10'000'000'000 || c.last.grid.count() != points) {
        points = c.last.grid.count();
        Log.info("Room light made {} times in {:.0f} s ({} points): {:.1f} ms each, {:.1f} at most", makes,
                 static_cast<double>(steady_ns() - since) * 1e-9, c.last.grid.count(), total / makes, most);
        makes = 0, total = most = 0.0, since = steady_ns();
      }
    }
    if (log)
      capture_log(what, frames != 0 ? frames : c.est.frames(), casts);
  }
  return mode;
}

extern "C" void aurora_xr_room_capture_camera(const float rotation[4], const float translation[3],
                                              const float intrinsics[4], int sensor_width, int sensor_height, int width,
                                              int height) {
  using namespace aurora::xr;
  (void)translation; // the room's light is at a distance
  std::lock_guard lock{g_capture.estMutex};
  auto& c = g_capture;
  roomtone::Camera cam;
  cam.lensRotation = {rotation[0], rotation[1], rotation[2], rotation[3]};
  cam.fx = intrinsics[0], cam.fy = intrinsics[1], cam.cx = intrinsics[2], cam.cy = intrinsics[3];
  cam.sensorWidth = sensor_width, cam.sensorHeight = sensor_height, cam.width = width, cam.height = height;
  c.est.setCamera(cam);
  c.rec.camera(cam);
  Log.info("Room capture: camera {}x{} of a {}x{} sensor, f {:.1f},{:.1f} c {:.1f},{:.1f}, lens {:.4f},{:.4f},{:.4f},{:.4f}",
           width, height, sensor_width, sensor_height, intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3],
           rotation[0], rotation[1], rotation[2], rotation[3]);
}

// A frame's grid (per cell linear r, g, b and the share clipped), with the
// head's latest pose. A frame without its exposure (the first, before
// Camera2's first result) is left out: unscaled, it read ~100 times too
// bright, a light wherever the head pointed then.
extern "C" void aurora_xr_room_capture_frame(const float* rgb, int cols, int rows, float exposure_ms, float iso) {
  using namespace aurora::xr;
  XrQuaternionf head;
  XrVector3f headAt;
  {
    std::lock_guard lock{g_viewMutex};
    if (!g_viewsValid)
      return;
    head = B.headOrientation;
    headAt = B.head;
  }
  auto& c = g_capture;
  int mode;
  bool haveDepth;
  roomtone::DepthImage depth;
  std::vector<uint16_t> depthPx;
  {
    std::lock_guard lock{c.mutex};
    mode = capture_mode_locked();
    if (mode == 0 || cols <= 0 || rows <= 0 || !(exposure_ms > 0.f && iso > 0.f))
      return;
    if (mode == 1 && c.untilNs == 0)
      c.untilNs = steady_ns() + kCaptureNs;
    haveDepth = std::exchange(c.depthPending, false);
    if (haveDepth) {
      depth = c.depth;
      depthPx.swap(c.depthPx);
    }
  }
  roomtone::FrameResult r;
  {
    std::lock_guard lock{c.estMutex};
    if (!c.est.hasCamera())
      return;
    if (haveDepth) {
      depth.depth = depthPx.data();
      c.est.addDepth(depth);
      c.rec.depth(depth);
    }
    const roomtone::Frame f{rgb, cols, rows, exposure_ms, iso, {rt(head), rt(headAt)}, steady_ns()};
    c.rec.frame(f);
    r = c.est.addFrame(f);
  }
  if (r.restarted)
    Log.info("Room capture: the room changed ({:.2f}x the capture where it saw well); starting over", r.change);
  std::lock_guard lock{c.mutex};
  if (r.used) {
    c.any = true;
    if (mode == 2)
      ++c.ambientFrames;
  }
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
  aurora::gfx::xr_room_fit_reset(); // and its baked light is another
  {
    std::lock_guard lock{g_grade.mutex};
    g_grade.stage = stage;
  }
  aurora_xr_set_room_grade(1.f, 1.f, 1.f, 1.f);
  g_roomStage = stage;
  {
    RoomLight r;
    {
      std::lock_guard lock{g_room.mutex};
      r = g_room.value;
    }
    room_light_apply(r); // the stage's gloss pipelines or none
  }
  std::lock_guard lock{g_arenaMutex};
  g_stage = stage;
  g_arenaCenter = {x, y, z};
  g_stageScale = scale > 0.f ? scale : 1.f;
  g_stageZoom = 1.f;
  g_hudOnTop = false;
  g_footprint = {};
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
