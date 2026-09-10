#ifndef AURORA_AURORA_H
#define AURORA_AURORA_H

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>

extern "C" {
#else
#include "stdbool.h"
#include "stddef.h"
#include "stdint.h"
#endif

typedef enum {
  SAMPLER_BILINEAR,
  SAMPLER_AREA,
  SAMPLER_CRT,
  SAMPLER_VIBRANT,
} AuroraSampler;

typedef enum {
  BACKEND_AUTO,
  BACKEND_D3D11,
  BACKEND_D3D12,
  BACKEND_METAL,
  BACKEND_VULKAN,
  BACKEND_OPENGL,
  BACKEND_OPENGLES,
  BACKEND_WEBGPU,
  BACKEND_NULL,
} AuroraBackend;

typedef enum {
  LOG_DEBUG,
  LOG_INFO,
  LOG_WARNING,
  LOG_ERROR,
  LOG_FATAL,
} AuroraLogLevel;

typedef struct {
  int32_t x;
  int32_t y;
} AuroraWindowPos;

typedef struct {
  uint32_t width;
  uint32_t height;

  /**
   * Width of the main GX framebuffer.
   */
  uint32_t fb_width;

  /**
   * Height of the main GX framebuffer.
   */
  uint32_t fb_height;

  /**
   * The size of the framebuffer used to present to the operating system.
   * May differ from fb_width if Aurora is instructed to force an aspect ratio or resolution configuration.
   */
  uint32_t native_fb_width;

  /**
   * The size of the framebuffer used to present to the operating system.
   * May differ from fb_height if Aurora is instructed to force an aspect ratio or resolution configuration.
   */
  uint32_t native_fb_height;
  float scale;
} AuroraWindowSize;

typedef struct SDL_Window SDL_Window;
typedef struct AuroraEvent AuroraEvent;

typedef void (*AuroraLogCallback)(AuroraLogLevel level, const char* module, const char* message, unsigned int len);
typedef void (*AuroraImGuiInitCallback)(const AuroraWindowSize* size);

#define MEM1_DEFAULT_SIZE (24 * 1024 * 1024)
#define ARAM_DEFAULT_SIZE (16 * 1024 * 1024)

typedef struct {
  const char* appName;
  const char* userPath;
  const char* cachePath;
  const char* resourcesPath;
  AuroraBackend desiredBackend;
  uint32_t msaa;
  uint16_t maxTextureAnisotropy;
  bool vsync;
  bool startFullscreen;
  /* smstrikers-port: open maximised; windowWidth/windowHeight is the size it restores to. */
  bool startMaximized;
  bool allowJoystickBackgroundEvents;
  bool pauseOnFocusLost;
  bool allowTextureDumps;
  bool allowCpuAdapter;
  int32_t windowPosX;
  int32_t windowPosY;
  uint32_t windowWidth;
  uint32_t windowHeight;
  void* iconRGBA8;
  uint32_t iconWidth;
  uint32_t iconHeight;
  AuroraLogCallback logCallback;
  AuroraLogLevel logLevel;
  AuroraImGuiInitCallback imGuiInitCallback;

  /*
   * The size of the GameCube's main memory, or MEM1 on the Wii.
   * Note that it will not be allocated at the exact 0x80000000 address, as that cannot be guaranteed.
   * This can be set to 0 to disable allocating this region.
   */
  uint32_t mem1Size;

  /*
   * The size of the GameCube's ARAM, or MEM2 on the Wii.
   * This can be set to 0 to disable allocating this region.
   */
  uint32_t mem2Size;

  /*
   * OpenXR builds: the eyes' resolution, a fixed share of the runtime's
   * recommended size (0.8-1.3), or 0 for dynamic resolution where it is
   * available. AURORA_XR_EYE_SCALE and AURORA_XR_DYNRES override it.
   */
  float xrEyeScale;
  /*
   * OpenXR builds: leave Application SpaceWarp off even where the runtime
   * offers it. AURORA_XR_APPSW overrides it.
   */
  bool xrNoSpaceWarp;
  /*
   * OpenXR builds, mixed reality, for the room around the stage. No
   * contact shadow on the surface under the stage (and no raycasting for
   * it); AURORA_XR_RAYCAST and AURORA_XR_SURFACE_SHADOW override it.
   */
  bool xrNoStageShadow;
} AuroraConfig;

typedef struct {
  AuroraBackend backend;
  /* Adapter the backend landed on: WebGPU device name, driver description,
   * PCI vendor/device id. Empty strings / 0 without a GPU backend. */
  const char* adapterName;
  const char* adapterDriver;
  uint32_t adapterVendorId;
  uint32_t adapterDeviceId;
  const char* userPath;
  const char* cachePath;
  SDL_Window* window;
  AuroraWindowSize windowSize;
} AuroraInfo;

AuroraInfo aurora_initialize(int argc, char* argv[], const AuroraConfig* config);
void aurora_shutdown();
/* smstrikers-port: write the next presented frame to `path` as a binary PPM. */
void aurora_capture_frame(const char* path);
/* smstrikers-port: queue submit to OnSubmittedWorkDone, which is not GPU
 * execution time on Metal: the MSAA sample count does not move it. */
void aurora_gpu_frame_time(uint64_t* lastNs, uint64_t* meanNs, uint64_t* maxNs, uint64_t* count);
/* smstrikers-port: size the GX framebuffer from the VI render mode times
 * `scale` rather than from the window. 0 restores the default. */
void aurora_set_frame_buffer_scale(float scale);
/* smstrikers-port: the window as it is right now, polled, because macOS reports a move between displays of different backing scale through a scale change, a resize, both or neither. */
AuroraWindowSize aurora_window_size();
/* smstrikers-port: rebuild the framebuffer from the window and the locked aspect now rather than on the next event pump. Safe between frames only; idempotent. */
void aurora_apply_frame_buffer_resize();
const AuroraEvent* aurora_update();
bool aurora_begin_frame();
void aurora_end_frame();

void aurora_set_log_level(AuroraLogLevel level);
void aurora_set_pause_on_focus_lost(bool value);
void aurora_set_background_input(bool value);
void aurora_set_resampler(AuroraSampler sampler);
/** Keeps the previous frame instead of clearing, for hosts that pause the
 *  game: with nothing drawing, a cleared framebuffer presents as black. */
void aurora_preserve_frame_buffer(bool preserve);
/** Sets the clock timescale. Default 1.0f. 0.0f is paused. Range 0.0f-16.0f. */
void aurora_set_timescale(float scale);

AuroraBackend aurora_get_backend();
const AuroraBackend* aurora_get_available_backends(size_t* count);
float aurora_get_timescale();

#ifdef __cplusplus
}
#endif

#endif
