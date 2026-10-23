#pragma once

#include "hash.hpp"
#include "../internal.hpp"
#include "../webgpu/gpu.hpp"

#include <aurora/gfx.h>
#include <aurora/gfx.hpp>
#include <aurora/math.hpp>
#include <dolphin/gx/GXEnum.h>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

namespace aurora::gfx {

// What the game said was being drawn (aurora_xr_camera), so the XR
// presenter can re-draw the world for each eye and the HUD on its own plane.
enum class XrCategory : uint8_t {
  Mono = 0,  // only part of the normal frame
  World = 1, // 3D fight geometry: also replayed per eye
  Hud = 2,   // HUD: also replayed onto the HUD plane
  Hidden = 3, // fight geometry left out of 3D: flat frame only, dropped with World
};
// World draws can carry an extra placement in the 3D view
// (aurora_xr_world_transform) and clip planes: index 0 is none, 1.. index the
// frame's list of distinct (placement, clip) entries. A draw past a full list
// goes out with neither: at 8, Great Bay (its turtle's, pier's and sea's cuts,
// the sea's move, fades) overflowed, and its sea flashed out to the horizon.
// Each entry is a small per-eye uniform buffer, written only when used.
constexpr uint32_t XrMaxTransforms = 32;
// Clip planes a world draw can carry (aurora_xr_world_clips4_fade). Their
// distances ride to the fragment shader four to a vec4.
constexpr int XrMaxClipPlanes = 8;
// A clip's floats: the planes, their fade bands, then the draw's opacity.
constexpr int XrClipFloats = XrMaxClipPlanes * 5 + 1;
// A draw's light probe (aurora_xr_light_probe): game world xyz, then 1 when
// set (0: the room light's grid at each vertex).
constexpr int XrProbeFloats = 4;
// A world transform entry: move 3x4, the clip, the light probe.
constexpr int XrTransformFloats = 12 + XrClipFloats + XrProbeFloats;
// The room light's probes (lib/xr's grid, the shader's room_sh): at most
// this many points, nine SH coefficients each.
constexpr int XrRoomProbes = 36;

using BindGroupRef = HashType;
using PipelineRef = HashType;
using SamplerRef = HashType;
using ShaderRef = HashType;

struct ClipRect {
  int32_t x;
  int32_t y;
  int32_t width;
  int32_t height;

  bool operator==(const ClipRect& rhs) const { return memcmp(this, &rhs, sizeof(*this)) == 0; }
  bool operator!=(const ClipRect& rhs) const { return !(*this == rhs); }
};

using webgpu::Viewport;

struct TextureRef;
using TextureHandle = std::shared_ptr<TextureRef>;
using AfterSubmitCallback = std::function<void()>;
using EndFrameCallback = std::function<void(wgpu::CommandEncoder&, std::vector<AfterSubmitCallback>)>;
} // namespace aurora::gfx
