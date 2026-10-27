#pragma once

#include "../gfx/types.hpp"
#include "gx.hpp"

namespace aurora::gfx::detail {
struct FramePacket;
} // namespace aurora::gfx::detail

namespace aurora::gx {
struct DrawData {
  gfx::PipelineRef pipeline;
  // XR multiview twin of `pipeline` (same config, multiview layout), for world
  // draws while lib/xr replays them with a view mask; 0 = none.
  gfx::PipelineRef xrPipeline;
  gfx::Range vertRange;
  gfx::Range idxRange;
  gfx::Range posRange; // decoded float3 positions (AURORA_POS_DECODE), in the storage buffer
  gfx::Range uniformRange;
  DrawImmediateData immediateData;
#ifdef __EMSCRIPTEN__
  gfx::Range immediateRange;
#endif
  uint32_t vtxCount;
  uint32_t indexCount;
  uint32_t instanceCount;
  GXBindGroups bindGroups;
  uint32_t dstAlpha;
  // Breadcrumb from the most recent GXInsertDebugMarker. Captured here
  // because the FIFO is processed on a worker thread, so reading a global at
  // render time attributes draws to whatever the game thread did last.
  uint32_t tag;
  // XR multiview replay only: the triangles drawn in the eyes when a clip
  // leaves some out (count 0: all of them, as idxRange/indexCount).
  gfx::Range xrIdxRange{};
  uint32_t xrIndexCount = 0;
  // XR multiview replay only: a clipped draw's triangles wholly inside the
  // clip's solid region, drawn after the rest with this pipeline (no
  // discard); count 0: none.
  gfx::PipelineRef xrSolidPipeline = 0;
  gfx::Range xrSolidIdxRange{};
  uint32_t xrSolidIndexCount = 0;
  // XR SpaceWarp motion replay only: the motion-vector twin of xrPipeline
  // (0: not drawn into the motion vectors).
  gfx::PipelineRef xrMotionPipeline = 0;
  // XR batching (AURORA_XR_BATCH): the batched twins of xrPipeline and
  // xrMotionPipeline (0: never batched), and the uniform without its
  // position and normal matrices, hashed (draws batch only when it agrees).
  // A draw with a solid part (xrSolid*): the batched twin of xrSolidPipeline,
  // for that part alone.
  gfx::PipelineRef xrBatchPipeline = 0;
  gfx::PipelineRef xrMotionBatchPipeline = 0;
  // The motion twin keeps the game's fragment stage (a cutout or translucent
  // one): batched only with draws of the same material, like the eyes' are.
  bool xrMotionMaterial = false;
  // Eye replay (AURORA_XR_EYE_SORT): an opaque, depth-tested and -writing draw
  // with no alpha test or polygon offset, which may be drawn out of order
  // with others like it (only exact depth ties between them could differ).
  bool xrReorderable = false;
  uint64_t uniformSharedKey = 0;
};

#ifdef __EMSCRIPTEN__
constexpr uint32_t GXPipelineConfigVersion = 10013;
#else
constexpr uint32_t GXPipelineConfigVersion = 13;
#endif
struct PipelineConfig {
  uint32_t version = GXPipelineConfigVersion;
  uint32_t msaaSamples = 1; // deprecated
  ShaderConfig shaderConfig;
  GXCompare depthFunc;
  GXCullMode cullMode;
  GXBlendMode blendMode;
  GXBlendFactor blendFacSrc, blendFacDst;
  GXLogicOp blendOp;
  uint32_t dstAlpha;
  uint32_t polygonOffsetBits;
  uint32_t polygonOffsetScaleBits;
  uint32_t polygonOffsetClampBits;
  bool depthCompare, depthUpdate, alphaUpdate, colorUpdate;
};
static_assert(std::has_unique_object_representations_v<PipelineConfig>);

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config, const gfx::RenderTargetLayout& layout);
void render(const DrawData& data, const wgpu::RenderPassEncoder& pass);
#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
// XR batching around an eye or motion replay (encode_xr_replay): begin, flush
// before anything that changes what later draws see (group 3), end (uploads).
void xr_batch_begin(const gfx::detail::FramePacket& frame, bool multiview, bool motion);
void xr_batch_flush(const wgpu::RenderPassEncoder& pass);
// The eye uniforms just set at group 3; a bindless batch binds them at 2.
void xr_batch_eye_group(const wgpu::BindGroup& group);
void xr_batch_end();
#endif

void queue_surface(const u8* dlStart, uint32_t dlSize, bool bigEndian) noexcept;
} // namespace aurora::gx
