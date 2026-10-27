#include "pipeline.hpp"

#include "../gfx/encoding.hpp"
#include "../gfx/resources.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "../gfx/recording.hpp"
#include "../gfx/resource_cache.hpp"
#include "../gfx/frame_packet.hpp"
#include "../webgpu/gpu.hpp"

#include "gx_fmt.hpp"
#include "shader_info.hpp"

#include <array>
#include <cstdlib>
#include <deque>
#include <string>

#include "../internal.hpp"

#include <absl/container/flat_hash_map.h>
#include <tracy/Tracy.hpp>

namespace aurora::gx {
namespace {
constexpr Module Log{"aurora::gx::pipeline"};
} // namespace

wgpu::RenderPipeline create_pipeline(const PipelineConfig& config, const gfx::RenderTargetLayout& layout) {
  ZoneScoped;
  const auto shader = build_shader(config.shaderConfig, layout);
  const auto label = fmt::format("GX Pipeline {:x}",
                                 xxh3_hash(layout.key, xxh3_hash(config, static_cast<HashType>(gfx::ShaderType::GX))));
  DecodedLayout decoded;
  if (config.shaderConfig.decodedAll && decoded_layout(config.shaderConfig, decoded)) {
    std::array<wgpu::VertexAttribute, 16> attrs{};
    for (u8 i = 0; i < decoded.count; ++i) {
      const auto& d = decoded.attrs[i];
      attrs[i] = {.format = d.format == DecodedFormat::U32     ? wgpu::VertexFormat::Uint32
                            : d.format == DecodedFormat::F32x2 ? wgpu::VertexFormat::Float32x2
                            : d.format == DecodedFormat::F32x3 ? wgpu::VertexFormat::Float32x3
                                                               : wgpu::VertexFormat::Float32x4,
                  .offset = d.offset,
                  .shaderLocation = i};
    }
    const std::array buffers{wgpu::VertexBufferLayout{.stepMode = wgpu::VertexStepMode::Vertex,
                                                      .arrayStride = decoded.stride,
                                                      .attributeCount = decoded.count,
                                                      .attributes = attrs.data()}};
    return build_pipeline(config, layout, buffers, shader, label.c_str());
  }
  if (config.shaderConfig.decodedPos) {
    static constexpr wgpu::VertexAttribute posAttr{
        .format = wgpu::VertexFormat::Float32x3, .offset = 0, .shaderLocation = 0};
    const std::array buffers{wgpu::VertexBufferLayout{
        .stepMode = wgpu::VertexStepMode::Vertex, .arrayStride = 12, .attributeCount = 1, .attributes = &posAttr}};
    return build_pipeline(config, layout, buffers, shader, label.c_str());
  }
  return build_pipeline(config, layout, {}, shader, label.c_str());
}

// Diagnostics for the untextured-white-quad artifact in melee-pc:
//   AURORA_SKIP_UNTEX=1      drop every draw that binds no texture
//   AURORA_SKIP_UNTEX_VTX=n  drop only untextured draws with n vertices
//   AURORA_LOG_UNTEX=1       report untextured draws, with the breadcrumb
// A draw with no texture bind group falls back to flat colour, which is what
// a solid white quad looks like on screen. aurora_draw_tag lets the game
// record which subsystem is currently rendering, so an offending draw can be
// traced back past the GX boundary.
extern "C" {
unsigned int aurora_draw_tag = 0;
}

static bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' && *v != '0';
}

#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
// AURORA_XR_LINK_STATS=1 (with the game's link markers): the eye replay's
// draws and indices per GX link, every 20000 draws. AURORA_XR_SKIP_LINKS=a,b:
// leave those links out of the eyes (measuring what they cost).
static bool xr_link_filter(const DrawData& data) {
  static const bool stats = env_flag("AURORA_XR_LINK_STATS");
  static const uint64_t skip = [] {
    uint64_t mask = 0;
    if (const char* v = std::getenv("AURORA_XR_SKIP_LINKS"))
      for (const char* p = v; *p != '\0';) {
        char* end = nullptr;
        const unsigned long n = std::strtoul(p, &end, 10);
        if (end == p)
          break;
        if (n < 64)
          mask |= uint64_t{1} << n;
        p = *end == ',' ? end + 1 : end;
      }
    return mask;
  }();
  const uint32_t link = (data.tag >> 16) - 1; // UINT32_MAX: none
  if (stats) {
    static std::array<uint64_t, 34> draws{}, indices{};
    static uint32_t n = 0, pipelineSwitches = 0, textureSwitches = 0;
    static gfx::PipelineRef lastPipeline = 0;
    static decltype(data.bindGroups.textureBindGroup) lastTextures{};
    pipelineSwitches += data.xrPipeline != lastPipeline ? 1 : 0;
    textureSwitches += data.bindGroups.textureBindGroup != lastTextures ? 1 : 0;
    lastPipeline = data.xrPipeline;
    lastTextures = data.bindGroups.textureBindGroup;
    const size_t i = std::min<uint32_t>(link + 1, 33);
    ++draws[i];
    indices[i] += data.xrIndexCount != 0 ? data.xrIndexCount : data.indexCount != 0 ? data.indexCount : data.vtxCount;
    if (++n % 20000 == 0) {
      std::string line;
      for (size_t k = 0; k < draws.size(); ++k)
        if (draws[k] != 0)
          line += fmt::format(" link {}: {} draws {} idx;", k == 0 ? -1 : static_cast<int>(k) - 1, draws[k], indices[k]);
      Log.info("XR eye draws by GX link (last 20000):{}; pipeline changes {}, texture changes {}", line,
               pipelineSwitches, textureSwitches);
      draws = {};
      indices = {};
      pipelineSwitches = textureSwitches = 0;
    }
  }
  return link < 64 && (skip >> link & 1) != 0;
}
#endif

static void render_direct(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
  if (gfx::xr_multiview_replay() && !gfx::xr_motion_replay() && xr_link_filter(data))
    return;
  // A multiview replay can only bind multiview pipelines.
  if (gfx::xr_motion_replay()) {
    if (data.xrMotionPipeline == 0 || !gfx::bind_pipeline(data.xrMotionPipeline, pass))
      return;
  } else if (gfx::xr_multiview_replay() &&
             (data.xrPipeline == 0 || !gfx::bind_pipeline(data.xrPipeline, pass))) {
    return;
  }
  if (!gfx::xr_multiview_replay() && !gfx::bind_pipeline(data.pipeline, pass)) {
#else
  if (!gfx::bind_pipeline(data.pipeline, pass)) {
#endif
    /* AURORA_LOG_SKIPPED=1: report draws dropped because their pipeline is
     * still compiling. Pipelines are built asynchronously and an unready one
     * silently skips its draw, so a short-lived screen can miss its geometry
     * entirely on a cold cache. */
    if (env_flag("AURORA_LOG_SKIPPED")) {
      static u32 count = 0;
      if (++count <= 4000) {
        std::fprintf(stderr, "[WARN] aurora::gx: skipped draw: pipeline not ready (vtx %u, tag %u)\n",
                     static_cast<unsigned>(data.vtxCount), aurora_draw_tag);
      }
    }
    return;
  }

  // AURORA_ONLY_UNTEX_VTX=<n>: render ONLY untextured draws with n vertices
  // and drop everything else, so the offending geometry can be seen in
  // isolation on a black frame. Suppressing a draw proves it is responsible;
  // isolating it shows WHAT it is.
  {
    static const long onlyN = [] {
      const char* v = std::getenv("AURORA_ONLY_UNTEX_VTX");
      return v != nullptr ? std::strtol(v, nullptr, 10) : 0L;
    }();
    if (onlyN != 0 &&
        !(!data.bindGroups.textureBindGroup && static_cast<long>(data.vtxCount) == onlyN)) {
      return;
    }
  }

  if (!data.bindGroups.textureBindGroup) {
    static const bool skip = env_flag("AURORA_SKIP_UNTEX");
    static const bool log = env_flag("AURORA_LOG_UNTEX");
    // AURORA_SKIP_UNTEX_VTX=<n>: only drop untextured draws with exactly n
    // vertices, so a suspect quad can be removed without also removing the
    // legitimate untextured geometry that shares this path.
    static const long onlyVtx = [] {
      const char* v = std::getenv("AURORA_SKIP_UNTEX_VTX");
      return v != nullptr ? std::strtol(v, nullptr, 10) : 0L;
    }();
    // Untextured quads are the shape the artifact takes, so report those
    // individually and the rest only sparsely. The per-vertex byte stride
    // discriminates a 2D sprite quad (XY + ST floats = 16B) from 3D geometry.
    if (log) {
      static uint64_t n = 0;
      const bool quad = data.vtxCount == 4;
      if (quad || (n % 2000) == 0) {
        const uint32_t stride = data.vtxCount != 0 ? data.vertRange.size / data.vtxCount : 0;
        fmt::print(stderr, "untex draw #{}{}: idx={} vtx={} stride={} tag={}\n", n, quad ? " QUAD" : "",
                   data.indexCount, data.vtxCount, stride, data.tag);
      }
      ++n;
    }
    if (onlyVtx != 0) {
      if (static_cast<long>(data.vtxCount) == onlyVtx) {
        return;
      }
    } else if (skip) {
      return;
    }
  }

  const auto& resources = gfx::detail::resources();
#ifdef __EMSCRIPTEN__
  const std::array immediateOffsets{data.immediateRange.offset};
  pass.SetBindGroup(3, resources.uniformBindGroup, immediateOffsets.size(), immediateOffsets.data());
#else
#if defined(AURORA_ENABLE_OPENXR)
  if (gfx::xr_motion_replay() && gfx::xr_motion_zero_prev() && data.immediateData.xrPrevMtx != 0) {
    DrawImmediateData imm = data.immediateData;
    imm.xrPrevMtx = 0;
    pass.SetImmediates(0, &imm, sizeof(imm));
  } else
#endif
    pass.SetImmediates(0, &data.immediateData, sizeof(data.immediateData));
#endif
  const std::array offsets{data.uniformRange.offset};
  pass.SetBindGroup(1, resources.uniformBindGroup, offsets.size(), offsets.data());
  if (data.bindGroups.textureBindGroup) {
    pass.SetBindGroup(2, gfx::find_bind_group(data.bindGroups.textureBindGroup));
  }
#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
  // The motion pass draws a clipped draw whole: its clip test discards what
  // the eyes leave out, and the eyes' solid part (xrSolid*) is in it too.
  const bool xrCulled = gfx::xr_multiview_replay() && !gfx::xr_motion_replay() && data.xrIndexCount != 0;
#else
  constexpr bool xrCulled = false;
#endif
  const gfx::Range& idxRange = xrCulled ? data.xrIdxRange : data.idxRange;
  pass.SetIndexBuffer(resources.indexBuffer, wgpu::IndexFormat::Uint16, idxRange.offset, idxRange.size);
  if (data.posRange.size != 0) {
    pass.SetVertexBuffer(0, resources.storageBuffer, data.posRange.offset, data.posRange.size);
  }
  if (data.dstAlpha != UINT32_MAX) {
    const wgpu::Color color{0.f, 0.f, 0.f, data.dstAlpha / 255.f};
    pass.SetBlendConstant(&color);
  }
  if (xrCulled) {
    pass.DrawIndexed(data.xrIndexCount, data.instanceCount);
  } else if (data.indexCount == 0) {
    pass.Draw(data.vtxCount, data.instanceCount);
  } else {
    pass.DrawIndexed(data.indexCount, data.instanceCount);
  }
#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
  // The same draw's triangles inside the clip's solid region, without the
  // discard (same bind groups and buffers; the pipelines share a layout).
  if (gfx::xr_multiview_replay() && !gfx::xr_motion_replay() && data.xrSolidIndexCount != 0 &&
      gfx::bind_pipeline(data.xrSolidPipeline, pass)) {
    pass.SetImmediates(0, &data.immediateData, sizeof(data.immediateData));
    if (data.dstAlpha != UINT32_MAX) {
      const wgpu::Color color{0.f, 0.f, 0.f, data.dstAlpha / 255.f};
      pass.SetBlendConstant(&color);
    }
    pass.SetIndexBuffer(resources.indexBuffer, wgpu::IndexFormat::Uint16, data.xrSolidIdxRange.offset,
                        data.xrSolidIdxRange.size);
    pass.DrawIndexed(data.xrSolidIndexCount, data.instanceCount);
  }
#endif
}

#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
// XR batching (AURORA_XR_BATCH): in an eye or motion replay, a run of
// consecutive draws that differ only in their matrices and per-draw
// constants goes as one draw. Their indices, rebased onto the storage pool's
// start (decoded vertices are aligned to their stride for this), are joined
// into one 32-bit list; each draw's push constants and uniform offset go in a
// record, and a table of (first vertex, record) lets each vertex find its
// draw (the batched shader variant, gx/shader.cpp). Eye draws batch with the
// same pipeline, textures, blend constant, fog and uniform but for the
// position and normal matrices; motion draws with the same pipeline. The data
// goes in this frame's slot of the batch buffer, uploaded at the replay's end.
namespace {
struct XrBatcher {
  bool active = false;
  bool motion = false;
  const uint8_t* shadow = nullptr; // the frame's CPU copy of its indices
  uint64_t frameId = UINT64_MAX;
  uint64_t used = 0, uploaded = 0; // bytes of this frame's slot
  std::vector<uint8_t> staging = std::vector<uint8_t>(gfx::detail::XrBatchSlotSize);
  std::vector<const DrawData*> members;
  uint32_t memberIndices = 0;
  // A replay's straddling draws as their two parts (render): stable for the
  // members and buckets that point at them until the replay ends.
  std::deque<DrawData> split;
  uint64_t draws = 0, batches = 0, batchedDraws = 0;
  // Motion replay (AURORA_XR_MOTION_SORT, on by default): the pass writes
  // only depth and motion, depth-tested, so its draws' order doesn't matter
  // (but for exact depth ties, where both sides carry the same surface's
  // motion). Draws go into buckets of joinable ones, emitted at a flush, not
  // runs of neighbours: half the motion draws were left alone, ~6 us of GPU
  // each on four-CPU Brinstar.
  struct Bucket {
    std::vector<const DrawData*> members;
    uint32_t indices = 0;
  };
  bool sorting = false;
  // Eye replay (AURORA_XR_EYE_SORT, on by default): runs of reorderable opaque draws go in
  // buckets too, emitted when anything else comes (blended, depth-test-off,
  // alpha-tested, decal) or at a flush; everything else keeps its order.
  bool eyeSorting = false;
  // Bindless eye batches (gfx::xr_bindless): draws join across textures and
  // uniforms; each record carries its draw's texture and sampler slots in
  // the resource table (sBindless), bound once per pass.
  bool bindless = false;
  bool tableBound = false;
  // The eye uniforms the replay has at group 3 (xr_batch_eye_group): a
  // bindless batch takes them at group 2, the table being group 3.
  wgpu::BindGroup eyeGroup;
  std::vector<Bucket> buckets;
  size_t bucketsUsed = 0;
} sXrBatch;

// The resource table bindless eye batches sample through (render worker).
// A texture view or sampler gets a slot when a batched draw first uses it and
// keeps it while used; one unused for a few seconds is removed (Dawn reuses
// the slot only once the GPU is done with it). The table holds a reference
// to each, so a handle in `slots` can't be a dead object's reused address.
struct BindlessTable {
  static constexpr uint32_t kSize = 8192;
  static constexpr uint64_t kKeepFrames = 240, kSweepEvery = 120;
  wgpu::ResourceTable table;
  struct Entry {
    uint32_t slot;
    uint64_t lastFrame;
  };
  absl::flat_hash_map<const void*, Entry> slots;
  uint64_t frame = 0, lastSweep = 0;

  void begin_frame(uint64_t frameId) {
    if (!table) {
      const wgpu::ResourceTableDescriptor desc{.label = "XR bindless eye batches", .size = kSize};
      table = webgpu::g_device.CreateResourceTable(&desc);
    }
    frame = frameId;
    if (frame - lastSweep < kSweepEvery)
      return;
    lastSweep = frame;
    for (auto it = slots.begin(); it != slots.end();) {
      if (frame - it->second.lastFrame > kKeepFrames) {
        table.Remove(it->second.slot);
        slots.erase(it++);
      } else {
        ++it;
      }
    }
  }

  // The slot of a view or sampler (UINT32_MAX: the table is full).
  uint32_t slot(const void* handle, bool sampler) {
    if (const auto it = slots.find(handle); it != slots.end()) {
      it->second.lastFrame = frame;
      return it->second.slot;
    }
    wgpu::BindingResource res{};
    if (sampler)
      res.sampler = wgpu::Sampler(static_cast<WGPUSampler>(const_cast<void*>(handle)));
    else
      res.textureView = wgpu::TextureView(static_cast<WGPUTextureView>(const_cast<void*>(handle)));
    const uint32_t s = table.Insert(&res);
    if (s == wgpu::kInvalidBinding)
      return UINT32_MAX;
    slots.emplace(handle, Entry{s, frame});
    return s;
  }
} sBindless;

// A draw's 8 texture and 8 sampler slots (false: the table is full).
bool bindless_slots(const DrawData& d, uint32_t* out) {
  gfx::BindGroupResources res;
  if (!d.bindGroups.textureBindGroup || !gfx::bind_group_resources(d.bindGroups.textureBindGroup, res))
    res = {};
  for (uint32_t i = 0; i < 8; ++i) {
    const WGPUTextureView view = res.views[i * 2] ? res.views[i * 2].Get() : empty_texture_view();
    const WGPUSampler sampler = res.samplers[i * 2 + 1] ? res.samplers[i * 2 + 1].Get() : empty_sampler();
    out[i] = sBindless.slot(view, false);
    out[8 + i] = sBindless.slot(sampler, true);
    if (out[i] == UINT32_MAX || out[8 + i] == UINT32_MAX)
      return false;
  }
  return true;
}

gfx::PipelineRef batch_pipeline(const DrawData& d) {
  return sXrBatch.motion ? d.xrMotionBatchPipeline : d.xrBatchPipeline;
}

uint32_t draw_index_count(const DrawData& d) {
  return d.xrIndexCount != 0 ? d.xrIndexCount : d.indexCount != 0 ? d.indexCount : d.vtxCount;
}

bool batchable(const DrawData& d) {
  return batch_pipeline(d) != 0 && d.instanceCount == 1 && d.posRange.size != 0 && d.vtxCount != 0 &&
         d.posRange.size % d.vtxCount == 0 && d.posRange.offset % (d.posRange.size / d.vtxCount) == 0 &&
         (sXrBatch.motion || d.xrSolidIndexCount == 0);
}

constexpr size_t kBatchMaxMembers = 128;
constexpr uint32_t kBatchMaxIndices = 65536 * 4;

// d could go in one batched draw with first (limits aside).
bool compatible(const DrawData& first, const DrawData& d) {
  if (batch_pipeline(d) != batch_pipeline(first))
    return false;
  if (sXrBatch.motion && !d.xrMotionMaterial)
    return true;
  // Bindless: textures, uniform and fog are the draw's own (its record).
  if (!sXrBatch.motion && sXrBatch.bindless)
    return d.dstAlpha == first.dstAlpha;
  // AURORA_XR_BATCH_IGNORE_TEXTURES=1 (a measurement: wrong textures and
  // texture sizes): eye draws join regardless of their textures and uniform,
  // the GPU time bindless batching could save at most.
  static const bool ignoreTextures = env_flag("AURORA_XR_BATCH_IGNORE_TEXTURES");
  if (ignoreTextures && !sXrBatch.motion)
    return d.dstAlpha == first.dstAlpha && d.immediateData.fogRangeBase == first.immediateData.fogRangeBase;
  return d.bindGroups.textureBindGroup == first.bindGroups.textureBindGroup && d.dstAlpha == first.dstAlpha &&
         d.uniformSharedKey == first.uniformSharedKey &&
         d.immediateData.fogRangeBase == first.immediateData.fogRangeBase;
}

bool joins(const DrawData& first, const DrawData& d) {
  if (sXrBatch.members.size() >= kBatchMaxMembers || sXrBatch.memberIndices + draw_index_count(d) > kBatchMaxIndices)
    return false;
  return compatible(first, d);
}

// Bytes in this frame's slot, 16-aligned (null: full).
uint8_t* slot_alloc(uint64_t bytes, uint64_t& offset) {
  auto& b = sXrBatch;
  const uint64_t at = (b.used + 15) & ~uint64_t{15};
  if (at + bytes > gfx::detail::XrBatchSlotSize)
    return nullptr;
  b.used = at + bytes;
  offset = (b.frameId % gfx::detail::XrBatchSlots) * gfx::detail::XrBatchSlotSize + at;
  return b.staging.data() + at;
}

void emit_batch(const wgpu::RenderPassEncoder& pass) {
  auto& b = sXrBatch;
  const auto& first = *b.members.front();
  const auto n = static_cast<uint32_t>(b.members.size());
  // Bindless records add the draw's texture slots (17-24) and sampler slots
  // (25-32).
  const uint32_t kRecordWords = b.bindless ? 33 : 17;
  static std::vector<uint32_t> slots;
  if (b.bindless) {
    slots.resize(size_t{n} * 16);
    for (uint32_t i = 0; i < n; ++i) {
      if (!bindless_slots(*b.members[i], slots.data() + size_t{i} * 16)) {
        for (const DrawData* d : b.members)
          render_direct(*d, pass);
        return;
      }
    }
  }
  uint64_t idxOffset = 0, tableOffset = 0;
  auto* idx = reinterpret_cast<uint32_t*>(slot_alloc(uint64_t{b.memberIndices} * 4, idxOffset));
  auto* table = idx != nullptr ? reinterpret_cast<uint32_t*>(slot_alloc(uint64_t{n} * (2 + kRecordWords) * 4, tableOffset))
                               : nullptr;
  if (table == nullptr || !gfx::bind_pipeline(batch_pipeline(first), pass)) {
    for (const DrawData* d : b.members)
      render_direct(*d, pass);
    return;
  }
  const auto tableWord = static_cast<uint32_t>(tableOffset / 4);
  uint32_t* records = table + n * 2;
  uint32_t written = 0;
  for (uint32_t i = 0; i < n; ++i) {
    const DrawData& d = *b.members[i];
    const uint32_t stride = d.posRange.size / d.vtxCount;
    const uint32_t vbase = d.posRange.offset / stride;
    uint32_t* rec = records + i * kRecordWords;
    static_assert(sizeof(d.immediateData) == 16 * 4);
    std::memcpy(rec, &d.immediateData, sizeof(d.immediateData));
    if (b.motion && gfx::xr_motion_zero_prev())
      rec[3] = 0; // xrPrevMtx
    rec[16] = d.uniformRange.offset / 4;
    if (b.bindless)
      std::memcpy(rec + 17, slots.data() + size_t{i} * 16, 16 * sizeof(uint32_t));
    table[i * 2] = vbase;
    table[i * 2 + 1] = tableWord + n * 2 + i * kRecordWords;
    if (d.indexCount == 0 && d.xrIndexCount == 0) {
      for (uint32_t v = 0; v < d.vtxCount; ++v)
        idx[written++] = vbase + v;
    } else {
      const gfx::Range& r = d.xrIndexCount != 0 ? d.xrIdxRange : d.idxRange;
      const uint32_t count = d.xrIndexCount != 0 ? d.xrIndexCount : d.indexCount;
      const auto* src = reinterpret_cast<const uint16_t*>(b.shadow + r.offset);
      for (uint32_t k = 0; k < count; ++k)
        idx[written++] = vbase + src[k];
    }
  }
  // Table entries by first vertex (draws push their vertices in order, so
  // this is almost always already so).
  bool sorted = true;
  for (uint32_t i = 1; i < n; ++i)
    sorted &= table[i * 2] > table[(i - 1) * 2];
  if (!sorted) {
    std::vector<std::pair<uint32_t, uint32_t>> entries(n);
    for (uint32_t i = 0; i < n; ++i)
      entries[i] = {table[i * 2], table[i * 2 + 1]};
    std::sort(entries.begin(), entries.end());
    for (uint32_t i = 0; i < n; ++i) {
      table[i * 2] = entries[i].first;
      table[i * 2 + 1] = entries[i].second;
    }
  }
  const auto& resources = gfx::detail::resources();
  const struct {
    uint32_t table, count, fogRangeBase, pad;
  } bimm{tableWord, n, first.immediateData.fogRangeBase, 0};
  pass.SetImmediates(0, &bimm, sizeof(bimm));
  if (b.bindless && !b.tableBound) {
    pass.SetResourceTable(sBindless.table);
    b.tableBound = true;
  }
  const std::array offsets{first.uniformRange.offset};
  pass.SetBindGroup(1, resources.uniformBindGroup, offsets.size(), offsets.data());
  if (b.bindless)
    pass.SetBindGroup(2, b.eyeGroup);
  else if (first.bindGroups.textureBindGroup)
    pass.SetBindGroup(2, gfx::find_bind_group(first.bindGroups.textureBindGroup));
  pass.SetIndexBuffer(resources.xrBatchBuffer, wgpu::IndexFormat::Uint32, idxOffset, uint64_t{written} * 4);
  pass.SetVertexBuffer(0, resources.storageBuffer);
  if (first.dstAlpha != UINT32_MAX) {
    const wgpu::Color color{0.f, 0.f, 0.f, first.dstAlpha / 255.f};
    pass.SetBlendConstant(&color);
  }
  pass.DrawIndexed(written);
  if (b.bindless) {
    // Back to the groups the other draws' layout has: textures at 2, the eye at 3.
    pass.SetBindGroup(2, g_emptyTextureBindGroup);
    pass.SetBindGroup(3, b.eyeGroup);
  }
  ++b.batches;
  b.batchedDraws += n;
}
} // namespace

void xr_batch_eye_group(const wgpu::BindGroup& group) { sXrBatch.eyeGroup = group; }

void xr_batch_begin(const gfx::detail::FramePacket& frame, bool multiview, bool motion) {
  auto& b = sXrBatch;
  b.active = multiview && gfx::detail::xr_batching() && frame.indexShadow != nullptr;
  b.motion = motion;
  b.shadow = frame.indexShadow.get();
  if (frame.frameId != b.frameId) {
    b.frameId = frame.frameId;
    b.used = b.uploaded = 0;
  }
  b.members.clear();
  b.memberIndices = 0;
  b.split.clear();
  static const bool sort = [] {
    const char* v = std::getenv("AURORA_XR_MOTION_SORT");
    return v == nullptr || *v != '0';
  }();
  b.sorting = b.active && motion && sort;
  b.eyeSorting = b.active && !motion && gfx::xr_eye_sort();
  b.bindless = b.active && !motion && gfx::xr_bindless();
  b.tableBound = false;
  if (b.bindless)
    sBindless.begin_frame(frame.frameId);
  b.bucketsUsed = 0;
}

namespace {
// The current run (b.members) as one draw, or alone if it's one.
void emit_members(const wgpu::RenderPassEncoder& pass) {
  auto& b = sXrBatch;
  if (b.members.empty())
    return;
  if (b.members.size() == 1)
    render_direct(*b.members.front(), pass);
  else
    emit_batch(pass);
  b.members.clear();
  b.memberIndices = 0;
}

void emit_bucket(XrBatcher::Bucket& bucket, const wgpu::RenderPassEncoder& pass) {
  auto& b = sXrBatch;
  b.members.swap(bucket.members);
  b.memberIndices = bucket.indices;
  emit_members(pass);
  bucket.members.clear();
  bucket.indices = 0;
}

// Motion replay, sorting: d joins the first bucket it's compatible with (a
// full one goes out first), else starts one.
void bucket_add(const DrawData& d, const wgpu::RenderPassEncoder& pass) {
  auto& b = sXrBatch;
  const uint32_t count = draw_index_count(d);
  for (size_t k = 0; k < b.bucketsUsed; ++k) {
    auto& bucket = b.buckets[k];
    if (!compatible(*bucket.members.front(), d))
      continue;
    if (bucket.members.size() >= kBatchMaxMembers || bucket.indices + count > kBatchMaxIndices) {
      emit_bucket(bucket, pass);
    }
    bucket.members.push_back(&d);
    bucket.indices += count;
    return;
  }
  if (b.bucketsUsed == b.buckets.size())
    b.buckets.emplace_back();
  auto& bucket = b.buckets[b.bucketsUsed++];
  bucket.members.assign(1, &d);
  bucket.indices = count;
}
} // namespace

void emit_buckets(const wgpu::RenderPassEncoder& pass) {
  auto& b = sXrBatch;
  for (size_t k = 0; k < b.bucketsUsed; ++k)
    emit_bucket(b.buckets[k], pass);
  b.bucketsUsed = 0;
}

void xr_batch_flush(const wgpu::RenderPassEncoder& pass) {
  emit_members(pass);
  emit_buckets(pass);
}

void xr_batch_end() {
  auto& b = sXrBatch;
  if (!b.active)
    return;
  b.active = false;
  if (b.used > b.uploaded) {
    const uint64_t base = (b.frameId % gfx::detail::XrBatchSlots) * gfx::detail::XrBatchSlotSize;
    webgpu::g_queue.WriteBuffer(gfx::detail::resources().xrBatchBuffer, base + b.uploaded, b.staging.data() + b.uploaded,
                             b.used - b.uploaded);
    b.uploaded = b.used;
  }
  // AURORA_XR_BATCH_STATS=1: replayed draws against draws actually issued.
  static const bool stats = env_flag("AURORA_XR_BATCH_STATS");
  if (stats && b.draws >= 50000) {
    Log.info("XR batching: {} replayed draws in {} batches of {} draws, the rest alone", b.draws, b.batches,
             b.batchedDraws);
    b.draws = b.batches = b.batchedDraws = 0;
  }
}
#endif

void render(const DrawData& data, const wgpu::RenderPassEncoder& pass) {
#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
  auto& b = sXrBatch;
  if (b.active) {
    // A straddling draw split by the clip (xrSolid*): its solid triangles
    // batch as an inside draw's do, and its triangles in the fade, which
    // dither, go alone (render_direct's order: those, then the solid ones).
    // Icicle Mountain's segments cross the box's fade all the time: 60% of
    // its eye draws went out one by one, Battlefield's 23%.
    if (data.xrSolidIndexCount != 0 && data.xrBatchPipeline != 0 && gfx::xr_multiview_replay() &&
        !gfx::xr_motion_replay()) {
      DrawData& edge = b.split.emplace_back(data);
      edge.xrSolidIndexCount = 0;
      edge.xrBatchPipeline = 0;
      DrawData& solid = b.split.emplace_back(data);
      solid.xrPipeline = data.xrSolidPipeline;
      solid.xrIdxRange = data.xrSolidIdxRange;
      solid.xrIndexCount = data.xrSolidIndexCount;
      solid.xrSolidIndexCount = 0;
      render(edge, pass);
      render(solid, pass);
      return;
    }
    ++b.draws;
    if (b.sorting && gfx::xr_multiview_replay()) {
      // Order doesn't matter here: what can't batch goes out now.
      if (batchable(data))
        bucket_add(data, pass);
      else
        render_direct(data, pass);
      return;
    }
    if (b.eyeSorting && gfx::xr_multiview_replay() && !gfx::xr_motion_replay()) {
      if (data.xrReorderable && !xr_link_filter(data)) {
        emit_members(pass); // a run before it keeps its place
        if (batchable(data))
          bucket_add(data, pass);
        else
          render_direct(data, pass); // reorderable too: the buckets stay open
        return;
      }
      emit_buckets(pass); // anything else comes after every bucketed draw
    }
    if (!gfx::xr_multiview_replay() || !batchable(data) ||
        (!gfx::xr_motion_replay() && xr_link_filter(data))) {
      xr_batch_flush(pass);
      render_direct(data, pass);
      return;
    }
    if (!b.members.empty() && !joins(*b.members.front(), data))
      xr_batch_flush(pass);
    b.members.push_back(&data);
    b.memberIndices += draw_index_count(data);
    return;
  }
#endif
  render_direct(data, pass);
}

} // namespace aurora::gx
