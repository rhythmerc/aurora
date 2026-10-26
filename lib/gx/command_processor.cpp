#include <bit>
#include <cstdlib>
#include <cstring>
#include <string>
#include "command_processor.hpp"

#include "../gfx/depth_peek.hpp"
#include "../gfx/frame_packet.hpp"
#include "../gfx/pipeline_cache.hpp"
#include "../gfx/recording.hpp"
#include "../internal.hpp"
#include "dolphin/gd/GDGeometry.h"
#include "dolphin/gx/GXAurora.h"
#include "gx.hpp"
#include "pipeline.hpp"
#include "regs.hpp"
#include "shader_info.hpp"
#include "texture.hpp"
#include "../gfx/texture_convert.hpp"

#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <tracy/Tracy.hpp>
#include <xxhash.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <span>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

namespace aurora::gx::fifo {
namespace {
constexpr Module Log{"aurora::gx::fifo"};

// Triangle indices for a draw, written straight into the buffer: appending
// them one u16 at a time made each a capacity check and a memcpy call.
u16 prepare_idx_buffer(ByteBuffer& buf, GXPrimitive prim, u16 vtxStart, u16 vtxCount) noexcept {
  u16 numIndices = 0;
  if (prim == GX_QUADS) {
    // As before, a partial last quad still makes its two triangles.
    numIndices = static_cast<u16>(((vtxCount + 3) / 4) * 6);
    auto* out = reinterpret_cast<u16*>(buf.append_uninit(numIndices * sizeof(u16)));
    for (u16 v = 0; v < vtxCount; v += 4) {
      const u16 i0 = vtxStart + v;
      *out++ = i0;
      *out++ = i0 + 1;
      *out++ = i0 + 2;
      *out++ = i0 + 2;
      *out++ = i0 + 3;
      *out++ = i0;
    }
  } else if (prim == GX_TRIANGLES) {
    numIndices = vtxCount;
    auto* out = reinterpret_cast<u16*>(buf.append_uninit(numIndices * sizeof(u16)));
    for (u16 v = 0; v < vtxCount; ++v)
      out[v] = vtxStart + v;
  } else if (prim == GX_TRIANGLEFAN || prim == GX_TRIANGLESTRIP) {
    const bool fan = prim == GX_TRIANGLEFAN;
    numIndices = vtxCount < 3 ? vtxCount : static_cast<u16>((vtxCount - 3) * 3 + 3);
    auto* out = reinterpret_cast<u16*>(buf.append_uninit(numIndices * sizeof(u16)));
    for (u16 v = 0; v < vtxCount; ++v) {
      const u16 idx = vtxStart + v;
      if (v < 3) {
        *out++ = idx;
      } else if (fan) {
        *out++ = vtxStart;
        *out++ = idx - 1;
        *out++ = idx;
      } else if ((v & 1) == 0) {
        *out++ = idx - 2;
        *out++ = idx - 1;
        *out++ = idx;
      } else {
        *out++ = idx - 1;
        *out++ = idx - 2;
        *out++ = idx;
      }
    }
  } else if (prim == GX_LINES || prim == GX_LINESTRIP || prim == GX_POINTS) {
    buf.reserve_extra(6 * sizeof(u16));
    buf.append<u16>(0);
    buf.append<u16>(1);
    buf.append<u16>(3);
    buf.append<u16>(3);
    buf.append<u16>(2);
    buf.append<u16>(0);
    numIndices = 6;
  } else
    UNLIKELY FATAL("unsupported primitive type {}", static_cast<u32>(prim));
  return numIndices;
}

// GX FIFO opcodes - use CP_ prefix to avoid clashing with GXCommandList.h macros
constexpr u8 CP_CMD_NOP = GX_NOP;
constexpr u8 CP_CMD_LOAD_CP_REG = GX_LOAD_CP_REG;
constexpr u8 CP_CMD_LOAD_XF_REG = GX_LOAD_XF_REG;
constexpr u8 CP_CMD_LOAD_INDX_A = GX_LOAD_INDX_A;
constexpr u8 CP_CMD_LOAD_INDX_B = GX_LOAD_INDX_B;
constexpr u8 CP_CMD_LOAD_INDX_C = GX_LOAD_INDX_C;
constexpr u8 CP_CMD_LOAD_INDX_D = GX_LOAD_INDX_D;
constexpr u8 CP_CMD_CALL_DL = GX_CMD_CALL_DL;
constexpr u8 CP_CMD_INVAL_VTX = GX_CMD_INVL_VC;
constexpr u8 CP_CMD_LOAD_BP_REG = GX_LOAD_BP_REG & GX_OPCODE_MASK;

// Primitive type mask
constexpr u8 CP_OPCODE_MASK = GX_OPCODE_MASK;
constexpr u8 CP_VAT_MASK = GX_VAT_MASK;

struct FogRangeLutKey {
  std::array<u16, 10> rangeK;
  f32 rangeCenter;
  f32 renderWidth;
  u32 targetWidth;

  bool operator==(const FogRangeLutKey&) const = default;
};

struct FogRangeLutEntry {
  FogRangeLutKey key;
  std::vector<f32> factors;
};

constexpr size_t MaxFogRangeLuts = 32;
std::vector<FogRangeLutEntry> sFogRangeLuts;

struct DrawCache {
  uint64_t targetLayoutKey = 0;
  PipelineConfig config{};
  ShaderInfo shaderInfo{};
  gfx::PipelineRef pipelineRef{};
  // XR multiview twin of pipelineRef, for world draws (and what it was made for).
  gfx::PipelineRef xrPipelineRef{};
  gfx::PipelineRef xrForPipeline{};
  uint64_t xrForLayout = 0;
  // Its dithering twin, for clipped world draws that straddle a clip.
  gfx::PipelineRef xrSoftRef{};
  gfx::PipelineRef xrSoftForPipeline{};
  uint64_t xrSoftForLayout = 0;
  // Its motion-vector twins (Application SpaceWarp): [clipped + 2 * cutout],
  // for plain opaque draws, ones straddling a clip, and alpha-tested ones.
  struct XrTwin {
    gfx::PipelineRef ref{};
    gfx::PipelineRef forPipeline{};
    uint64_t forLayout = 0;
  } xrMotionTwin[6];
  // XR batching: the batched twins of the eye and motion pipelines.
  gfx::PipelineRef xrBatchRef{}, xrBatchForPipeline{};
  uint64_t xrBatchForLayout = 0;
  gfx::PipelineRef xrMotionBatchRef{}, xrMotionBatchForPipeline{};
  uint64_t xrMotionBatchForLayout = 0;
  GXBindGroups bindGroups{};
  uint64_t uniformSharedKey = 0; // build_uniform's key for uniformRange
  uint64_t bindGeneration = 0;
  GXVtxFmt fmt = GX_MAX_VTXFMT;
  u8 lineMode = 0;
  bool hasPipeline = false;
  gfx::Range uniformRange{};
  gfx::Range fogRange{};
  FogRangeLutKey fogRangeKey{};
  bool hasFogRange = false;
  GXVtxFmt lastDrawFmt = GX_MAX_VTXFMT;
};
DrawCache sDrawCache;

FogRangeLutKey fog_range_lut_key() noexcept {
  const auto& state = g_gxState.fog;
  const f32 logicalWidth = std::max(g_gxState.logicalViewport.width, 1.f);
  const f32 renderWidth = std::max(g_gxState.renderViewport.width, 1.f);
  return {
      .rangeK = state.rangeK,
      .rangeCenter = ((static_cast<f32>(state.rangeCenter) - g_gxState.logicalViewport.left) / logicalWidth) * 2.f -
                     1.f + (g_gxState.renderViewport.left / renderWidth) * 2.f,
      .renderWidth = renderWidth,
      .targetWidth = gfx::get_render_target_size().x,
  };
}

std::vector<f32> build_fog_range_lut(const FogRangeLutKey& key) {
  std::array<f32, 10> rangeK;
  for (u32 i = 0; i < rangeK.size(); ++i) {
    const u32 source = (i & ~1u) | (1u - (i & 1u));
    rangeK[i] = static_cast<f32>(key.rangeK[source]) / 64.f;
  }

  std::vector<f32> lut(key.targetWidth);
  for (u32 x = 0; x < key.targetWidth; ++x) {
    const f32 screenX = ((static_cast<f32>(x) + 0.5f) / key.renderWidth) * 2.f - 1.f;
    const f32 offset = screenX - key.rangeCenter;
    const f32 rangeIndex = std::clamp(9.f - std::abs(offset) * 9.f, 0.f, 9.f);
    const u32 lower = static_cast<u32>(rangeIndex);
    const u32 upper = std::min(lower + 1, 9u);
    const f32 fraction = rangeIndex - static_cast<f32>(lower);
    const f32 k = std::max(rangeK[lower] * (1.f - fraction) + rangeK[upper] * fraction, 0.000001f);
    lut[x] = std::sqrt(offset * offset + k * k) / k;
  }
  return lut;
}

const std::vector<f32>& resolve_fog_range_lut(const FogRangeLutKey& key) {
  for (const auto& entry : sFogRangeLuts) {
    if (entry.key == key) {
      return entry.factors;
    }
  }
  if (sFogRangeLuts.size() == MaxFogRangeLuts) {
    sFogRangeLuts.erase(sFogRangeLuts.begin());
  }
  sFogRangeLuts.emplace_back(FogRangeLutEntry{key, build_fog_range_lut(key)});
  return sFogRangeLuts.back().factors;
}

gfx::Range push_fog_range_lut(const FogRangeLutKey& key) {
  const auto& lut = resolve_fog_range_lut(key);
  return gfx::push_storage(reinterpret_cast<const u8*>(lut.data()), lut.size() * sizeof(f32));
}

u8 line_mode_for_prim(GXPrimitive prim) noexcept {
  switch (prim) {
  case GX_LINES:
    return 1;
  case GX_LINESTRIP:
    return 2;
  case GX_POINTS:
    return 3;
  default:
    return 0;
  }
}
} // namespace

static uint32_t sMarkerTag = 0;
static uint32_t sLinkTag = 0; // AURORA_XR_LINK_STATS: the game's GX link + 1 ("L<n>" markers)

static void handle_draw(u8 cmd, ByteReader& reader) noexcept;
static void handle_aurora(ByteReader& reader) noexcept;
u64 decode_misses() noexcept;

// AURORA_FIFO_PROBE=1: a frame whose commands kept the FIFO thread busy over
// AURORA_FIFO_PROBE_MS (10) logs where it went (draws, Aurora's commands, the
// rest), its slowest draw, and the decode misses and pipelines it made. (A
// SpaceWarp late drop is the world's draws recorded late: on Yoshi's Story
// every few seconds, the game thread done in time.)
struct FifoProbe {
  bool on = false;
  int64_t thresholdNs = 10'000'000;
  uint64_t seq = UINT64_MAX;
  int64_t busyNs = 0, drawNs = 0, auroraNs = 0;
  uint32_t draws = 0;
  int64_t slowestNs = 0;
  u16 slowestVerts = 0;
  u8 slowestCmd = 0;
  uint32_t slowestLink = 0;
  u64 missesAt = 0;
  uint32_t pipelinesAt = 0;
  int logged = 0;
  int64_t windowNs = 0;
} sFifoProbe;

static int64_t probe_now() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// A new frame's commands: the last frame's line, if it was slow.
static void fifo_probe_frame(int64_t now) noexcept {
  auto& p = sFifoProbe;
  const uint64_t seq = gfx::detail::recording_seq();
  if (seq == p.seq)
    return;
  if (p.seq != UINT64_MAX && p.busyNs > p.thresholdNs) {
    if (now - p.windowNs > 10'000'000'000) {
      p.windowNs = now;
      p.logged = 0;
    }
    if (p.logged++ < 12)
      Log.info("FIFO probe: {:.1f} ms busy (draws {:.1f} in {}, Aurora's commands {:.1f}, the rest {:.1f}); slowest "
               "draw {:.2f} ms ({} vertices, command 0x{:02X}, link {}); {} decode misses, {} pipelines made",
               p.busyNs / 1e6, p.drawNs / 1e6, p.draws, p.auroraNs / 1e6, (p.busyNs - p.drawNs - p.auroraNs) / 1e6,
               p.slowestNs / 1e6, p.slowestVerts, p.slowestCmd, static_cast<int>(p.slowestLink) - 1,
               decode_misses() - p.missesAt, gfx::pipelines_created() - p.pipelinesAt);
  }
  p.seq = seq;
  p.busyNs = p.drawNs = p.auroraNs = p.slowestNs = 0;
  p.draws = 0;
  p.missesAt = decode_misses();
  p.pipelinesAt = gfx::pipelines_created();
}

static ProcessResult process_commands(const u8* data, u32 size) noexcept;

ProcessResult process(const u8* data, u32 size) noexcept {
  auto& p = sFifoProbe;
  static const bool on = [] {
    const char* v = std::getenv("AURORA_FIFO_PROBE");
    const char* ms = std::getenv("AURORA_FIFO_PROBE_MS");
    if (ms != nullptr && std::atof(ms) > 0)
      sFifoProbe.thresholdNs = static_cast<int64_t>(std::atof(ms) * 1e6);
    return v != nullptr && *v == '1';
  }();
  p.on = on;
  if (!on)
    return process_commands(data, size);
  const int64_t start = probe_now();
  fifo_probe_frame(start);
  const ProcessResult result = process_commands(data, size);
  p.busyNs += probe_now() - start;
  return result;
}

static ProcessResult process_commands(const u8* data, u32 size) noexcept {
  ZoneScoped;
  ByteReader reader{{data, size}};

  while (!reader.empty()) {
    const u8 cmd = reader.read<u8>();
    u8 opcode = cmd & CP_OPCODE_MASK;

    switch (opcode) {
    case CP_CMD_NOP:
      continue;

    case CP_CMD_LOAD_BP_REG: {
      const u32 value = reader.read<u32>();
      handle_bp(value);
      if (reg_get(value, 8, 24) == GX_BP_REG_DRAWDONE) {
        return {static_cast<u32>(reader.offset()), true};
      }
      break;
    }

    case CP_CMD_LOAD_CP_REG: {
      const u8 addr = reader.read<u8>();
      handle_cp(addr, reader.read<u32>());
      break;
    }

    case CP_CMD_LOAD_XF_REG: {
      const u32 header = reader.read<u32>();
      const u32 count = ((header >> 16) & 0xFFFF) + 1;
      const u16 addr = header & 0xFFFF;
      handle_xf(addr, reader.take(count * sizeof(u32)));
      break;
    }

    case CP_CMD_LOAD_INDX_A:
    case CP_CMD_LOAD_INDX_B:
    case CP_CMD_LOAD_INDX_C:
    case CP_CMD_LOAD_INDX_D: {
      ZoneScopedN("LOAD_INDX");
      const u32 arrayType = GX_POS_MTX_ARRAY + (opcode - CP_CMD_LOAD_INDX_A) / 0x08;
      const u16 srcArrayIdx = reader.read<u16>();
      const u16 addrLen = reader.read<u16>();

      const u16 len = (addrLen >> 12) + 1;
      const u16 dstAddr = addrLen & 0x0FFF;
      auto const& array = g_gxState.arrays[arrayType];
      const u32 srcOffset = static_cast<u32>(srcArrayIdx) * array.stride;
      const u32 srcSize = static_cast<u32>(len) * sizeof(u32);
      AURORA_ASSERT(array.data != nullptr, "indexed XF load from unmapped array {}", arrayType);
      AURORA_ASSERT(srcOffset <= array.size && srcSize <= array.size - srcOffset,
                    "indexed XF load outside array {}: offset={}, size={}, array size={}", arrayType, srcOffset,
                    srcSize, array.size);
      auto const* srcData = static_cast<const u8*>(array.data) + srcOffset;
      if (!copy_xf_data(dstAddr, srcData, len, array.le ? std::endian::little : std::endian::big)) {
#ifndef NDEBUG
        Log.debug("Unimplemented indexed XF load (opcode 0x{:02X}, dstAddr=%04x)", opcode, dstAddr);
#endif
      }
      break;
    }

    case CP_CMD_CALL_DL: {
      // Call display list: 8 bytes (address + size)
      Log.warn("Ignoring nested GX_CMD_CALL_DL");
      reader.skip(8);
      break;
    }

    case CP_CMD_INVAL_VTX: {
      for (auto& array : g_gxState.arrays) {
        array.cachedRange = {};
        array.hashed = false;
      }
      g_gxState.dirty |= DirtyImmediates;
      break;
    }

    case GX_AURORA: {
      if (sFifoProbe.on) {
        const int64_t start = probe_now();
        handle_aurora(reader);
        sFifoProbe.auroraNs += probe_now() - start;
      } else {
        handle_aurora(reader);
      }
      break;
    }

    // Draw commands: 0x80-0xBF
    case GX_DRAW_QUADS:
    case GX_DRAW_TRIANGLES:
    case GX_DRAW_TRIANGLE_STRIP:
    case GX_DRAW_TRIANGLE_FAN:
    case GX_DRAW_LINES:
    case GX_DRAW_LINE_STRIP:
    case GX_DRAW_POINTS: {
      handle_draw(cmd, reader);
      break;
    }

    default:
      // Check if it's a draw command (0x80-0xBF range)
      if (cmd >= 0x80) {
        handle_draw(cmd, reader);
      } else {
        // Hex dump surrounding bytes for debugging
        {
          const size_t pos = reader.offset();
          size_t dumpStart = (pos > 17) ? pos - 17 : 0;
          size_t dumpEnd = (pos + 16 < size) ? pos + 16 : size;
          std::string hex;
          for (size_t i = dumpStart; i < dumpEnd; i++) {
            if (i == pos - 1)
              hex += fmt::format("[{:02x}]", data[i]);
            else
              hex += fmt::format(" {:02x}", data[i]);
          }
          Log.error("  hex dump (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
        }
        FATAL("command_processor: unknown opcode 0x{:02X} at pos {}", cmd, reader.offset() - 1);
      }
      break;
    }
  }
  return {size, false};
}

[[noreturn]] static void handle_draw_overrun(size_t totalVtxBytes, const ByteReader& reader) noexcept {
  // Hex dump around the draw command for debugging
  const size_t pos = reader.offset();
  const size_t size = reader.size();
  const u8* data = reader.data();
  size_t cmdPos = pos - 2 - 1; // opcode byte position (before vtxCount and pos++)
  size_t dumpStart = (cmdPos > 16) ? cmdPos - 16 : 0;
  size_t dumpEnd = (cmdPos + 32 < size) ? cmdPos + 32 : size;
  std::string hex;
  for (size_t i = dumpStart; i < dumpEnd; i++) {
    if (i == cmdPos)
      hex += fmt::format("[{:02x}]", data[i]);
    else
      hex += fmt::format(" {:02x}", data[i]);
  }
  Log.error("  hex dump around draw cmd (pos {}-{}):{}", dumpStart, dumpEnd - 1, hex);
  FATAL("draw vertex data overrun: need {} bytes at pos {}, have {}", totalVtxBytes, pos, reader.remaining());
}

static u32 calc_vtx_size(GXVtxFmt fmt) noexcept {
  u32 vtxSize = 0;
  const auto& vtxFmt = g_gxState.vtxFmts[fmt];
  for (int i = GX_VA_PNMTXIDX; i <= GX_VA_TEX7; ++i) {
    const auto& attrFmt = vtxFmt.attrs[i];
    switch (g_gxState.vtxDesc[i]) {
    case GX_NONE:
      break;
    case GX_DIRECT: {
      const auto attr = static_cast<GXAttr>(i);
      vtxSize += comp_type_size(attr, attrFmt.type) * comp_cnt_count(attr, attrFmt.cnt);
      break;
    }
    case GX_INDEX8:
      vtxSize += i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 ? 3 : 1;
      break;
    case GX_INDEX16:
      vtxSize += i == GX_VA_NRM && attrFmt.cnt == GX_NRM_NBT3 ? 6 : 2;
      break;
    }
  }
  g_gxState.lastVtxFmt = fmt;
  g_gxState.lastVtxSize = vtxSize;
  return vtxSize;
}

// AURORA_POS_DECODE: positions decoded to float3 on the CPU, exactly as the
// shader's fetch_* helpers read them (shader.cpp), for a vertex buffer the
// GPU fetches in hardware. `raw` is the draw's GX vertex stream (big-endian
// indices and direct data); indexed positions come from the bound array.
static float decode_component(const u8* p, u8 compType, float scale, bool le) noexcept {
  const auto u16v = [&] { return le ? u16(p[0] | (p[1] << 8)) : u16((p[0] << 8) | p[1]); };
  switch (compType) {
  case GX_U8:
    return static_cast<float>(p[0]) * scale;
  case GX_S8:
    return static_cast<float>(static_cast<s8>(p[0])) * scale;
  case GX_U16:
    return static_cast<float>(u16v()) * scale;
  case GX_S16:
    return static_cast<float>(static_cast<s16>(u16v())) * scale;
  case GX_F32: {
    u32 v = le ? (u32(p[0]) | (u32(p[1]) << 8) | (u32(p[2]) << 16) | (u32(p[3]) << 24))
               : ((u32(p[0]) << 24) | (u32(p[1]) << 16) | (u32(p[2]) << 8) | u32(p[3]));
    return std::bit_cast<float>(v);
  }
  default:
    return 0.f;
  }
}

static float decode_component(const u8* p, u8 compType, u8 frac, bool le) noexcept {
  return decode_component(p, compType, 1.f / static_cast<float>(1u << frac), le);
}

// Where one attribute of one vertex lives (NBT slice `slice`), and its byte
// order: the same address attr_address/attr_load_nbt_slice build in the
// shader. Null when an index runs past its array.
static const u8* decoded_source(const AttrConfig& m, GXAttr attr, u32 slice, const u8* vtx, bool& le,
                                u32 compSize) noexcept {
  const u32 within = attr == GX_VA_NRM ? slice * 3 * compSize : 0;
  if (m.attrType == GX_DIRECT) {
    le = false;
    return vtx + m.offset + within;
  }
  const u32 dlExtra = m.nbt3 ? (m.attrType == GX_INDEX8 ? slice : slice * 2) : 0;
  const u8* ip = vtx + m.offset + dlExtra;
  const u32 idx = m.attrType == GX_INDEX8 ? ip[0] : (u32(ip[0]) << 8) | ip[1];
  const auto& array = g_gxState.arrays[attr];
  const u32 elemSize = attr == GX_VA_CLR0 || attr == GX_VA_CLR1 ? compSize : compSize * std::min<u32>(m.cnt, 3);
  const size_t at = static_cast<size_t>(idx) * m.stride + within;
  if (array.data == nullptr || at + elemSize > array.size) {
    return nullptr;
  }
  le = m.le;
  return static_cast<const u8*>(array.data) + at;
}

static const u8* decoded_source(const AttrConfig& m, GXAttr attr, u32 slice, const u8* vtx, bool& le) noexcept {
  return decoded_source(m, attr, slice, vtx, le, comp_type_size(attr, static_cast<GXCompType>(m.compType)));
}

// fetch_rgb565 .. fetch_rgba8 (shader.cpp).
// Reciprocal multiplies, not divides: per vertex on the FIFO thread.
static void decode_color(const u8* p, u8 compType, bool le, float* out) noexcept {
  const auto u16v = [&] { return le ? u32(p[0] | (p[1] << 8)) : u32((p[0] << 8) | p[1]); };
  switch (compType) {
  case GX_RGB565: {
    const u32 v = u16v();
    out[0] = float((v >> 11) & 0x1F) * (1.f / 31.f), out[1] = float((v >> 5) & 0x3F) * (1.f / 63.f),
    out[2] = float(v & 0x1F) * (1.f / 31.f), out[3] = 1.f;
    break;
  }
  case GX_RGB8:
  case GX_RGBX8:
    out[0] = p[0] * (1.f / 255.f), out[1] = p[1] * (1.f / 255.f), out[2] = p[2] * (1.f / 255.f), out[3] = 1.f;
    break;
  case GX_RGBA4: {
    const u32 v = u16v();
    out[0] = float((v >> 12) & 0xF) * (1.f / 15.f), out[1] = float((v >> 8) & 0xF) * (1.f / 15.f),
    out[2] = float((v >> 4) & 0xF) * (1.f / 15.f), out[3] = float(v & 0xF) * (1.f / 15.f);
    break;
  }
  case GX_RGBA6: {
    const u32 v = le ? u32(p[0] | (p[1] << 8) | (p[2] << 16)) : u32((p[0] << 16) | (p[1] << 8) | p[2]);
    out[0] = float((v >> 18) & 0x3F) * (1.f / 63.f), out[1] = float((v >> 12) & 0x3F) * (1.f / 63.f),
    out[2] = float((v >> 6) & 0x3F) * (1.f / 63.f), out[3] = float(v & 0x3F) * (1.f / 63.f);
    break;
  }
  default: // GX_RGBA8
    out[0] = p[0] * (1.f / 255.f), out[1] = p[1] * (1.f / 255.f), out[2] = p[2] * (1.f / 255.f), out[3] = p[3] * (1.f / 255.f);
    break;
  }
}

// One draw's vertices for the GPU's own vertex fetch: positions only
// (decodedPos, float3), or every attribute interleaved (decodedAll,
// decoded_layout). `raw` is the draw's GX vertex stream. Decoded straight
// into the storage pool, each attribute's formats worked out once per draw:
// this is most of the FIFO thread's time on the Quest (Pokemon Stadium's
// fire form spent 21% of it in memcpy here, and 5% sizing components).
static DecodedLayout decode_layout_for(const ShaderConfig& config) noexcept {
  DecodedLayout layout;
  if (!config.decodedAll || !decoded_layout(config, layout)) {
    layout = {};
    layout.attrs[0] = {GX_VA_POS, 0, DecodedFormat::F32x3, 0};
    layout.count = 1;
    layout.stride = 12;
  }
  return layout;
}

// The primitive of the draw being decoded (draw_prim), whose triangles the
// smooth normals are made from; 0 for a draw with its own index buffer
// (GX_AURORA_DRAW_INDEXED), which gets none.
u8 sDrawPrim = 0;

// An additive draw (glows, fire, lasers, screens' light): light of its own,
// which the room doesn't light.
static bool xr_additive(const PipelineConfig& c) noexcept {
  return c.blendMode == GX_BM_BLEND && c.blendFacDst == GX_BL_ONE;
}

// XR room lighting's fit of the stage's baked light (SMOOTH shading): the
// luminance of the vertex colours the stage's unlit draws are lit by, as a
// function of the outward normal in the stage's world, c0 + c1 x + c2 y +
// c3 z, least squares weighted by area. Made from the draws as they first
// decode, and fixed once 30 frames bring no more (a new generation in the
// decode key: they decode again, now with each vertex's de-light factor,
// the fit up over the fit at its normal: the baked light divided out, as
// the light from above; as much of it as the fit explains). AURORA_XR_ROOM_FIT=0
// leaves it out (an A/B). aurora_xr_set_stage starts another.
// The same window gathers the faces of the stage's translucent draws for
// their soft normals (smooth_normals).
struct RoomFit {
  u32 epoch = 0; // gfx::xr_room_fit_epoch() it's for
  u32 gen = 1;   // in the smooth draws' decode key
  bool frozen = false, valid = false;
  double ata[4][4]{}, atb[4]{}, btb = 0.0;
  u64 verts = 0;
  u32 draws = 0, lastAdd = 0;
  float c[4]{};
  float trust = 0.f; // how much of the fit is used, by how much it explains
  // Soft normals: each translucent face once (by its points), as its mesh
  // (position array), centre and normal at twice its area; when frozen,
  // each mesh's radius and the faces sorted into cells of that size, with
  // each cell's run of them.
  struct SoftFace {
    u64 mesh, cell = 0;
    std::array<float, 3> c, a;
  };
  absl::flat_hash_set<u64> softSeen;
  std::vector<SoftFace> soft;
  absl::flat_hash_map<u64, float> softRadius;
  absl::flat_hash_map<u64, std::pair<u32, u32>> softCells;
};
RoomFit sRoomFit;

// The soft normals' cell holding `p`, moved by `d` cells.
static u64 room_soft_cell(u64 mesh, const std::array<float, 3>& p, float radius,
                          const std::array<s32, 3>& d = {}) noexcept {
  const std::array<s32, 3> c{static_cast<s32>(std::floor(p[0] / radius)) + d[0],
                             static_cast<s32>(std::floor(p[1] / radius)) + d[1],
                             static_cast<s32>(std::floor(p[2] / radius)) + d[2]};
  return XXH3_64bits_withSeed(c.data(), sizeof(c), mesh);
}

// A soft normal's reach: an eighth of its mesh's extent (Kongo Jungle's
// foam: wide enough to pass over its ribbons' folds, not so wide that the
// river's top and the falls' face run together).
static void room_soft_freeze(RoomFit& f) noexcept {
  absl::flat_hash_map<u64, std::array<float, 6>> box;
  for (const auto& sf : f.soft) {
    auto [it, fresh] = box.try_emplace(sf.mesh, std::array{sf.c[0], sf.c[1], sf.c[2], sf.c[0], sf.c[1], sf.c[2]});
    for (int k = 0; k < 3; ++k) {
      it->second[k] = std::min(it->second[k], sf.c[k]);
      it->second[k + 3] = std::max(it->second[k + 3], sf.c[k]);
    }
  }
  for (const auto& [mesh, b] : box) {
    const float dx = b[3] - b[0], dy = b[4] - b[1], dz = b[5] - b[2];
    f.softRadius[mesh] = std::max(std::sqrt(dx * dx + dy * dy + dz * dz) / 8.f, 1e-3f);
  }
  for (auto& sf : f.soft)
    sf.cell = room_soft_cell(sf.mesh, sf.c, f.softRadius[sf.mesh]);
  std::sort(f.soft.begin(), f.soft.end(), [](const auto& a, const auto& b) { return a.cell < b.cell; });
  for (u32 i = 0; i < f.soft.size();) {
    u32 j = i + 1;
    while (j < f.soft.size() && f.soft[j].cell == f.soft[i].cell)
      ++j;
    f.softCells.emplace(f.soft[i].cell, std::pair{i, j});
    i = j;
  }
  if (!f.soft.empty())
    Log.info("XR room soft normals: {} translucent faces, {} meshes", f.soft.size(), f.softRadius.size());
}

static void room_fit_tick() noexcept {
  auto& f = sRoomFit;
  if (const u32 epoch = gfx::xr_room_fit_epoch(); epoch != f.epoch) {
    const u32 gen = f.gen + 1;
    f = {};
    f.epoch = epoch;
    f.gen = gen;
    return;
  }
  if (f.frozen || (f.verts == 0 && f.soft.empty()) || g_gxState.frameSerial - f.lastAdd < 30)
    return;
  f.frozen = true;
  // The draws decode again: translucent ones soft, and de-lit where the
  // fit holds.
  ++f.gen;
  room_soft_freeze(f);
  if (f.verts == 0)
    return;
  // Normal equations, a little ridge on the directional terms.
  double m[4][5];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j)
      m[i][j] = f.ata[i][j] + (i == j && i > 0 ? 1e-3 * f.ata[0][0] : 0.0);
    m[i][4] = f.atb[i];
  }
  for (int i = 0; i < 4; ++i) {
    int piv = i;
    for (int r = i + 1; r < 4; ++r)
      if (std::abs(m[r][i]) > std::abs(m[piv][i]))
        piv = r;
    if (!(std::abs(m[piv][i]) > 1e-12))
      return;
    std::swap(m[i], m[piv]);
    for (int r = 0; r < 4; ++r) {
      if (r == i)
        continue;
      const double k = m[r][i] / m[i][i];
      for (int j = i; j < 5; ++j)
        m[r][j] -= k * m[i][j];
    }
  }
  double c[4];
  for (int i = 0; i < 4; ++i)
    c[i] = m[i][4] / m[i][i];
  double sse = f.btb;
  for (int i = 0; i < 4; ++i) {
    sse -= 2.0 * c[i] * f.atb[i];
    for (int j = 0; j < 4; ++j)
      sse += c[i] * f.ata[i][j] * c[j];
  }
  const double sst = f.btb - f.atb[0] * f.atb[0] / f.ata[0][0];
  for (int i = 0; i < 4; ++i)
    f.c[i] = static_cast<float>(c[i]);
  static const bool use = [] {
    const char* v = std::getenv("AURORA_XR_ROOM_FIT");
    return v == nullptr || *v != '0';
  }();
  // A fit that explains little is mostly the stage's colours varying with
  // where they face, not light (Peach's Castle, R2 0.11): none of it under
  // 0.15, all of it from 0.45 (Pokemon Stadium, 0.44).
  const double r2 = sst > 0.0 ? 1.0 - sse / sst : 0.0;
  f.trust = std::clamp(static_cast<float>((r2 - 0.15) / 0.3), 0.f, 1.f);
  f.valid = use && f.c[0] + f.c[2] > 0.02f && f.trust > 0.f;
  Log.info("XR room fit: {} draws, {} vertices: luminance {:.3f} {:+.3f} x {:+.3f} y {:+.3f} z (up {:.3f}, down {:.3f}, "
           "side {:.3f}), R2 {:.2f}, used {:.0f}%",
           f.draws, f.verts, c[0], c[1], c[2], c[3], c[0] + c[2], c[0] - c[2], c[0], r2, f.valid ? f.trust * 100.f : 0.f);
}

// XR room lighting's SMOOTH shading (kDecodedXrNrm): each vertex's normal,
// in model space, from the draw's own triangles. Vertices at the same place
// (same position and matrix) are one point, and a vertex takes the faces
// around its point within the crease angle (AURORA_XR_ROOM_CREASE, degrees,
// default 50) of the faces it's a corner of, weighted by area: a curved
// surface shades smoothly, a corner stays sharp. Faces are summed turned to
// agree, so a mesh's mixed windings don't cancel; the sum is then turned
// outward by the winding of the vertex's own faces (front faces clockwise
// on screen, GX's). Made on a decode cache miss only; a draw's own
// triangles only (seams between draws keep a crease). A translucent draw
// (indexed positions), once the room fit's window has gathered its mesh,
// takes a soft normal instead: its mesh's faces within a radius, from any
// of its draws, by how near they are, connected or not, without the crease.
// Kongo Jungle's waterfall foam is ribbons of 5 to 31 vertices, each its
// own draw, folded and overlapping, sharing no corners: shaded as they
// face, every ribbon showed; the effect needs the surface they make. In w,
// the vertex's de-light factor (RoomFit), 1 where its draw isn't lit by
// vertex colours.
namespace {
u64 array_content_hash(GXAttr attr) noexcept;
}
static void smooth_normals(const ShaderConfig& config, const DecodedLayout& layout, u16 vtxCount, u8* base,
                           u8 nrmOffset) noexcept {
  static const float cosCrease = [] {
    const char* v = std::getenv("AURORA_XR_ROOM_CREASE");
    const float deg = v != nullptr ? std::strtof(v, nullptr) : 50.f;
    return std::cos(std::clamp(deg, 0.f, 180.f) * 0.017453292f);
  }();
  // The faces' outward side, as wound: against the cross product of their
  // edges (seen on desktop: tops point up, walls toward their front).
  constexpr float outward = -1.f;
  using V3 = std::array<float, 3>;
  int posOffset = -1, mtxOffset = -1, clrOffset = -1;
  for (u8 i = 0; i < layout.count; ++i) {
    if (layout.attrs[i].attr == GX_VA_POS)
      posOffset = layout.attrs[i].offset;
    else if (layout.attrs[i].attr == GX_VA_PNMTXIDX)
      mtxOffset = layout.attrs[i].offset;
    else if (layout.attrs[i].attr == GX_VA_CLR0)
      clrOffset = layout.attrs[i].offset;
  }
  const auto at = [&](u32 v, int offset) { return base + static_cast<size_t>(v) * layout.stride + offset; };
  for (u32 v = 0; v < vtxCount; ++v) {
    constexpr float none[4] = {0.f, 0.f, 0.f, 1.f};
    std::memcpy(at(v, nrmOffset), none, 16);
  }
  if (posOffset < 0 || vtxCount < 3)
    return;
  static std::vector<V3> pos, area, unit, ref, wound;
  static std::vector<float> weight;
  static std::vector<std::array<u16, 3>> tris;
  static std::vector<u32> group, first, faces;
  static std::vector<u64> softKey;
  static std::unordered_map<u64, u32> points;
  pos.resize(vtxCount);
  group.resize(vtxCount);
  softKey.resize(vtxCount);
  points.clear();
  const auto& cfg = sDrawCache.config;
  const u8 posType = config.attrs[GX_VA_POS].attrType;
  const u64 mesh = cfg.blendMode == GX_BM_BLEND && !xr_additive(cfg) && (posType == GX_INDEX8 || posType == GX_INDEX16)
                       ? array_content_hash(GX_VA_POS)
                       : 0;
  for (u32 v = 0; v < vtxCount; ++v) {
    std::memcpy(pos[v].data(), at(v, posOffset), 12);
    u32 key[4]{};
    std::memcpy(key, at(v, posOffset), 12);
    if (mtxOffset >= 0)
      std::memcpy(&key[3], at(v, mtxOffset), 4);
    group[v] = points.try_emplace(XXH3_64bits(key, sizeof(key)), static_cast<u32>(points.size())).first->second;
    softKey[v] = XXH3_64bits_withSeed(key, sizeof(key), mesh);
  }
  // The triangles, as prepare_idx_buffer makes them (its windings).
  tris.clear();
  const auto prim = static_cast<GXPrimitive>(sDrawPrim);
  if (prim == GX_QUADS) {
    for (u32 v = 0; v + 3 < vtxCount; v += 4) {
      tris.push_back({u16(v), u16(v + 1), u16(v + 2)});
      tris.push_back({u16(v + 2), u16(v + 3), u16(v)});
    }
  } else if (prim == GX_TRIANGLES) {
    for (u32 v = 0; v + 2 < vtxCount; v += 3)
      tris.push_back({u16(v), u16(v + 1), u16(v + 2)});
  } else if (prim == GX_TRIANGLEFAN) {
    for (u32 v = 2; v < vtxCount; ++v)
      tris.push_back({0, u16(v - 1), u16(v)});
  } else if (prim == GX_TRIANGLESTRIP) {
    for (u32 v = 2; v < vtxCount; ++v)
      tris.push_back((v & 1) == 0 ? std::array<u16, 3>{u16(v - 2), u16(v - 1), u16(v)}
                                   : std::array<u16, 3>{u16(v - 1), u16(v - 2), u16(v)});
  }
  const auto sub = [](const V3& a, const V3& b) { return V3{a[0] - b[0], a[1] - b[1], a[2] - b[2]}; };
  const auto dot = [](const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  const auto cross = [](const V3& a, const V3& b) {
    return V3{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
  };
  // Each face's normal (its length twice its area) and unit normal; faces by
  // point (CSR: first[g] .. first[g + 1]); each vertex's own faces, summed
  // turned to agree (ref) and as wound (wound).
  const size_t n = tris.size();
  area.resize(n);
  unit.resize(n);
  ref.assign(vtxCount, V3{});
  wound.assign(vtxCount, V3{});
  first.assign(points.size() + 1, 0);
  for (size_t t = 0; t < n; ++t) {
    const auto& [a, b, c] = tris[t];
    area[t] = cross(sub(pos[b], pos[a]), sub(pos[c], pos[a]));
    const float len = std::sqrt(dot(area[t], area[t]));
    if (!(len > 1e-20f) || group[a] == group[b] || group[b] == group[c] || group[a] == group[c]) {
      unit[t] = {};
      continue;
    }
    unit[t] = {area[t][0] / len, area[t][1] / len, area[t][2] / len};
    for (const u16 v : tris[t]) {
      ++first[group[v] + 1];
      const float s = dot(ref[v], unit[t]) < 0.f ? -1.f : 1.f;
      for (int k = 0; k < 3; ++k) {
        ref[v][k] += s * unit[t][k];
        wound[v][k] += area[t][k];
      }
    }
  }
  for (size_t g = 0; g < points.size(); ++g)
    first[g + 1] += first[g];
  faces.resize(first.back());
  {
    static std::vector<u32> fill;
    fill.assign(first.begin(), first.end() - 1);
    for (size_t t = 0; t < n; ++t)
      if (unit[t][0] != 0.f || unit[t][1] != 0.f || unit[t][2] != 0.f)
        for (const u16 v : tris[t])
          faces[fill[group[v]]++] = static_cast<u32>(t);
  }
  // The soft normals' faces, each once (by its points, in any order), while
  // the window is open.
  auto& fit = sRoomFit;
  if (mesh != 0 && !fit.frozen)
    for (size_t t = 0; t < n; ++t) {
      if (unit[t][0] == 0.f && unit[t][1] == 0.f && unit[t][2] == 0.f)
        continue;
      const auto& [a, b, c] = tris[t];
      std::array<u64, 3> k{softKey[a], softKey[b], softKey[c]};
      std::sort(k.begin(), k.end());
      if (!fit.softSeen.insert(XXH3_64bits(k.data(), sizeof(k))).second)
        continue;
      fit.soft.push_back({mesh, 0,
                          V3{(pos[a][0] + pos[b][0] + pos[c][0]) / 3.f, (pos[a][1] + pos[b][1] + pos[c][1]) / 3.f,
                             (pos[a][2] + pos[b][2] + pos[c][2]) / 3.f},
                          area[t]});
      fit.lastAdd = g_gxState.frameSerial;
    }
  const auto softRadius = mesh != 0 && fit.frozen ? fit.softRadius.find(mesh) : fit.softRadius.end();
  // The fit's part: a stage draw lit by its vertex colours (as the census
  // tells them), with the game's view to take normals to the world.
  bool ras = false;
  for (u32 i = 0; i < config.tevStageCount; ++i) {
    const auto& st = config.tevStages[i];
    for (const auto arg : {st.colorPass.a, st.colorPass.b, st.colorPass.c, st.colorPass.d})
      ras |= arg == GX_CC_RASC || arg == GX_CC_RASA;
  }
  const auto& cc = config.colorChannels[0];
  const float* view = gfx::xr_world_view();
  const bool baked = ras && clrOffset >= 0 && cc.matSrc == GX_SRC_VTX && !cc.lightingEnabled && view != nullptr &&
                     !xr_additive(cfg);
  const bool accumulate = baked && !fit.frozen;
  const bool delight = baked && fit.frozen && fit.valid;
  const float up = fit.c[0] + fit.c[2];
  // Model space to the world, per position matrix: its cofactors (normals
  // and areas), then the view's inverse rotation (its transpose).
  const auto to_world = [&](u32 v, const V3& nm) {
    u32 mtx = g_gxState.currentPnMtx;
    if (mtxOffset >= 0) {
      u32 raw;
      std::memcpy(&raw, at(v, mtxOffset), 4);
      mtx = raw / 3u;
    }
    const f32* m = reinterpret_cast<const f32*>(&g_gxState.pnMtx[std::min<u32>(mtx, MaxPnMtx - 1)].pos);
    const V3 r0{m[0], m[1], m[2]}, r1{m[4], m[5], m[6]}, r2{m[8], m[9], m[10]};
    const V3 cam{dot(cross(r1, r2), nm), dot(cross(r2, r0), nm), dot(cross(r0, r1), nm)};
    return V3{cam[0] * view[0] + cam[1] * view[4] + cam[2] * view[8],
              cam[0] * view[1] + cam[1] * view[5] + cam[2] * view[9],
              cam[0] * view[2] + cam[1] * view[6] + cam[2] * view[10]};
  };
  if (accumulate) {
    weight.assign(vtxCount, 0.f);
    for (size_t t = 0; t < n; ++t)
      if (unit[t][0] != 0.f || unit[t][1] != 0.f || unit[t][2] != 0.f) {
        const V3 w = to_world(tris[t][0], area[t]);
        const float a3 = std::sqrt(dot(w, w)) / 6.f;
        for (const u16 v : tris[t])
          weight[v] += a3;
      }
    ++fit.draws;
    fit.lastAdd = g_gxState.frameSerial;
  }
  for (u32 v = 0; v < vtxCount; ++v) {
    const float rl = std::sqrt(dot(ref[v], ref[v]));
    if (!(rl > 1e-6f))
      continue;
    const V3 r{ref[v][0] / rl, ref[v][1] / rl, ref[v][2] / rl};
    V3 sum{};
    if (softRadius != fit.softRadius.end()) {
      // The cells around it, each face as wound (the mesh's outward side, not
      // this vertex's own faces': a fold's are the wrong way round), by
      // (1 - (d / radius)^2)^2.
      const float rad = softRadius->second;
      for (s32 dz = -1; dz <= 1; ++dz)
        for (s32 dy = -1; dy <= 1; ++dy)
          for (s32 dx = -1; dx <= 1; ++dx) {
            const auto cell = fit.softCells.find(room_soft_cell(mesh, pos[v], rad, {dx, dy, dz}));
            if (cell == fit.softCells.end())
              continue;
            for (u32 i = cell->second.first; i < cell->second.second; ++i) {
              const auto& sf = fit.soft[i];
              const V3 d = sub(sf.c, pos[v]);
              const float q = 1.f - dot(d, d) / (rad * rad);
              if (q <= 0.f)
                continue;
              for (int k = 0; k < 3; ++k)
                sum[k] += q * q * sf.a[k];
            }
          }
    }
    const bool soft = dot(sum, sum) > 1e-30f;
    if (!soft)
      for (u32 i = first[group[v]]; i < first[group[v] + 1]; ++i) {
        const u32 t = faces[i];
        const float d = dot(unit[t], r);
        if (std::abs(d) < cosCrease)
          continue;
        const float s = d < 0.f ? -1.f : 1.f;
        for (int k = 0; k < 3; ++k)
          sum[k] += s * area[t][k];
      }
    const float sl = std::sqrt(dot(sum, sum));
    if (!(sl > 1e-30f))
      continue;
    const float s = (soft || dot(sum, wound[v]) >= 0.f ? 1.f : -1.f) * outward / sl;
    float out[4] = {sum[0] * s, sum[1] * s, sum[2] * s, 1.f};
    if (accumulate || delight) {
      V3 w = to_world(v, {out[0], out[1], out[2]});
      const float wl = std::sqrt(dot(w, w));
      if (wl > 1e-20f) {
        w = {w[0] / wl, w[1] / wl, w[2] / wl};
        if (accumulate && weight[v] > 0.f) {
          float clr[4];
          std::memcpy(clr, at(v, clrOffset), 16);
          const double lum = 0.2126 * clr[0] + 0.7152 * clr[1] + 0.0722 * clr[2];
          const double basis[4] = {1.0, w[0], w[1], w[2]}, wt = weight[v];
          for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j)
              fit.ata[i][j] += wt * basis[i] * basis[j];
            fit.atb[i] += wt * basis[i] * lum;
          }
          fit.btb += wt * lum * lum;
          ++fit.verts;
        }
        if (delight) {
          const float at_n = fit.c[0] + fit.c[1] * w[0] + fit.c[2] * w[1] + fit.c[3] * w[2];
          // Within a factor of 2: the plane fits a bake's falloff loosely
          // at its far side (Pokemon Stadium's undersides blew out at 4).
          out[3] = 1.f + (std::clamp(up / std::max(at_n, 1e-3f), 0.5f, 2.f) - 1.f) * fit.trust;
        }
      }
    }
    if (soft)
      out[3] = -out[3]; // the shader leaves it unturned
    std::memcpy(at(v, nrmOffset), out, 16);
  }
}

static void decode_vertices(const ShaderConfig& config, const DecodedLayout& layout, const u8* raw, u16 vtxCount,
                            u8* base) noexcept {
  enum : u8 { kU32, kColor, kComps };
  struct Plan {
    const AttrConfig* m;
    GXAttr attr;
    u8 slice, kind, compType;
    float scale;
    u32 compSize, n, words, offset;
  };
  std::array<Plan, std::size(DecodedLayout{}.attrs)> plans;
  u8 planCount = 0;
  int smoothOffset = -1;
  for (u8 i = 0; i < layout.count; ++i) {
    const auto& d = layout.attrs[i];
    if (d.attr == kDecodedXrNrm) {
      smoothOffset = d.offset;
      continue;
    }
    const auto attr = static_cast<GXAttr>(d.attr);
    const auto& m = config.attrs[attr];
    const u8 kind = d.format == DecodedFormat::U32               ? kU32
                    : attr == GX_VA_CLR0 || attr == GX_VA_CLR1 ? kColor
                                                               : kComps;
    const u32 comps = d.format == DecodedFormat::F32x2 ? 2 : d.format == DecodedFormat::F32x3 ? 3 : 4;
    plans[planCount++] = {&m,
                attr,
                d.slice,
                kind,
                m.compType,
                1.f / static_cast<float>(1u << m.frac),
                comp_type_size(attr, static_cast<GXCompType>(m.compType)),
                std::min<u32>(m.cnt, 3),
                kind == kU32 ? 1u : comps,
                d.offset};
  }
  for (u32 v = 0; v < vtxCount; ++v) {
    const u8* vtx = raw + v * config.vtxStride;
    u8* dst = base + static_cast<size_t>(v) * layout.stride;
    for (u8 i = 0; i < planCount; ++i) {
      const auto& p = plans[i];
      bool le = false;
      const u8* src = decoded_source(*p.m, p.attr, p.slice, vtx, le, p.compSize);
      // 4-byte words at 4-byte offsets in a 4-aligned range, stored one by
      // one: a memcpy of a size picked at run time is a library call.
      u32* out = reinterpret_cast<u32*>(dst + p.offset);
      if (src == nullptr) {
        // Zeros, as the shader's bounds-checked loads return.
        for (u32 w = 0; w < p.words; ++w)
          out[w] = 0;
        continue;
      }
      if (p.kind == kU32) {
        out[0] = src[0];
        continue;
      }
      float f[4] = {0.f, 0.f, 0.f, 0.f};
      if (p.kind == kColor) {
        decode_color(src, p.compType, le, f);
      } else {
        for (u32 c = 0; c < p.n; ++c) {
          f[c] = decode_component(src + c * p.compSize, p.compType, p.scale, le);
        }
      }
      // Straight-line stores: a loop here is turned back into a memcpy call.
      out[0] = std::bit_cast<u32>(f[0]);
      out[1] = std::bit_cast<u32>(f[1]);
      if (p.words > 2)
        out[2] = std::bit_cast<u32>(f[2]);
      if (p.words > 3)
        out[3] = std::bit_cast<u32>(f[3]);
    }
  }
  if (smoothOffset >= 0)
    smooth_normals(config, layout, vtxCount, base, static_cast<u8>(smoothOffset));
}

// Decoded vertices kept across frames (AURORA_VTX_CACHE, default on with the
// decode; 0 turns it off). Most of a frame's draws decode the same vertices
// as last frame (the stage, the fighters' meshes: skinning is the GPU's), and
// decoding them was a third of the FIFO thread on the Quest (four-CPU Mute
// City: about 3 of 9.8 ms a frame). The key is the draw's own vertex stream
// (indices, or the data itself), its attribute formats, and the contents of
// every array it indexes, hashed once a frame (AttrArray::contentHash): an
// array the game rewrites gets a new key, so nothing goes stale. A hit is a
// copy into the frame's storage pool. AURORA_VTX_CACHE_CHECK=1 decodes every
// hit anyway and counts mismatches; AURORA_VTX_CACHE_STATS=1 logs hits.
namespace {
struct DecodeKey {
  u64 lo, hi;
  bool operator==(const DecodeKey&) const = default;
};
struct DecodeKeyHash {
  size_t operator()(const DecodeKey& k) const noexcept { return static_cast<size_t>(k.lo); }
};
struct ClipBox {
  u8 mtx; // position matrix (GX_DIRECT matrix indices; else the current one)
  float lo[3], hi[3];
};
struct DecodeEntry {
  DecodeKey key{};
  std::vector<u8> bytes;
  u32 lastFrame = 0;
  std::array<u32, 2> ordinal{}; // its position in DecodeCache::order, by frame parity
  // The draw's model-space bounds per position matrix, for classify_xr_clip.
  std::vector<ClipBox> boxes;
  bool haveBoxes = false;
};
struct DecodeCache {
  std::unordered_map<DecodeKey, DecodeEntry, DecodeKeyHash> entries;
  // Draws come in the same order frame after frame: last frame's entry at a
  // draw's position is tried before the table, whose lookups (a cache miss
  // or two each, thousands a frame) were half the FIFO thread's cost of a
  // hit. Cleared when entries are erased (they point into the table).
  std::vector<DecodeEntry*> order, lastOrder;
  size_t cursor = 0; // where in lastOrder the next draw is expected
  u32 orderFrame = 0;
  u64 predicted = 0, looked = 0; // logged with the hits
  size_t bytes = 0;
  u32 sweptFrame = 0;
  u64 hits = 0, misses = 0, hitBytes = 0, mismatches = 0, unstored = 0;
  u32 loggedFrame = 0;
  // Keys missed once lately (their low halves): a second miss stores one.
  // Two generations, the older dropped every kDecodeGhostFrames; exact
  // sets, open addressed (by low bits, two keys sharing a slot each evicted
  // the other every frame, and neither was ever stored).
  struct Ghosts {
    std::vector<u64> slots = std::vector<u64>(1u << 15);
    size_t count = 0;
    bool insert(u64 lo) noexcept { // false when already in
      lo |= 1; // 0 is an empty slot
      const size_t mask = slots.size() - 1;
      for (size_t i = (lo >> 17) & mask;; i = (i + 1) & mask) {
        if (slots[i] == lo)
          return false;
        if (slots[i] == 0) {
          slots[i] = lo;
          ++count;
          return true;
        }
      }
    }
    bool contains(u64 lo) const noexcept {
      lo |= 1;
      const size_t mask = slots.size() - 1;
      for (size_t i = (lo >> 17) & mask; slots[i] != 0; i = (i + 1) & mask)
        if (slots[i] == lo)
          return true;
      return false;
    }
    void clear() noexcept {
      std::fill(slots.begin(), slots.end(), 0);
      count = 0;
    }
  };
  std::array<Ghosts, 2> ghosts{};
  u32 ghostFlip = 0;
};
// A key missed again within this many to twice this many frames is stored.
constexpr u32 kDecodeGhostFrames = 60;
DecodeCache sDecodeCache;
// The entry push_decoded_vertices last used, and the draw it was for: the
// clip classification of the same draw follows it (push_gx_draw).
struct LastDecode {
  const u8* raw = nullptr;
  u16 vtxCount = 0;
  u32 frame = 0;
  DecodeEntry* entry = nullptr;
} sLastDecode;
// The key last worked out, and for which draw: the merge check classifies a
// draw before push_decoded_vertices decodes it, and both need its key.
struct LastKey {
  const u8* raw = nullptr;
  u16 vtxCount = 0;
  u32 frame = 0;
  DecodeKey key{};
} sLastKey;
#ifdef __ANDROID__
constexpr size_t kDecodeCacheBytes = 24u << 20;
#else
constexpr size_t kDecodeCacheBytes = 64u << 20;
#endif

bool decode_cache_on() noexcept {
  static const bool on = [] {
    const char* v = std::getenv("AURORA_VTX_CACHE");
    return v == nullptr || *v != '0';
  }();
  return on;
}

u64 array_content_hash(GXAttr attr) noexcept {
  auto& array = g_gxState.arrays[attr];
  if (!array.hashed) {
    array.contentHash = array.data != nullptr ? XXH3_64bits(array.data, array.size) : 0;
    array.hashed = true;
  }
  return array.contentHash;
}

DecodeKey decode_key(const ShaderConfig& config, const u8* raw, u16 vtxCount) noexcept {
  struct Parts {
    u64 raw = 0, attrs = 0;
    std::array<u64, GX_VA_TEX7 - GX_VA_POS + 1> arrays{};
    std::array<u32, GX_VA_TEX7 - GX_VA_POS + 1> sizes{};
    u32 vtxCount = 0;
    u8 vtxStride = 0, decodedAll = 0, decodedPos = 0, pad = 0;
    u32 fit = 0;
  } parts;
  parts.raw = XXH3_64bits(raw, static_cast<size_t>(vtxCount) * config.vtxStride);
  // The formats change only with the vertex format: hashed when they do.
  static decltype(config.attrs) lastAttrs{};
  static u64 lastAttrsHash = XXH3_64bits(lastAttrs.data(), sizeof(lastAttrs));
  if (std::memcmp(lastAttrs.data(), config.attrs.data(), sizeof(lastAttrs)) != 0) {
    lastAttrs = config.attrs;
    lastAttrsHash = XXH3_64bits(lastAttrs.data(), sizeof(lastAttrs));
  }
  parts.attrs = lastAttrsHash;
  for (int a = GX_VA_POS; a <= GX_VA_TEX7; ++a) {
    const u8 type = config.attrs[a].attrType;
    if (type != GX_INDEX8 && type != GX_INDEX16)
      continue;
    parts.arrays[a - GX_VA_POS] = array_content_hash(static_cast<GXAttr>(a));
    parts.sizes[a - GX_VA_POS] = g_gxState.arrays[a].size;
  }
  parts.vtxCount = vtxCount;
  parts.vtxStride = config.vtxStride;
  parts.decodedAll = config.decodedAll;
  parts.decodedPos = config.decodedPos;
  parts.pad = config.xrSmoothNrm ? sDrawPrim : 0; // the smooth normals' triangles
  parts.fit = config.xrSmoothNrm ? sRoomFit.gen : 0; // and de-light factors
  const XXH128_hash_t h = XXH3_128bits(&parts, sizeof(parts));
  return {h.low64, h.high64};
}

// Every few seconds, and at once past the budget: entries unused for two
// seconds go (a table of stale entries makes every lookup slower), then, if
// still over the budget, everything not used in the last two frames.
void sweep_decode_cache() noexcept {
  auto& c = sDecodeCache;
  const u32 now = g_gxState.frameSerial;
  if (c.sweptFrame == now || (c.bytes <= kDecodeCacheBytes && now - c.sweptFrame < 300))
    return;
  c.sweptFrame = now;
  c.order.clear();
  c.lastOrder.clear();
  sLastDecode = {};
  // Entries used in the last two frames stay whatever the budget: the
  // frame's own draws (none used yet as it sweeps) and the last one's.
  for (u32 age : {120u, 3u}) {
    for (auto it = c.entries.begin(); it != c.entries.end();) {
      if (now - it->second.lastFrame >= age) {
        c.bytes -= it->second.bytes.size();
        it = c.entries.erase(it);
      } else {
        ++it;
      }
    }
    if (c.bytes <= kDecodeCacheBytes * 3 / 4)
      break;
  }
}
} // namespace

// The decode cache's misses so far (the FIFO probe's count).
u64 decode_misses() noexcept { return sDecodeCache.misses; }

// `raw`'s key, reusing the one the merge check just worked out. The decode
// takes it (`take`): it's for that draw only, and a later draw's data could
// land at the same address.
static DecodeKey decode_key_for(const ShaderConfig& config, const u8* raw, u16 vtxCount, bool take) noexcept {
  auto& k = sLastKey;
  if (k.raw == raw && k.vtxCount == vtxCount && k.frame == g_gxState.frameSerial) {
    const DecodeKey key = k.key;
    if (take)
      k = {};
    return key;
  }
  const DecodeKey key = decode_key(config, raw, vtxCount);
  k = take ? LastKey{} : LastKey{raw, vtxCount, g_gxState.frameSerial, key};
  return key;
}

static gfx::Range push_decoded_vertices(const ShaderConfig& config, const u8* raw, u16 vtxCount,
                                        size_t alignment) noexcept {
  const DecodedLayout layout = decode_layout_for(config);
  const size_t size = static_cast<size_t>(vtxCount) * layout.stride;
  u8* base = nullptr;
  if (!decode_cache_on()) {
    const auto range = gfx::map_storage_aligned(size, alignment, base);
    if (base != nullptr)
      decode_vertices(config, layout, raw, vtxCount, base);
    return range;
  }
  auto& c = sDecodeCache;
  const DecodeKey key = decode_key_for(config, raw, vtxCount, true);
  const auto range = gfx::map_storage_aligned(size, alignment, base);
  if (base == nullptr)
    return range;
  const u32 now = g_gxState.frameSerial;
  if (c.orderFrame != now) {
    sweep_decode_cache();
    c.orderFrame = now;
    c.lastOrder.swap(c.order);
    c.order.clear();
    c.cursor = 0;
  }
  // Last frame's next draw after the previous match; on a miss the table,
  // and the prediction picks up after where that entry stood last frame, so
  // one draw more or less doesn't throw off every draw after it.
  DecodeEntry* entry = nullptr;
  if (c.cursor < c.lastOrder.size() && c.lastOrder[c.cursor] != nullptr && c.lastOrder[c.cursor]->key == key) {
    entry = c.lastOrder[c.cursor++];
    ++c.predicted;
  } else if (auto it = c.entries.find(key); it != c.entries.end()) {
    entry = &it->second;
    ++c.looked;
    const u32 was = entry->ordinal[(now - 1) & 1];
    if (entry->lastFrame + 1 == now && was < c.lastOrder.size() && c.lastOrder[was] == entry)
      c.cursor = was + 1;
  }
  if (entry != nullptr && entry->bytes.size() == size) {
    sLastDecode = {raw, vtxCount, now, entry};
    entry->ordinal[now & 1] = static_cast<u32>(c.order.size());
    c.order.push_back(entry);
    std::memcpy(base, entry->bytes.data(), size);
    entry->lastFrame = now;
    ++c.hits;
    c.hitBytes += size;
    static const bool check = [] {
      const char* v = std::getenv("AURORA_VTX_CACHE_CHECK");
      return v != nullptr && *v == '1';
    }();
    if (check) {
      static std::vector<u8> fresh;
      fresh.resize(size);
      decode_vertices(config, layout, raw, vtxCount, fresh.data());
      if (std::memcmp(fresh.data(), base, size) != 0 && ++c.mismatches <= 8)
        Log.warn("Decoded-vertex cache: stale entry ({} vertices, stride {})", vtxCount, layout.stride);
    }
  } else {
    decode_vertices(config, layout, raw, vtxCount, base);
    ++c.misses;
    // Stored on the key's second miss within kDecodeGhostFrames only. A
    // draw whose vertices change every frame (some 190 a frame on Yoshi's
    // Story) left an entry each frame, never used again, until a sweep over
    // the budget emptied the whole cache and the frame after decoded every
    // draw again: a 23 ms FIFO frame every 2.5 s on the Quest, its world
    // recorded too late for SpaceWarp (2026-10-25).
    if (now - c.ghostFlip >= kDecodeGhostFrames || c.ghosts[0].count > c.ghosts[0].slots.size() / 2) {
      c.ghostFlip = now;
      std::swap(c.ghosts[0], c.ghosts[1]);
      c.ghosts[0].clear();
    }
    if (!c.ghosts[1].contains(key.lo) && c.ghosts[0].insert(key.lo)) {
      ++c.unstored;
      c.order.push_back(nullptr); // keeps the next draws' predicted places
      sLastDecode = {raw, vtxCount, now, nullptr};
      return range;
    }
    auto& e = c.entries[key];
    c.bytes += size - e.bytes.size();
    e.key = key;
    e.bytes.assign(base, base + size);
    e.lastFrame = now;
    e.ordinal[now & 1] = static_cast<u32>(c.order.size());
    c.order.push_back(&e);
    sLastDecode = {raw, vtxCount, now, &e};
  }
  static const bool stats = [] {
    const char* v = std::getenv("AURORA_VTX_CACHE_STATS");
    return v != nullptr && *v == '1';
  }();
  if (stats && now - c.loggedFrame >= 600) {
    c.loggedFrame = now;
    Log.info("Decoded-vertex cache: {} hits ({:.1f} MiB; {} predicted, {} looked up), {} misses ({} not stored: "
             "first seen), {} mismatches; {} entries, {:.1f} MiB",
             c.hits, static_cast<double>(c.hitBytes) / (1 << 20), c.predicted, c.looked, c.misses, c.unstored,
             c.mismatches, c.entries.size(), static_cast<double>(c.bytes) / (1 << 20));
    c.hits = c.misses = c.hitBytes = c.predicted = c.looked = c.unstored = 0;
  }
  return range;
}

#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
enum class XrClipClass : u8 { Inside, Straddle, Outside };
// Margins (game units) cover float differences from the GPU's transform.
constexpr float kXrClipEps = 1e-2f;

// Where a clipped world draw lies against the current clip, from its
// vertices' distances to each plane (linear across a triangle, so the
// vertices bound every fragment). Inside: past every fade band, drawn
// solid without discard (keeps early depth). Outside: wholly cut by one
// plane, dropped from the eyes. Straddle: needs the dithering pipeline. A
// draw dissolving as a whole (opacity under 1) is never Inside.
// `outside`, when given (vtxCount entries): each vertex's planes it is past
// (bit per plane), for culling a straddling draw's triangles; set only when
// the vertices were read (`*masked`).
static XrClipClass classify_xr_clip(const ShaderConfig& config, const u8* raw, u16 vtxCount, u8* outside = nullptr,
                                    bool* masked = nullptr, u8* unsolid = nullptr) noexcept {
  static const bool enabled = [] {
    const char* v = std::getenv("AURORA_XR_CLIP_CLASSIFY");
    return v == nullptr || *v != '0';
  }();
  if (masked != nullptr)
    *masked = false;
  const auto* clip = gfx::xr_clip_camera();
  if (!enabled || clip == nullptr || config.lineMode != 0) {
    return XrClipClass::Straddle;
  }
  const bool dissolving = clip->opacity < 1.f;
  const auto& pos = config.attrs[GX_VA_POS];
  const auto& idx = config.attrs[GX_VA_PNMTXIDX];
  if (pos.attrType == GX_NONE || (idx.attrType != GX_NONE && idx.attrType != GX_DIRECT)) {
    return XrClipClass::Straddle;
  }
  int planes[gfx::XrMaxClipPlanes];
  int planeCount = 0;
  for (int k = 0; k < gfx::XrMaxClipPlanes; ++k) {
    const auto& pl = clip->planes[k];
    if (pl[0] != 0.f || pl[1] != 0.f || pl[2] != 0.f || pl[3] < 0.f) {
      planes[planeCount++] = k;
    }
  }
  if (planeCount == 0) {
    return dissolving ? XrClipClass::Straddle : XrClipClass::Inside;
  }
  // A plane in matrix `mtx`'s model space.
  const auto to_model = [&](u32 mtx, int n, float* q) {
    const f32* m = reinterpret_cast<const f32*>(&g_gxState.pnMtx[mtx].pos);
    const auto& pl = clip->planes[planes[n]];
    for (int j = 0; j < 4; ++j)
      q[j] = pl[0] * m[j] + pl[1] * m[4 + j] + pl[2] * m[8 + j];
    q[3] += pl[3];
  };
  // The draw's bounds, kept with its decoded vertices: a box wholly inside
  // every plane's fade, or wholly past one plane, settles it without reading
  // a vertex. Reading them all every frame was 15% of the FIFO thread (Mute
  // City, four CPUs).
  DecodeEntry* entry = nullptr;
  if (decode_cache_on() && config.decodedAll) {
    // Used once: a later draw's data could land at the same address.
    if (sLastDecode.raw == raw && sLastDecode.vtxCount == vtxCount && sLastDecode.frame == g_gxState.frameSerial) {
      entry = sLastDecode.entry;
      sLastDecode = {};
    } else {
      // The merge check, before the draw is decoded: the draw the cache
      // expects next, else the table.
      const DecodeKey key = decode_key_for(config, raw, vtxCount, false);
      auto& c = sDecodeCache;
      if (c.orderFrame == g_gxState.frameSerial && c.cursor < c.lastOrder.size() && c.lastOrder[c.cursor] != nullptr &&
          c.lastOrder[c.cursor]->key == key)
        entry = c.lastOrder[c.cursor];
      else if (auto it = c.entries.find(key); it != c.entries.end())
        entry = &it->second;
    }
  }
  static const bool checkBoxes = [] {
    const char* v = std::getenv("AURORA_VTX_CACHE_CHECK");
    return v != nullptr && *v == '1';
  }();
  auto boxResult = XrClipClass::Straddle;
  if (entry != nullptr && entry->haveBoxes) {
    float blo[gfx::XrMaxClipPlanes], bhi[gfx::XrMaxClipPlanes];
    std::fill_n(blo, gfx::XrMaxClipPlanes, INFINITY);
    std::fill_n(bhi, gfx::XrMaxClipPlanes, -INFINITY);
    for (const auto& b : entry->boxes) {
      const u32 mtx = idx.attrType == GX_DIRECT ? b.mtx : std::min<u32>(g_gxState.currentPnMtx, MaxPnMtx - 1);
      for (int n = 0; n < planeCount; ++n) {
        float q[4];
        to_model(mtx, n, q);
        float dlo = q[3], dhi = q[3];
        for (int j = 0; j < 3; ++j) {
          const float a = q[j] * b.lo[j], c = q[j] * b.hi[j];
          dlo += std::min(a, c);
          dhi += std::max(a, c);
        }
        blo[n] = std::min(blo[n], dlo);
        bhi[n] = std::max(bhi[n], dhi);
      }
    }
    bool inside = !dissolving, out = false;
    for (int n = 0; n < planeCount; ++n) {
      out |= bhi[n] < -kXrClipEps;
      inside &= blo[n] > clip->fades[planes[n]] + kXrClipEps;
    }
    boxResult = out ? XrClipClass::Outside : inside ? XrClipClass::Inside : XrClipClass::Straddle;
    if (boxResult != XrClipClass::Straddle && !checkBoxes)
      return boxResult;
    // Straddling, or a loose box: the vertices decide.
  }
  ClipBox boxes[MaxPnMtx];
  u8 boxOf[MaxPnMtx];
  u32 boxCount = 0;
  // Each plane in a position matrix's model space, made when first used.
  float model[MaxPnMtx][gfx::XrMaxClipPlanes][4];
  u16 made = 0;
  float lo[gfx::XrMaxClipPlanes], hi[gfx::XrMaxClipPlanes];
  std::fill_n(lo, gfx::XrMaxClipPlanes, INFINITY);
  std::fill_n(hi, gfx::XrMaxClipPlanes, -INFINITY);
  const u32 compSize = comp_type_size(GX_VA_POS, static_cast<GXCompType>(pos.compType));
  const u32 comps = std::min<u32>(pos.cnt, 3);
  for (u32 v = 0; v < vtxCount; ++v) {
    const u8* vtx = raw + v * config.vtxStride;
    const u32 mtx = idx.attrType == GX_DIRECT ? std::min<u32>(vtx[idx.offset] / 3u, MaxPnMtx - 1)
                                              : std::min<u32>(g_gxState.currentPnMtx, MaxPnMtx - 1);
    if ((made & (1u << mtx)) == 0) {
      made |= 1u << mtx;
      for (int n = 0; n < planeCount; ++n)
        to_model(mtx, n, model[mtx][n]);
      boxOf[mtx] = static_cast<u8>(boxCount);
      boxes[boxCount++] = {static_cast<u8>(mtx), {INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}};
    }
    bool le = false;
    const u8* src = decoded_source(pos, GX_VA_POS, 0, vtx, le);
    if (src == nullptr) {
      return XrClipClass::Straddle;
    }
    float p[3] = {0.f, 0.f, 0.f};
    for (u32 c = 0; c < comps; ++c)
      p[c] = decode_component(src + c * compSize, pos.compType, pos.frac, le);
    auto& box = boxes[boxOf[mtx]];
    for (int j = 0; j < 3; ++j) {
      box.lo[j] = std::min(box.lo[j], p[j]);
      box.hi[j] = std::max(box.hi[j], p[j]);
    }
    u8 past = 0, soft = 0;
    for (int n = 0; n < planeCount; ++n) {
      const float* q = model[mtx][n];
      const float d = q[0] * p[0] + q[1] * p[1] + q[2] * p[2] + q[3];
      lo[n] = std::min(lo[n], d);
      hi[n] = std::max(hi[n], d);
      past |= d < -kXrClipEps ? u8(1u << n) : u8(0);
      soft |= d <= clip->fades[planes[n]] + kXrClipEps ? u8(1u << n) : u8(0);
    }
    if (outside != nullptr)
      outside[v] = past;
    if (unsolid != nullptr)
      unsolid[v] = soft;
  }
  if (masked != nullptr)
    *masked = outside != nullptr;
  if (entry != nullptr && !entry->haveBoxes) {
    entry->boxes.assign(boxes, boxes + boxCount);
    entry->haveBoxes = true;
  }
  bool inside = !dissolving;
  auto result = XrClipClass::Straddle;
  for (int n = 0; n < planeCount; ++n) {
    if (hi[n] < -kXrClipEps) {
      result = XrClipClass::Outside;
      break;
    }
    inside &= lo[n] > clip->fades[planes[n]] + kXrClipEps;
  }
  if (result != XrClipClass::Outside && inside)
    result = XrClipClass::Inside;
  if (checkBoxes && boxResult != XrClipClass::Straddle && boxResult != result) {
    static u32 warned = 0;
    if (++warned <= 8)
      Log.warn("Clip bounds classed a draw {} but its vertices say {}", static_cast<int>(boxResult),
               static_cast<int>(result));
  }
  return result;
}

// XR room lighting's class for the current draw, from the game's link
// markers ("L<n>"): 1 fighters and their items (GX links 5 and 6), 2 the
// stage (links 0 to 4), 0 anything else (effects, the HUD, unmarked draws).
// 0 too while the room light is off.
// A texture's content key for the room light's per-stage tables: its base
// level (and palette), hashed once per texture object and version.
static u64 xr_texture_key(const GXTexObj_& obj) noexcept {
  struct Memo {
    const void* data;
    u32 version;
    const void* tlut;
    u64 key;
  };
  static std::unordered_map<const void*, Memo> memo;
  const bool palette = is_palette_format(obj.format()) && obj.tlut < g_gxState.loadedTluts.size();
  const GXTlutObj_* tlut = palette ? &g_gxState.loadedTluts[obj.tlut] : nullptr;
  const void* tlutData = tlut != nullptr ? tlut->data : nullptr;
  auto& m = memo[obj.data];
  if (m.data == obj.data && m.version == obj.texDataVersion && m.tlut == tlutData)
    return m.key;
  const u64 seed = tlutData != nullptr ? XXH3_64bits(tlutData, static_cast<size_t>(tlut->numEntries) * 2) : 0;
  const u32 bytes = GXGetTexBufferSize(obj.width(), obj.height(), obj.format(), false, 0);
  m = {obj.data, obj.texDataVersion, tlutData, obj.data != nullptr ? XXH3_64bits_withSeed(obj.data, bytes, seed) : 0};
  return m.key;
}

// A texture's base level as RGBA8 (empty: a format it can't convert).
static gfx::ConvertedTexture xr_texture_rgba(const GXTexObj_& obj) noexcept {
  gfx::ConvertedTexture px{};
  const ArrayRef data{static_cast<const uint8_t*>(obj.data), UINT32_MAX};
  if (is_palette_format(obj.format())) {
    const auto& t = g_gxState.loadedTluts[obj.tlut];
    if (t.data != nullptr)
      px = gfx::convert_texture_palette(obj.format(), obj.width(), obj.height(), 1, data, t.format, t.numEntries,
                                        {static_cast<const uint8_t*>(t.data), static_cast<size_t>(t.numEntries) * 2});
  } else {
    px = gfx::convert_texture(obj.format(), obj.width(), obj.height(), 1, data);
  }
  if (px.format != wgpu::TextureFormat::RGBA8Unorm || px.data.size() < size_t(obj.width()) * obj.height() * 4)
    px.data = {};
  return px;
}

// AURORA_XR_ROOM_TEXDUMP=dir (for AURORA_XR_ROOM_EMISSIVE's entries): each
// texture a stage, fighter or item draw samples with the room light on,
// once, as dir/<key>.ppm (over grey by alpha), with a log line (its link,
// blend mode, and whether it's an EFB copy).
static void xr_room_texdump() noexcept {
  static const char* dir = std::getenv("AURORA_XR_ROOM_TEXDUMP");
  if (dir == nullptr)
    return;
  static absl::flat_hash_set<u64> seen;
  const auto& sc = sDrawCache.config.shaderConfig;
  for (u32 i = 0; i < sc.tevStageCount; ++i) {
    const auto map = sc.tevStages[i].texMapId;
    if (map == GX_TEXMAP_NULL || map >= MaxTextures)
      continue;
    const auto& obj = g_gxState.loadedTextures[map];
    if (obj.data == nullptr)
      continue;
    const bool copy = g_gxState.copyTextures.contains(obj.data);
    const u64 key = copy ? reinterpret_cast<uintptr_t>(obj.data) : xr_texture_key(obj);
    if (!seen.insert(key).second)
      continue;
    Log.info("XR room texture {:016x}: {}x{} format {}, link {}, blend {}, copy {}", key, obj.width(), obj.height(), obj.format(),
             sLinkTag - 1, (int)sDrawCache.config.blendMode, copy);
    if (copy)
      continue;
    const auto px = xr_texture_rgba(obj);
    if (px.data.empty())
      continue;
    if (FILE* f = std::fopen(fmt::format("{}/{:016x}.ppm", dir, key).c_str(), "wb")) {
      std::fprintf(f, "P6\n%u %u\n255\n", obj.width(), obj.height());
      for (size_t p = 0; p < size_t(obj.width()) * obj.height(); ++p) {
        const u8* c = px.data.data() + p * 4;
        for (int k = 0; k < 3; ++k)
          std::fputc((c[k] * c[3] + 128 * (255 - c[3])) / 255, f);
      }
      std::fclose(f);
    }
  }
}

// XR room lighting's emissive stage surfaces: drawn as the game draws them,
// the room's light left out (lava, screens, glowing parts: a dark room
// darkened them), by the textures they sample (xr_texture_key). An entry
// with an emitter also lights the stage around it (lib/xr's room_ebox): its
// draws' box in the game's world each frame, the texture's mean colour by
// `gain`, reaching `reach` game units out from the box. Items too (link 6:
// Corneria's Arwing and Great Fox lasers are items). A colour EFB copy a
// stage draw samples is a screen (Pokemon Stadium's, showing the fighters:
// RGB565). Not the intensity ones: those are the fighters' shadows
// (HSD_Shadow, 256x256 I4), cast on the stage; taken for screens, every
// surface a shadow could fall on went Vanilla (Corneria's hull, Yoshi's
// Story, Onett, Green Greens) and its Vanilla eye pipelines compiled
// mid-match.
// AURORA_XR_ROOM_EMISSIVE="key[:emitter:gain:reach],..." adds entries (hex
// keys, as AURORA_XR_ROOM_TEXDUMP names them; emitter -1 for none).
struct XrEmissive {
  u64 tex;
  s8 emitter;
  float gain, reach;
};
constexpr XrEmissive kXrEmissive[] = {
    // Battlefield: the orb under the stage (lighting its underside), the
    // platforms' rainbow edges and the trims.
    {0x3a28b2f5a4c58102, 1, 1.5f, 45.f},
    {0x19068b36199d0936, -1, 0.f, 0.f},
    {0x4650f649e5ba6c50, -1, 0.f, 0.f},
    {0xed9d81f37efc710f, -1, 0.f, 0.f},
    {0x8d1b9372a3533027, -1, 0.f, 0.f},
    // Brinstar: the acid (lighting the stage from below as it rises), the
    // light strips and panels.
    {0x9a8c1205e32400b5, 0, 3.f, 80.f},
    {0x55a7d331ff91e72b, -1, 0.f, 0.f},
    {0xc63a8679b17881b8, -1, 0.f, 0.f},
    {0xe208076c16b57ff6, -1, 0.f, 0.f},
    {0x7d5f15aa3ba27a19, -1, 0.f, 0.f},
    // Corneria: the glow sprites (engines, lights; blended). Its lasers are
    // items, marked by kind (melee-pc's pc_xr_glow_begin). Not the glows its
    // opaque draws sample (3dbb5b15..., b0eeafe5...): the Great Fox's hull
    // takes them as reflections, and as entries the hull went Vanilla.
    {0x15c2c6042abe7d09, -1, 0.f, 0.f},
    {0x4380472ef3ff055b, -1, 0.f, 0.f},
    // Mute City: the GO signs, the light bars, and the underpasses' neon
    // tubes (pink and blue; animated, a texture each frame).
    {0x048f9b8b3cc10ca6, -1, 0.f, 0.f},
    {0x09452562b35cf5d9, -1, 0.f, 0.f},
    {0x0b934455646ea995, -1, 0.f, 0.f},
    {0x29b8ced6c6ce79a3, -1, 0.f, 0.f},
    {0x496f789610bb0d5e, -1, 0.f, 0.f},
    {0xb99ff32a19a68dcf, -1, 0.f, 0.f},
    {0x091409ee73dc5837, -1, 0.f, 0.f},
    {0x270bc2afcc1198ec, -1, 0.f, 0.f},
    {0x34fda0e7e4ba1b50, -1, 0.f, 0.f},
    {0x433d2bdd220258de, -1, 0.f, 0.f},
    {0x4f0b3b99376771c8, -1, 0.f, 0.f},
    {0x68bef71433ba3878, -1, 0.f, 0.f},
    {0x71300605363173d1, -1, 0.f, 0.f},
    {0x82ff4643694cab76, -1, 0.f, 0.f},
    {0xab08bed5cc0fe1fd, -1, 0.f, 0.f},
    {0xab8879d69e3a8b60, -1, 0.f, 0.f},
    {0xcb3b1dfa105854ec, -1, 0.f, 0.f},
    {0x3d593f121c03673d, -1, 0.f, 0.f},
    {0x439c83afe619add1, -1, 0.f, 0.f},
    {0x805a8c3296f16c54, -1, 0.f, 0.f},
    {0xc9bb812e002a7193, -1, 0.f, 0.f},
    {0xf2fced2569cc9a56, -1, 0.f, 0.f},
    {0x04b48943d6b25c1b, -1, 0.f, 0.f},
    {0x0cf319c73f4e9a90, -1, 0.f, 0.f},
    {0x2645b1eba3b50d99, -1, 0.f, 0.f},
    {0x69b143be91dec939, -1, 0.f, 0.f},
    {0xe7a54ce2df2325bd, -1, 0.f, 0.f},
};

static const XrEmissive* xr_emissive_entry(u64 key) noexcept {
  static const auto table = [] {
    absl::flat_hash_map<u64, XrEmissive> t;
    for (const auto& e : kXrEmissive)
      if (e.tex != 0)
        t[e.tex] = e;
    if (const char* v = std::getenv("AURORA_XR_ROOM_EMISSIVE")) {
      for (const char* p = v; *p != '\0';) {
        char* end = nullptr;
        XrEmissive e{std::strtoull(p, &end, 16), -1, 1.f, 0.f};
        if (end == p)
          break;
        p = end;
        if (*p == ':')
          e.emitter = static_cast<s8>(std::strtol(p + 1, &end, 10)), p = end;
        if (*p == ':')
          e.gain = std::strtof(p + 1, &end), p = end;
        if (*p == ':')
          e.reach = std::strtof(p + 1, &end), p = end;
        t[e.tex] = e;
        Log.info("XR room light: emissive texture {:016x}, emitter {}, gain {:.2f}, reach {:.0f}", e.tex, e.emitter,
                 e.gain, e.reach);
        while (*p == ',' || *p == ' ')
          ++p;
      }
    }
    return t;
  }();
  const auto it = table.find(key);
  return it != table.end() ? &it->second : nullptr;
}

// The current draw's emissive entry (nullptr: none), and the texture it
// matched by. A draw asks several times (its class decides its pipelines):
// the last answer stands while the draw's textures and link are the same.
static const XrEmissive* xr_room_emissive(const GXTexObj_** texOut = nullptr) noexcept {
  static constexpr XrEmissive screen{0, -1, 0.f, 0.f};
  const auto& sc = sDrawCache.config.shaderConfig;
  u64 sig = sLinkTag;
  for (u32 i = 0; i < sc.tevStageCount; ++i) {
    const auto map = sc.tevStages[i].texMapId;
    if (map != GX_TEXMAP_NULL && map < MaxTextures) {
      const auto& obj = g_gxState.loadedTextures[map];
      sig = (sig ^ reinterpret_cast<uintptr_t>(obj.data) ^ obj.texDataVersion) * 0x9E3779B97F4A7C15ull + map;
    }
  }
  static u64 lastSig = 0;
  static const XrEmissive* last = nullptr;
  static const GXTexObj_* lastTex = nullptr;
  if (sig == lastSig) {
    if (texOut != nullptr)
      *texOut = lastTex;
    return last;
  }
  lastSig = sig;
  last = nullptr;
  lastTex = nullptr;
  for (u32 i = 0; i < sc.tevStageCount && last == nullptr; ++i) {
    const auto map = sc.tevStages[i].texMapId;
    if (map == GX_TEXMAP_NULL || map >= MaxTextures)
      continue;
    const auto& obj = g_gxState.loadedTextures[map];
    if (obj.data == nullptr)
      continue;
    const u32 fmt = obj.format();
    const bool colour = fmt == GX_TF_RGB565 || fmt == GX_TF_RGB5A3 || fmt == GX_TF_RGBA8;
    if (sLinkTag - 1 <= 4 && colour && g_gxState.copyTextures.contains(obj.data))
      last = &screen, lastTex = &obj;
    else if (const auto* e = xr_emissive_entry(xr_texture_key(obj)))
      last = e, lastTex = &obj;
  }
  if (texOut != nullptr)
    *texOut = lastTex;
  return last;
}

// A texture's mean colour (by alpha), once per key.
static std::array<float, 3> xr_texture_mean(const GXTexObj_& obj, u64 key) noexcept {
  static absl::flat_hash_map<u64, std::array<float, 3>> means;
  if (const auto it = means.find(key); it != means.end())
    return it->second;
  std::array<float, 3> mean{1.f, 1.f, 1.f};
  const auto px = xr_texture_rgba(obj);
  const size_t n = static_cast<size_t>(obj.width()) * obj.height();
  if (!px.data.empty()) {
    double sum[3]{}, wsum = 0.0;
    for (size_t p = 0; p < n; ++p) {
      const u8* c = px.data.data() + p * 4;
      const double w = c[3] / 255.0;
      for (int k = 0; k < 3; ++k)
        sum[k] += w * c[k] / 255.0;
      wsum += w;
    }
    if (wsum > 0.0)
      for (int k = 0; k < 3; ++k)
        mean[k] = static_cast<float>(sum[k] / wsum);
  }
  Log.info("XR room light: emitter texture {:016x}, mean colour {:.2f} {:.2f} {:.2f}", key, mean[0], mean[1], mean[2]);
  return means[key] = mean;
}

// The current draw's vertices' box in the game's world (false: positions it
// can't read).
static bool xr_draw_world_box(const u8* raw, u16 vtxCount, float* lo, float* hi) noexcept {
  const float* view = gfx::xr_world_view();
  const auto& config = sDrawCache.config.shaderConfig;
  const auto& pos = config.attrs[GX_VA_POS];
  const auto& idx = config.attrs[GX_VA_PNMTXIDX];
  if (view == nullptr || pos.attrType == GX_NONE || (idx.attrType != GX_NONE && idx.attrType != GX_DIRECT))
    return false;
  const u32 compSize = comp_type_size(GX_VA_POS, static_cast<GXCompType>(pos.compType));
  const u32 comps = std::min<u32>(pos.cnt, 3);
  std::fill_n(lo, 3, INFINITY);
  std::fill_n(hi, 3, -INFINITY);
  for (u32 v = 0; v < vtxCount; ++v) {
    const u8* vtx = raw + v * config.vtxStride;
    const u32 mtx = idx.attrType == GX_DIRECT ? std::min<u32>(vtx[idx.offset] / 3u, MaxPnMtx - 1)
                                              : std::min<u32>(g_gxState.currentPnMtx, MaxPnMtx - 1);
    bool le = false;
    const u8* src = decoded_source(pos, GX_VA_POS, 0, vtx, le);
    if (src == nullptr)
      return false;
    float p[3] = {0.f, 0.f, 0.f};
    for (u32 c = 0; c < comps; ++c)
      p[c] = decode_component(src + c * compSize, pos.compType, pos.frac, le);
    // Model -> game camera -> the world (the view's inverse: its rotation's
    // transpose).
    const f32* m = reinterpret_cast<const f32*>(&g_gxState.pnMtx[mtx].pos);
    float cam[3];
    for (int r = 0; r < 3; ++r)
      cam[r] = m[r * 4] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3] - view[r * 4 + 3];
    for (int c = 0; c < 3; ++c) {
      const float w = view[c] * cam[0] + view[4 + c] * cam[1] + view[8 + c] * cam[2];
      lo[c] = std::min(lo[c], w);
      hi[c] = std::max(hi[c], w);
    }
  }
  return lo[0] <= hi[0];
}

// An emitter's draw (a world draw with an emissive entry that has one): its
// box into this frame's emitter.
static void xr_room_emitter(const u8* raw, u16 vtxCount) noexcept {
  if (gfx::xr_room_light() == 0 || !gfx::xr_recording_world() || sLinkTag == 0 || sLinkTag - 1 > 4)
    return;
  const GXTexObj_* tex = nullptr;
  const auto* e = xr_room_emissive(&tex);
  float lo[3], hi[3];
  if (e == nullptr || e->emitter < 0 || tex == nullptr || !xr_draw_world_box(raw, vtxCount, lo, hi))
    return;
  auto rgb = xr_texture_mean(*tex, xr_texture_key(*tex));
  for (float& c : rgb)
    c *= e->gain;
  gfx::xr_add_emitter(e->emitter, lo, hi, rgb.data(), e->reach);
}

// The current draw's class by its link, lit or not: 0 none, 1 fighters, 2
// the stage.
static u8 xr_link_class() noexcept {
  if (sLinkTag == 0 || xr_additive(sDrawCache.config))
    return 0;
  const u32 link = sLinkTag - 1;
  if (link <= 6 && xr_room_emissive() != nullptr)
    return 0;
  return link == 5 || link == 6 ? 1 : link <= 4 ? 2 : 0;
}

static u8 xr_room_class() noexcept {
  if (gfx::xr_room_light() == 0 || sLinkTag == 0 || xr_additive(sDrawCache.config))
    return 0;
  if (sLinkTag - 1 <= 6)
    xr_room_texdump();
  return xr_link_class();
}

// A triangle draw without its own normals, its every attribute decoded, with
// room for SMOOTH shading's (the last config's answer stands for its draws;
// `fresh`: a config not yet in sDrawCache.pipelineRef).
static bool xr_smooth_fits(const ShaderConfig& sc, bool fresh = false) noexcept {
  if (!sc.decodedAll || sc.lineMode != 0 || sDrawPrim == 0 || sc.attrs[GX_VA_NRM].attrType != GX_NONE)
    return false;
  static gfx::PipelineRef lastRef = 0;
  static bool lastFits = false;
  if (!fresh && sDrawCache.pipelineRef == lastRef && lastRef != 0)
    return lastFits;
  ShaderConfig with = sc;
  with.xrSmoothNrm = 1;
  DecodedLayout layout;
  const bool fits = decoded_layout(with, layout);
  if (!fresh) {
    lastRef = sDrawCache.pipelineRef;
    lastFits = fits;
  }
  return fits;
}

// SMOOTH shading's normals for the current draw (ShaderConfig::xrSmoothNrm):
// a stage draw that fits them.
static bool xr_room_smooth(const ShaderConfig& sc, bool fresh = false) noexcept {
  room_fit_tick();
  return (gfx::xr_room_light() & 32u) != 0 && xr_room_class() == 2 && xr_smooth_fits(sc, fresh);
}

// The current draw's variant under Roomtone, whatever the lighting now
// (gx::roomtone_config), for the pipeline cache to record against the
// pipelines it asks for: its eye pipelines' (`eye`), or its others', which
// only SMOOTH normals change. Without the eyes, none (0xFF).
static u8 xr_roomtone_variant(bool eye) noexcept {
  if (gfx::xr_multiview_layout() == nullptr)
    return 0xFF;
  const u8 cls = xr_link_class();
  const bool smooth = cls == 2 && xr_smooth_fits(sDrawCache.config.shaderConfig);
  if (!eye)
    return smooth ? 8 : 0;
  return cls | (cls == 2 && gfx::xr_roomtone_gloss() ? 4 : 0) | (smooth ? 8 : 0);
}

// AURORA_XR_ROOM_CENSUS=1 (a survey, with or without the room light): how
// the stage's world draws get their colour, by vertices: lit or not, with normals or not, and for the
// unlit ones where any baked light would be: vertex colours the TEV reads
// (the material from the vertex), a constant colour, or textures alone;
// additive draws (glows) apart. Every 5000 of them.
static void xr_room_census(u16 vtxCount) noexcept {
  static const bool on = std::getenv("AURORA_XR_ROOM_CENSUS") != nullptr;
  if (!on)
    return;
  // The world's draws the room doesn't light, by link (unmarked as -1) and
  // whether additive: effects (link 7 and up) and glows by design, anything
  // else a stage part left out.
  static std::map<int, std::array<u64, 2>> skipped; // link * 2 + additive: draws, vertices
  const bool skip = sLinkTag == 0 || sLinkTag - 1 > 6 || xr_additive(sDrawCache.config);
  if (skip) {
    auto& e = skipped[(static_cast<int>(sLinkTag) - 1) * 2 + (xr_additive(sDrawCache.config) ? 1 : 0)];
    ++e[0];
    e[1] += vtxCount;
  }
  if (sLinkTag == 0 || sLinkTag - 1 > 4)
    return;
  const auto& c = sDrawCache.config;
  const auto& sc = c.shaderConfig;
  bool ras = false, tex = false;
  for (u32 i = 0; i < sc.tevStageCount; ++i) {
    const auto& st = sc.tevStages[i];
    for (const auto arg : {st.colorPass.a, st.colorPass.b, st.colorPass.c, st.colorPass.d})
      ras |= arg == GX_CC_RASC || arg == GX_CC_RASA;
    tex |= st.texMapId != GX_TEXMAP_NULL;
  }
  const auto& cc = sc.colorChannels[0];
  const bool additive = xr_additive(c);
  const bool nrm = sc.attrs[GX_VA_NRM].attrType != GX_NONE;
  const bool vtxClr = cc.matSrc == GX_SRC_VTX && sc.attrs[GX_VA_CLR0].attrType != GX_NONE;
  enum { Additive, LitNrm, LitNoNrm, UnlitVtx, UnlitConst, UnlitTex, Other, Kinds };
  const int kind = additive             ? Additive
                   : cc.lightingEnabled ? (nrm ? LitNrm : LitNoNrm)
                   : ras && vtxClr      ? UnlitVtx
                   : ras                ? UnlitConst
                   : tex                ? UnlitTex
                                        : Other;
  static std::array<u64, Kinds> verts{}, draws{};
  static std::array<u64, Kinds> nrms{};
  static u32 n = 0;
  verts[kind] += vtxCount;
  ++draws[kind];
  nrms[kind] += nrm ? vtxCount : 0;
  if (++n % 5000 != 0)
    return;
  u64 total = 0;
  for (const u64 v : verts)
    total += v;
  constexpr std::array<const char*, Kinds> names{"additive", "lit, normals", "lit, no normals",
                                                 "unlit, vertex colours", "unlit, constant colour",
                                                 "unlit, textures only", "other"};
  std::string line;
  for (int k = 0; k < Kinds; ++k)
    if (verts[k] != 0)
      line += fmt::format(" {} {:.0f}% ({} draws{});", names[k], 100.0 * verts[k] / std::max<u64>(total, 1), draws[k],
                          nrms[k] != 0 && k > LitNoNrm ? fmt::format(", {:.0f}% with normals", 100.0 * nrms[k] / verts[k])
                                                       : std::string{});
  std::string out;
  for (const auto& [key, e] : skipped)
    out += fmt::format(" link {}{} {} draws ({} vertices);", key < 0 ? -1 : key / 2, key < 0 ? (key % 2 != 0 ? " additive" : "") : key % 2 != 0 ? " additive" : "", e[0], e[1]);
  Log.info("XR room census: stage draws by vertices:{} not lit by the room:{}", line, out);
  verts = {}, draws = {}, nrms = {};
  skipped.clear();
}

// What the eye config adds to the twins' cache keys (0 with the room light
// off, so every key stays as it was).
static uint64_t xr_room_salt() noexcept {
  const u8 cls = xr_room_class();
  return cls == 0 ? 0 : (gfx::xr_room_light() << 2 | cls) * 0x9E3779B97F4A7C15ull;
}

// The eye pipelines' config. AURORA_XR_EYE_NOLIGHT=1 (a measurement, it
// looks wrong): no GX lighting in the eyes, to see what the vertex stage's
// lighting costs there (it runs once per eye with multiview). With the room
// light on, a fighter or stage draw takes its class and the room's options
// (ShaderConfig::xrRoom*).
static const PipelineConfig& xr_eye_config() noexcept {
  static const bool noLight = std::getenv("AURORA_XR_EYE_NOLIGHT") != nullptr;
  const u8 cls = xr_room_class();
  if (!noLight && cls == 0)
    return sDrawCache.config;
  static PipelineConfig c;
  c = sDrawCache.config;
  if (noLight)
    for (auto& ch : c.shaderConfig.colorChannels) {
      ch.lightingEnabled = false;
      ch.diffFn = GX_DF_NONE;
      ch.attnFn = GX_AF_NONE;
    }
  if (cls != 0) {
    const u32 room = gfx::xr_room_light();
    c.shaderConfig.xrRoom = cls;
    c.shaderConfig.xrRoomFacet = (room >> 4) & 1;
    c.shaderConfig.xrRoomPixel = ((room >> 1) & 1) | c.shaderConfig.xrRoomFacet;
    c.shaderConfig.xrRoomView = (room >> 2) & 3;
    c.shaderConfig.xrRoomGloss = cls == 2 && ((room >> 6) & 1);
  }
  return c;
}

// The multiview twin of the current pipeline for `layout`, cached per
// (pipeline, layout).
static gfx::PipelineRef xr_twin(const gfx::RenderTargetLayout& layout, gfx::PipelineRef& ref,
                                gfx::PipelineRef& forPipeline, uint64_t& forLayout) noexcept {
  auto& cache = sDrawCache;
  // The Roomtone variant too: a config drawn as another's is recorded once more.
  const u8 variant = xr_roomtone_variant(true);
  const uint64_t key = layout.key ^ xr_room_salt() ^ (variant + 1ull) * 0xC2B2AE3D27D4EB4Full;
  if (forPipeline != cache.pipelineRef || forLayout != key) {
    ref = gfx::find_pipeline(xr_eye_config(), layout, variant);
    forPipeline = cache.pipelineRef;
    forLayout = key;
  }
  return ref;
}

// `opacityOnly`: the fade has no plane in play, only an opacity (the
// variant without plane distances, xrSoftClip 2).
static gfx::PipelineRef xr_soft_twin(const gfx::RenderTargetLayout& mv, bool opacityOnly = false) noexcept {
  static gfx::RenderTargetLayout soft[2];
  static uint64_t softFor[2] = {0, 0};
  const int v = opacityOnly ? 1 : 0;
  if (softFor[v] != mv.key) {
    softFor[v] = mv.key;
    soft[v] = mv;
    soft[v].xrSoftClip = opacityOnly ? 2 : 1;
    gfx::detail::finalize_render_target_layout(soft[v]);
  }
  const auto& soft_ = soft[v];
  auto& cache = sDrawCache;
  return xr_twin(soft_, cache.xrSoftRef, cache.xrSoftForPipeline, cache.xrSoftForLayout);
}

// The fade has only an opacity: every clip plane passes (0, 0, 0, 1).
static bool xr_fade_opacity_only() noexcept {
  const auto* clip = gfx::xr_clip_camera();
  if (clip == nullptr || clip->opacity >= 1.f)
    return false;
  for (const auto& p : clip->planes)
    if (p[0] != 0.f || p[1] != 0.f || p[2] != 0.f || p[3] != 1.f)
      return false;
  return true;
}

// The eye pipeline for a world draw's vertices (0: not drawn in the eyes).
// `cls`, when given: how the draw lies against the clip (Inside without
// one); `outside`/`masked` as for classify_xr_clip.
static gfx::PipelineRef xr_world_pipeline(const u8* raw, u16 vtxCount, XrClipClass* clsOut = nullptr,
                                          u8* outside = nullptr, bool* masked = nullptr,
                                          u8* unsolid = nullptr) noexcept {
  if (clsOut != nullptr)
    *clsOut = XrClipClass::Inside;
  if (masked != nullptr)
    *masked = false;
  if (!gfx::xr_recording_world()) {
    return 0;
  }
  const auto* mv = gfx::xr_multiview_layout();
  if (mv == nullptr) {
    return 0;
  }
  auto& cache = sDrawCache;
  if (gfx::xr_soft_clip_active()) {
    const auto cls = classify_xr_clip(cache.config.shaderConfig, raw, vtxCount, outside, masked, unsolid);
    if (clsOut != nullptr)
      *clsOut = cls;
    // AURORA_XR_CLIP_STATS=1: how clipped draws were classed, every 20000.
    static const bool stats = std::getenv("AURORA_XR_CLIP_STATS") != nullptr;
    if (stats) {
      static u32 draws[3], verts[3], n;
      ++draws[u8(cls)];
      verts[u8(cls)] += vtxCount;
      if (++n % 20000 == 0) {
        Log.info("XR clip draws (verts): inside {} ({}), straddle {} ({}), outside {} ({})", draws[0], verts[0],
                 draws[1], verts[1], draws[2], verts[2]);
        std::fill_n(draws, 3, 0u), std::fill_n(verts, 3, 0u);
      }
    }
    switch (cls) {
    case XrClipClass::Outside:
      return 0;
    case XrClipClass::Straddle:
      return xr_soft_twin(*mv, xr_fade_opacity_only());
    case XrClipClass::Inside:
      break;
    }
  }
  return xr_twin(*mv, cache.xrPipelineRef, cache.xrForPipeline, cache.xrForLayout);
}

// How a world draw goes into the SpaceWarp depth and motion (None: not at
// all). Plain: opaque, writing depth and colour, no alpha test or depth
// textures: the SpaceWarp depth and motion are of the nearest solid surface;
// whatever is blended over it is not. Cutout: the same but alpha-tested
// (trees, fences, leaves), whose twin keeps the game's fragment stage for
// the test, so only the surviving texels carry depth and motion. Translucent:
// source-alpha blended without depth writes (leaves, fences of mist): the
// cutout twin with a cut at half alpha, so the mostly solid texels carry the
// surface's depth and motion and the faint edges leave what is behind them.
enum class XrMotionKind { None, Plain, Cutout, Translucent };
static XrMotionKind xr_motion_kind() noexcept {
  if (gfx::xr_motion_layout() == nullptr || !gfx::xr_recording_world())
    return XrMotionKind::None;
  const auto& c = sDrawCache.config;
  if (!(c.depthCompare && c.colorUpdate && c.shaderConfig.lineMode == 0 && c.shaderConfig.zTexOp == 0))
    return XrMotionKind::None;
  if (!c.depthUpdate) {
    const bool translucent = (gfx::xr_motion_classes() & 2) != 0;
    // AURORA_XR_APPSW_TRANS_PLAIN=1 (a test): the whole draw, as if opaque.
    static const bool asPlain = std::getenv("AURORA_XR_APPSW_TRANS_PLAIN") != nullptr;
    return translucent && c.blendMode == GX_BM_BLEND && c.blendFacSrc == GX_BL_SRCALPHA &&
                   c.blendFacDst == GX_BL_INVSRCALPHA
               ? (asPlain ? XrMotionKind::Plain : XrMotionKind::Translucent)
               : XrMotionKind::None;
  }
  if (c.shaderConfig.alphaCompare)
    return (gfx::xr_motion_classes() & 1) != 0 ? XrMotionKind::Cutout : XrMotionKind::None;
  const bool opaque = c.blendMode == GX_BM_NONE ||
                      (c.blendMode == GX_BM_BLEND && c.blendFacSrc == GX_BL_ONE && c.blendFacDst == GX_BL_ZERO);
  return opaque ? XrMotionKind::Plain : XrMotionKind::None;
}

// AURORA_XR_MOTION_WHY=1: every 2000 world draws, which draws SpaceWarp's
// depth and motion leave out and why, by the game's GX link (set
// AURORA_XR_LINK_STATS=1 for the link markers). Works without SpaceWarp.
static void xr_motion_why(u32 vtxCount) noexcept {
  static const bool on = std::getenv("AURORA_XR_MOTION_WHY") != nullptr;
  if (!on || !gfx::xr_recording_world())
    return;
  static u32 n, ok;
  static std::unordered_map<std::string, u32> combos;
  const auto& c = sDrawCache.config;
  const bool opaque = c.blendMode == GX_BM_NONE ||
                      (c.blendMode == GX_BM_BLEND && c.blendFacSrc == GX_BL_ONE && c.blendFacDst == GX_BL_ZERO);
  const bool alphaTest = static_cast<bool>(c.shaderConfig.alphaCompare);
  const bool solid = c.depthCompare && c.depthUpdate && c.colorUpdate && c.shaderConfig.lineMode == 0 &&
                     c.shaderConfig.zTexOp == 0;
  ++n;
  const auto kind = xr_motion_kind();
  if (kind == XrMotionKind::Plain)
    ++ok;
  else
    ++combos[fmt::format("{} link={} blend={} src={} dst={} ztest={} zwrite={} alphacmp={} tex={} tevStages={} verts~{}",
                         kind == XrMotionKind::Cutout        ? "CUTOUT"
                         : kind == XrMotionKind::Translucent ? "TRANSLUCENT"
                                                             : "left out",
                         static_cast<int>(sLinkTag) - 1, static_cast<int>(c.blendMode),
                         static_cast<int>(c.blendFacSrc), static_cast<int>(c.blendFacDst), c.depthCompare,
                         c.depthUpdate, alphaTest, sDrawCache.bindGroups.textureBindGroup != 0,
                         c.shaderConfig.tevStageCount, vtxCount)];
  if (n % 2000 == 0) {
    Log.info("XR motion why: {} draws, {} plain, rest:", n, ok);
    for (const auto& [k, v] : combos)
      Log.info("XR motion why:   {} x {}", v, k);
    n = ok = 0;
    combos.clear();
  }
}

// The motion twin's config: only what decides where a vertex lands and how
// its depth tests (vertex format, depth state, culling, depth bias). The
// fragment stage is replaced and the GX outputs are dead code, so materials,
// lighting, texgen and blending don't matter: draws share a handful of
// pipelines instead of one per material (fewer pipeline switches in the
// pass, far fewer pipelines to compile).
static PipelineConfig xr_motion_config(const PipelineConfig& from) noexcept {
  PipelineConfig c = from;
  const ShaderConfig& sc = from.shaderConfig;
  c.shaderConfig = {};
  c.shaderConfig.vtxStride = sc.vtxStride;
  c.shaderConfig.lineMode = sc.lineMode;
  c.shaderConfig.decodedPos = sc.decodedPos;
  c.shaderConfig.decodedAll = sc.decodedAll;
  c.shaderConfig.xrSmoothNrm = sc.xrSmoothNrm; // in the decoded vertex: its stride
  c.shaderConfig.attrs = sc.attrs;
  c.shaderConfig.tevStageCount = 1; // one stage of zeros: nothing sampled, nothing lit
  c.blendMode = GX_BM_NONE;
  c.blendFacSrc = GX_BL_ONE;
  c.blendFacDst = GX_BL_ZERO;
  c.blendOp = GX_LO_CLEAR;
  c.dstAlpha = UINT32_MAX;
  c.alphaUpdate = true;
  c.colorUpdate = true;
  return c;
}

// `clipped`: the twin for a draw straddling a clip, with the eye's clip test
// (a hard cut or a dithered fade), else its parts past the clip would be
// missing from SpaceWarp's depth and motion.
// The motion layout for a kind of draw (null once adaptive SpaceWarp turns it
// off), the base one's variants remade when it changes.
static const gfx::RenderTargetLayout* xr_motion_layout_for(XrMotionKind kind, bool clipped, size_t& slot) noexcept {
  const auto* base = gfx::xr_motion_layout();
  if (base == nullptr || kind == XrMotionKind::None)
    return nullptr;
  slot = (clipped ? 1 : 0) + (kind == XrMotionKind::Cutout ? 2 : kind == XrMotionKind::Translucent ? 4 : 0);
  static gfx::RenderTargetLayout variants[6];
  static uint64_t variantsFor = 0;
  if (variantsFor != base->key) {
    variantsFor = base->key;
    for (size_t k = 0; k < 6; ++k) {
      variants[k] = *base;
      variants[k].xrSoftClip = k & 1 ? 1 : 0;
      variants[k].xrMotion = 1 + (k >> 1);
      gfx::detail::finalize_render_target_layout(variants[k]);
    }
  }
  return &variants[slot];
}

// The pipeline config of a kind's motion twin. A cutout or translucent twin
// needs the game's fragment stage for its alpha test; the translucent one
// writes the depth its draw leaves out.
static PipelineConfig xr_motion_twin_config(XrMotionKind kind) noexcept {
  static const bool shared = [] {
    const char* v = std::getenv("AURORA_XR_APPSW_SHARED");
    return v == nullptr || *v != '0';
  }();
  PipelineConfig config = sDrawCache.config;
  config.depthUpdate = true; // translucent draws don't write depth in the eyes
  return shared && kind == XrMotionKind::Plain ? xr_motion_config(config) : config;
}

static gfx::PipelineRef xr_motion_pipeline(XrMotionKind kind, bool clipped = false) noexcept {
  size_t slot = 0;
  const auto* layout = xr_motion_layout_for(kind, clipped, slot);
  if (layout == nullptr)
    return 0;
  auto& cache = sDrawCache;
  auto& twin = cache.xrMotionTwin[slot];
  if (twin.forPipeline != cache.pipelineRef || twin.forLayout != layout->key) {
    twin.ref = gfx::find_pipeline(xr_motion_twin_config(kind), *layout, xr_roomtone_variant(false));
    twin.forPipeline = cache.pipelineRef;
    twin.forLayout = layout->key;
  }
  return twin.ref;
}

// XR batching: a layout's batched twin (gx::xr_batch_*).
static const gfx::RenderTargetLayout& xr_batch_layout(const gfx::RenderTargetLayout& of) noexcept {
  static gfx::RenderTargetLayout batched[2];
  static uint64_t batchedFor[2] = {0, 0};
  const int i = of.xrMotion != 0 ? 1 : 0;
  if (batchedFor[i] != of.key) {
    batchedFor[i] = of.key;
    batched[i] = of;
    // Eye batches sample through the resource table when bindless.
    batched[i].xrBatch = i == 0 && gfx::xr_bindless() ? 2 : 1;
    gfx::detail::finalize_render_target_layout(batched[i]);
  }
  return batched[i];
}

// The batched twins of a draw's eye and motion pipelines (0: never batched:
// lines and points, or batching off).
static gfx::PipelineRef xr_batch_pipeline(bool motion, XrMotionKind kind = XrMotionKind::Plain) noexcept {
  auto& cache = sDrawCache;
  if (!gfx::detail::xr_batching() || cache.config.shaderConfig.lineMode != 0)
    return 0;
  size_t slot = 0;
  const auto* base = motion ? xr_motion_layout_for(kind, false, slot) : gfx::xr_multiview_layout();
  if (base == nullptr)
    return 0;
  const auto& layout = xr_batch_layout(*base);
  auto& ref = motion ? cache.xrMotionBatchRef : cache.xrBatchRef;
  auto& forPipeline = motion ? cache.xrMotionBatchForPipeline : cache.xrBatchForPipeline;
  auto& forLayout = motion ? cache.xrMotionBatchForLayout : cache.xrBatchForLayout;
  const u8 variant = xr_roomtone_variant(!motion);
  const uint64_t key = layout.key ^ (motion ? 0 : xr_room_salt()) ^ (variant + 1ull) * 0xC2B2AE3D27D4EB4Full;
  if (forPipeline != cache.pipelineRef || forLayout != key) {
    ref = gfx::find_pipeline(motion ? xr_motion_twin_config(kind) : xr_eye_config(), layout, variant);
    forPipeline = cache.pipelineRef;
    forLayout = key;
  }
  return ref;
}

// SpaceWarp motion vectors: each opaque world draw's position matrices are
// kept by what it draws (its vertex bytes and position array), so next frame
// the same draw finds where its vertices were. Display lists are copied into
// the FIFO, so the bytes, not their address, are what stays the same. Draws
// of the same thing (Mushroom Kingdom's blocks, one model each) are told
// apart by where they are: each takes the nearest of last frame's not yet
// taken, if nearer than half the gap between last frame's copies. By order
// alone, a block that blinks as it respawns (drawn every other frame)
// shifted the blocks after it onto their neighbours' last positions, and the
// blinking block itself took a neighbour's: SpaceWarp warped them by a block
// width. A draw found gets last frame's matrices pushed to the storage
// buffer; one not found (new, immediate-mode data that changes, back from
// hidden, or a jump: a respawn, a cut) is taken as fixed in the game world.
namespace {
struct XrPrevMatrices {
  std::array<Mat3x4<float>, MaxPnMtx> pos;
  bool taken = false;  // matched by a draw this frame (in prev)
  float gap2 = -1.f;   // squared distance to the nearest other copy (in prev; -1 not yet)
};
struct XrMotionHistory {
  uint64_t seq = 0;
  std::unordered_map<uint64_t, std::vector<XrPrevMatrices>> prev, cur;
} sXrMotion;
// aurora_xr_world_still: the draws go in as fixed in the game world.
bool sXrStill = false;
// Over this many draws of the same thing, nearest-matching costs too much;
// they go by order.
constexpr size_t kXrNearestMax = 64;
float xr_dist2(const Mat3x4<float>& a, const Mat3x4<float>& b) noexcept {
  const float dx = a.m0[3] - b.m0[3], dy = a.m1[3] - b.m1[3], dz = a.m2[3] - b.m2[3];
  return dx * dx + dy * dy + dz * dz;
}
} // namespace

static u32 xr_prev_matrices(const u8* raw, size_t bytes, GXVtxFmt fmt) noexcept {
  // AURORA_XR_APPSW_MATCH=0: every draw as if fixed in the game world.
  static const bool match = [] {
    const char* v = std::getenv("AURORA_XR_APPSW_MATCH");
    return v == nullptr || *v != '0';
  }();
  if (!match || sXrStill)
    return 0;
  ZoneScoped;
  auto& h = sXrMotion;
  if (const uint64_t seq = gfx::detail::recording_seq(); seq != h.seq) {
    // Frames went by unmatched (no world, or SpaceWarp off): too old.
    const bool consecutive = seq == h.seq + 1;
    h.seq = seq;
    h.prev.swap(h.cur);
    if (!consecutive)
      h.prev.clear();
    h.cur.clear();
  }
  const auto& state = g_gxState;
  uint64_t content = XXH3_64bits_withSeed(raw, bytes, fmt);
  content ^= reinterpret_cast<uintptr_t>(state.arrays[GX_VA_POS].data) * 0x9E3779B97F4A7C15ull;
  XrPrevMatrices now;
  for (u32 i = 0; i < MaxPnMtx; ++i)
    now.pos[i] = state.pnMtx[i].pos;
  auto& mine = h.cur[content];
  const size_t order = mine.size();
  mine.push_back(now);
  const u32 m = std::min<u32>(state.currentPnMtx, MaxPnMtx - 1);
  XrPrevMatrices* found = nullptr;
  float found2 = 0.f;
  if (const auto it = h.prev.find(content); it != h.prev.end()) {
    auto& last = it->second;
    if (last.size() == 1) {
      found = &last[0];
      found2 = xr_dist2(found->pos[m], now.pos[m]);
    } else if (last.size() > kXrNearestMax) {
      if (order < last.size()) {
        found = &last[order];
        found2 = xr_dist2(found->pos[m], now.pos[m]);
      }
    } else {
      if (last[0].gap2 < 0.f) {
        for (auto& p : last) {
          p.gap2 = std::numeric_limits<float>::max();
          for (const auto& q : last)
            if (&p != &q)
              p.gap2 = std::min(p.gap2, xr_dist2(p.pos[m], q.pos[m]));
        }
      }
      for (auto& p : last) {
        if (p.taken)
          continue;
        // Copies drawn at one spot (a model drawn twice) have no gap: the
        // cut's limit only.
        const float d = xr_dist2(p.pos[m], now.pos[m]);
        const float limit = p.gap2 < 1e-4f ? 40.f * 40.f : p.gap2 * 0.25f;
        if (d < limit && (found == nullptr || d < found2)) {
          found = &p;
          found2 = d;
        }
      }
    }
  }
  // AURORA_XR_MOTION_STATS=1: how many opaque world draws found themselves
  // in last frame, every 20000.
  static const bool stats = std::getenv("AURORA_XR_MOTION_STATS") != nullptr;
  if (stats) {
    static u32 matched, n;
    matched += found != nullptr ? 1 : 0;
    if (++n % 20000 == 0) {
      Log.info("XR motion: {} of {} opaque world draws matched last frame's", matched, n);
      matched = n = 0;
    }
  }
  if (found == nullptr)
    return 0;
  found->taken = true;
  // A jump of over 40 units (camera space) in the draw's own matrix is a
  // cut, not motion.
  if (found2 > 40.f * 40.f)
    return 0;
  const auto range = gfx::push_storage(reinterpret_cast<const u8*>(found->pos.data()), sizeof(found->pos));
  return range.offset / sizeof(u32) + 1; // 0 is none
}

// The eye pipeline without the clip's discard, for a straddling draw's
// triangles wholly inside the solid region.
static gfx::PipelineRef xr_solid_pipeline() noexcept {
  const auto* mv = gfx::xr_multiview_layout();
  if (mv == nullptr)
    return 0;
  auto& cache = sDrawCache;
  return xr_twin(*mv, cache.xrPipelineRef, cache.xrForPipeline, cache.xrForLayout);
}
#endif

static void push_gx_draw(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange, gfx::Range idxRange,
                         u32 numIndices, const u8* raw) noexcept {
  auto& state = g_gxState;
  auto& cache = sDrawCache;

  DrawImmediateData immediates{.vtxStart = vertRange.offset, .currentPnMtx = state.currentPnMtx};
  for (int i = GX_VA_POS; i <= GX_VA_TEX7; ++i) {
    if (state.vtxDesc[i] != GX_INDEX8 && state.vtxDesc[i] != GX_INDEX16) {
      continue;
    }
    auto& array = state.arrays[i];
    if (array.cachedRange.size == 0) {
      array.cachedRange = gfx::push_storage(static_cast<const uint8_t*>(array.data), array.size);
    }
    immediates.arrayStart[i - GX_VA_POS] = array.cachedRange.offset;
  }

  const u8 lineMode = line_mode_for_prim(prim);
  const auto targetLayoutKey = gfx::get_render_target_layout().key;
  const bool pipelineValid = cache.hasPipeline && (state.dirty & DirtyPipeline) == 0 && cache.fmt == fmt &&
                             cache.lineMode == lineMode && cache.targetLayoutKey == targetLayoutKey;
  if (!pipelineValid) {
    const bool hadPipeline = cache.hasPipeline;
    const auto prevSampledTextures = cache.shaderInfo.sampledTextures;
    const auto prevSampledIndTextures = cache.shaderInfo.sampledIndTextures;
    populate_pipeline_config(cache.config, prim, fmt);
#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
    // SMOOTH normals first: the pipeline without them was asked for (and
    // built) for nothing.
    cache.config.shaderConfig.xrSmoothNrm = xr_room_smooth(cache.config.shaderConfig, true);
#endif
    cache.shaderInfo = build_shader_info(cache.config.shaderConfig);
    cache.pipelineRef = gfx::pipeline_ref(cache.config);
    cache.targetLayoutKey = targetLayoutKey;
    cache.fmt = fmt;
    cache.lineMode = lineMode;
    cache.hasPipeline = true;
    state.dirty = (state.dirty & ~DirtyPipeline) | DirtyUniform;
    if (!hadPipeline || prevSampledTextures != cache.shaderInfo.sampledTextures ||
        prevSampledIndTextures != cache.shaderInfo.sampledIndTextures) {
      cache.bindGeneration = 0;
    }
  }
#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
  // Not GX state: follows the room light and the draw's link.
  if (const bool smooth = xr_room_smooth(cache.config.shaderConfig); smooth != cache.config.shaderConfig.xrSmoothNrm) {
    cache.config.shaderConfig.xrSmoothNrm = smooth;
    cache.pipelineRef = gfx::pipeline_ref(cache.config);
  }
  // Its Roomtone variant for the pipeline cache, as the eye and motion
  // pipelines record theirs.
  if (const u8 variant = xr_roomtone_variant(false); variant != 0xFF) {
    static gfx::PipelineRef notedRef = 0;
    static u8 notedVariant = 0xFF;
    if (cache.pipelineRef != notedRef || variant != notedVariant) {
      gfx::note_roomtone_variant(cache.config, gfx::get_render_target_layout(), variant);
      notedRef = cache.pipelineRef;
      notedVariant = variant;
    }
  }
#endif

  const bool bindGroupsValid =
      (state.dirty & DirtyTextures) == 0 && cache.bindGeneration == texture::current_bind_generation();
  if (!bindGroupsValid) {
    const auto prevBindGroup = cache.bindGroups.textureBindGroup;
    resolve_sampled_textures(cache.shaderInfo);
    cache.bindGroups = build_bind_groups(cache.shaderInfo);
    // AURORA_LOG_TEV=1: when a draw ends up sampling no texture, report what
    // the TEV stages actually asked for. A stage whose texMapId is
    // GX_TEXMAP_NULL means the game never pointed the stage at a texture.
    if (!cache.bindGroups.textureBindGroup && std::getenv("AURORA_LOG_TEV") != nullptr) {
      static uint64_t n = 0;
      if ((n++ % 4000) == 0) {
        const auto& sc = cache.config.shaderConfig;
        fmt::print(stderr, "no-tex draw #{}: tevStages={}", n, sc.tevStageCount);
        for (uint32_t i = 0; i < sc.tevStageCount && i < 4; ++i) {
          fmt::print(stderr, " [s{} texMap={} texCoord={} coloridx={}]", i, static_cast<int>(sc.tevStages[i].texMapId),
                     static_cast<int>(sc.tevStages[i].texCoordId), static_cast<int>(sc.tevStages[i].channelId));
        }
        fmt::print(stderr, "\n");
      }
    }
    cache.bindGeneration = texture::current_bind_generation();
    state.dirty &= ~DirtyTextures;
    // For texture_size_bias uniform
    if (cache.bindGroups.textureBindGroup != prevBindGroup) {
      state.dirty |= DirtyUniform;
    }
  }

  const bool uniformValid = (state.dirty & DirtyUniform) == 0 && cache.uniformRange.size != 0;
  if (!uniformValid) {
    cache.uniformRange = build_uniform(cache.shaderInfo, &cache.uniformSharedKey);
    state.dirty &= ~DirtyUniform;
  }
  if (cache.config.shaderConfig.fogRangeEnabled) {
    const auto key = fog_range_lut_key();
    if (!cache.hasFogRange || cache.fogRangeKey != key) {
      cache.fogRange = push_fog_range_lut(key);
      cache.fogRangeKey = key;
      cache.hasFogRange = true;
    }
  }
  immediates.fogRangeBase = cache.fogRange.offset / sizeof(u32);

  state.dirty &= ~DirtyImmediates;

  uint32_t instanceCount = 1;
  if (prim == GX_LINES) {
    instanceCount = vtxCount / 2;
  } else if (prim == GX_LINESTRIP) {
    instanceCount = vtxCount - 1;
  } else if (prim == GX_POINTS) {
    instanceCount = vtxCount;
  }
  // AURORA_LOG_UNTEX_REGS=1: for an untextured 4-vertex quad (the shape of
  // the white-HUD artifact), report the TEV colour registers as they stand
  // AT RECORD TIME. The FIFO is drained on a worker thread, so this state
  // must be read here and not at render time.
  {
    static const bool logRegs = [] {
      const char* v = std::getenv("AURORA_LOG_UNTEX_REGS");
      return v != nullptr && *v != '\0' && *v != '0';
    }();
    if (logRegs && vtxCount == 4 && !cache.shaderInfo.sampledTextures.any()) {
      static uint64_t n = 0;
      // Sample across the WHOLE run, not just the first draws: the opening
      // movie and title issue thousands of these before gameplay starts, so
      // a first-N sample describes the wrong scene entirely.
      ++n;
      static uint64_t n_add = 0;
      // Filter to the ARTIFACT CLASS (additive: dst=GX_BL_ONE) rather than
      // sampling by count. There are ~57000 untextured quads per run and only
      // the additive ones are the white blobs, so a periodic sample almost
      // never lands on one -- which is why every field previously looked
      // faithful: it was describing the wrong draws.
      if (state.blendFacDst == GX_BL_ONE && n_add++ < 12) {
        const auto& c0 = state.colorRegs[0];
        const auto& c1 = state.colorRegs[1];
        const auto& c2 = state.colorRegs[2];
        fmt::print(stderr,
                   "untex quad #{}: stages={} c0=({:.3f},{:.3f},{:.3f},{:.3f}) "
                   "c1=({:.3f},{:.3f},{:.3f},{:.3f}) c2=({:.3f},{:.3f},{:.3f},{:.3f}) "
                   "s0.color a={} b={} c={} d={} s0.alpha a={} b={} c={} d={} "
                   "s0.texMap={} s0.chan={} | chan0 lit={} matSrc={} ambSrc={} "
                   "mat=({:.3f},{:.3f},{:.3f},{:.3f}) amb=({:.3f},{:.3f},{:.3f},{:.3f}) "
                   "vtxClr0={} clr0fmt cnt={} type={} desc={} | blend={} src={} dst={} op={} alphaUpd={} numChans={} | posDesc={} posCnt={} posType={} arrStride={}\n",
                   n, u32{cache.config.shaderConfig.tevStageCount},
                   c0[0], c0[1], c0[2], c0[3], c1[0], c1[1], c1[2], c1[3],
                   c2[0], c2[1], c2[2], c2[3],
                   underlying(cache.config.shaderConfig.tevStages[0].colorPass.a),
                   underlying(cache.config.shaderConfig.tevStages[0].colorPass.b),
                   underlying(cache.config.shaderConfig.tevStages[0].colorPass.c),
                   underlying(cache.config.shaderConfig.tevStages[0].colorPass.d),
                   underlying(cache.config.shaderConfig.tevStages[0].alphaPass.a),
                   underlying(cache.config.shaderConfig.tevStages[0].alphaPass.b),
                   underlying(cache.config.shaderConfig.tevStages[0].alphaPass.c),
                   underlying(cache.config.shaderConfig.tevStages[0].alphaPass.d),
                   underlying(cache.config.shaderConfig.tevStages[0].texMapId),
                   underlying(cache.config.shaderConfig.tevStages[0].channelId),
                   state.colorChannelConfig[0].lightingEnabled,
                   underlying(state.colorChannelConfig[0].matSrc),
                   underlying(state.colorChannelConfig[0].ambSrc),
                   state.colorChannelState[0].matColor[0], state.colorChannelState[0].matColor[1],
                   state.colorChannelState[0].matColor[2], state.colorChannelState[0].matColor[3],
                   state.colorChannelState[0].ambColor[0], state.colorChannelState[0].ambColor[1],
                   state.colorChannelState[0].ambColor[2], state.colorChannelState[0].ambColor[3],
                   state.vtxDesc[GX_VA_CLR0] != GX_NONE,
                   underlying(state.vtxFmts[fmt].attrs[GX_VA_CLR0].cnt),
                   underlying(state.vtxFmts[fmt].attrs[GX_VA_CLR0].type),
                   underlying(state.vtxDesc[GX_VA_CLR0]),
                   underlying(state.blendMode), underlying(state.blendFacSrc),
                   underlying(state.blendFacDst), underlying(state.blendOp),
                   state.alphaUpdate, state.numChans,
                   underlying(state.vtxDesc[GX_VA_POS]),
                   underlying(state.vtxFmts[fmt].attrs[GX_VA_POS].cnt),
                   underlying(state.vtxFmts[fmt].attrs[GX_VA_POS].type),
                   state.arrays[GX_VA_POS].stride);
      }
    }
  }

  cache.lastDrawFmt = fmt;
  const auto& sc = cache.config.shaderConfig;
#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
  // Before the decoded positions: a draw merging into this one needs them
  // to end the storage pool.
  xr_motion_why(vtxCount);
  const XrMotionKind xrMotionKind = xr_motion_kind();
  const u32 xrPrevMtx = xrMotionKind != XrMotionKind::None ? xr_prev_matrices(raw, vertRange.size, fmt) : 0;
#endif
  const gfx::Range posRange =
      sc.decodedPos || sc.decodedAll
          // Aligned to its stride: XR batching indexes it from the pool's start.
          ? push_decoded_vertices(sc, raw, vtxCount, gfx::detail::xr_batching() ? decode_layout_for(sc).stride : 4)
          : gfx::Range{};
  gfx::PipelineRef xrPipeline = 0;
  gfx::Range xrIdxRange{};
  u32 xrIndexCount = 0;
  gfx::PipelineRef xrSolidPipeline = 0;
  gfx::Range xrSolidIdxRange{};
  u32 xrSolidIndexCount = 0;
  gfx::PipelineRef xrMotionPipeline = 0;
  bool cls_inside = false; // the draw lies wholly inside any clip (batchable)
#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
  // Clipped stage parts: only draws that straddle a clip use the dithering
  // variant; the rest keep early depth (no discard) or are dropped.
  {
    static std::vector<u8> outside, unsolid;
    outside.resize(vtxCount);
    unsolid.resize(vtxCount);
    XrClipClass cls;
    bool masked;
    xrPipeline = xr_world_pipeline(raw, vtxCount, &cls, outside.data(), &masked, unsolid.data());
    if (gfx::xr_recording_world()) {
      xr_room_census(vtxCount);
      xr_room_emitter(raw, vtxCount);
    }
    cls_inside = cls == XrClipClass::Inside;
    if (xrPipeline != 0 && cls != XrClipClass::Outside &&
        (xrMotionPipeline = xr_motion_pipeline(xrMotionKind, cls == XrClipClass::Straddle)) != 0)
      immediates.xrPrevMtx = xrPrevMtx;
    // A straddling draw's triangles wholly past one plane are left out of
    // the eyes (their own index list; the flat frame keeps every one). On
    // the Quest the dithering pipeline's discard costs early depth for the
    // whole draw, so a long strip mostly outside the clip (Brinstar's acid
    // runs the length of the cave) cost ~5 ms of eye pass.
    if (xrPipeline != 0 && cls == XrClipClass::Straddle && masked &&
        (prim == GX_TRIANGLES || prim == GX_TRIANGLESTRIP || prim == GX_TRIANGLEFAN || prim == GX_QUADS)) {
      static ByteBuffer all;
      static std::vector<u16> kept, solid;
      const u32 n = prepare_idx_buffer(all, prim, 0, vtxCount);
      const u16* tri = reinterpret_cast<const u16*>(all.data());
      // Triangles wholly inside the solid region (every vertex inside each
      // plane's fade; the region is convex) need no discard: drawn with the
      // plain pipeline, they keep early depth. Brinstar's acid had every
      // one of its draws straddle the box, so all of it discarded.
      static const bool split = [] {
        const char* v = std::getenv("AURORA_XR_CLIP_SPLIT");
        return v == nullptr || *v != '0';
      }();
      const auto* clip = gfx::xr_clip_camera();
      const bool splitting = split && clip != nullptr && clip->opacity >= 1.f;
      kept.clear();
      solid.clear();
      for (u32 i = 0; i + 2 < n; i += 3) {
        if ((outside[tri[i]] & outside[tri[i + 1]] & outside[tri[i + 2]]) != 0)
          continue;
        if (splitting && (unsolid[tri[i]] | unsolid[tri[i + 1]] | unsolid[tri[i + 2]]) == 0)
          solid.insert(solid.end(), {tri[i], tri[i + 1], tri[i + 2]});
        else
          kept.insert(kept.end(), {tri[i], tri[i + 1], tri[i + 2]});
      }
      all.clear();
      if (!solid.empty() && (xrSolidPipeline = xr_solid_pipeline()) != 0) {
        xrSolidIdxRange = gfx::push_indices(reinterpret_cast<const u8*>(solid.data()), solid.size() * sizeof(u16), 4);
        xrSolidIndexCount = static_cast<u32>(solid.size());
      } else {
        kept.insert(kept.end(), solid.begin(), solid.end());
      }
      if (kept.empty() && xrSolidIndexCount != 0) {
        // All of it solid: one plain draw.
        xrPipeline = xrSolidPipeline;
        xrIdxRange = xrSolidIdxRange;
        xrIndexCount = xrSolidIndexCount;
        xrSolidPipeline = 0;
        xrSolidIndexCount = 0;
      } else if (kept.empty()) {
        xrPipeline = 0;
      } else if (kept.size() < n) {
        xrIdxRange = gfx::push_indices(reinterpret_cast<const u8*>(kept.data()), kept.size() * sizeof(u16), 4);
        xrIndexCount = static_cast<u32>(kept.size());
      }
    }
  }
#endif
#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
  // AURORA_XR_BATCH_STATS=1: how many eye draws would merge with the one
  // before (same eye pipeline, textures, fog, uniforms but the matrices),
  // and what breaks the rest.
  {
    static const bool stats = std::getenv("AURORA_XR_BATCH_STATS") != nullptr;
    if (stats && xrPipeline != 0) {
      struct Key {
        gfx::PipelineRef pipeline = 0;
        uint64_t uniform = 0;
        GXBindGroups textures{};
        u32 dstAlpha = 0, fog = 0;
        bool eligible = false;
      };
      static Key last;
      static u32 draws = 0, batches = 0, frames = 0, seq = 0;
      static std::array<u32, 5> breaks{}; // ineligible, pipeline, textures, uniform, fog/alpha
      // Reordering opaque draws: distinct materials among them per frame.
      static std::unordered_map<uint64_t, u32> opaqueKeys;
      static u32 opaque = 0, opaqueGroups = 0, ordered = 0;
      if (const u32 s = static_cast<u32>(gfx::detail::recording_seq()); s != seq) {
        seq = s;
        last = {};
        opaqueGroups += static_cast<u32>(opaqueKeys.size());
        opaqueKeys.clear();
        if (++frames % 600 == 0) {
          Log.info("XR batching: {} eye draws would go as {} batches ({:.0f}%); breaks: ineligible {}, pipeline {}, "
                   "textures {}, uniforms {}, fog/alpha {}",
                   draws, batches, draws ? 100.0 * batches / draws : 0.0, breaks[0], breaks[1], breaks[2], breaks[3],
                   breaks[4]);
          Log.info("XR batching, opaque draws regrouped by material: {} opaque as {} groups ({:.0f}%), {} others in order",
                   opaque, opaqueGroups, opaque ? 100.0 * opaqueGroups / opaque : 0.0, ordered);
          draws = batches = 0;
          breaks = {};
          opaque = opaqueGroups = ordered = 0;
        }
      }
      const Key key{xrPipeline, cache.uniformSharedKey, cache.bindGroups, state.dstAlpha, immediates.fogRangeBase,
                    instanceCount == 1 && cache.config.shaderConfig.lineMode == 0 && xrSolidIndexCount == 0};
      ++draws;
      int why = -1;
      if (!key.eligible || !last.eligible)
        why = 0;
      else if (key.pipeline != last.pipeline)
        why = 1;
      else if (std::memcmp(&key.textures, &last.textures, sizeof(key.textures)) != 0)
        why = 2;
      else if (key.uniform != last.uniform)
        why = 3;
      else if (key.dstAlpha != last.dstAlpha || key.fog != last.fog)
        why = 4;
      if (why >= 0) {
        ++batches;
        ++breaks[why];
      }
      last = key;
      const auto& c = cache.config;
      const bool isOpaque = key.eligible && c.depthCompare && c.depthUpdate &&
                            (c.blendMode == GX_BM_NONE ||
                             (c.blendMode == GX_BM_BLEND && c.blendFacSrc == GX_BL_ONE && c.blendFacDst == GX_BL_ZERO));
      if (isOpaque) {
        ++opaque;
        uint64_t k = XXH3_64bits_withSeed(&key.pipeline, sizeof(key.pipeline), key.uniform);
        k = XXH3_64bits_withSeed(&key.textures, sizeof(key.textures), k);
        ++opaqueKeys[k];
      } else {
        ++ordered;
      }
    }
  }
#endif
  gfx::push_draw_command(DrawData{
      .pipeline = cache.pipelineRef,
      .xrPipeline = xrPipeline,
      .vertRange = vertRange,
      .idxRange = idxRange,
      .posRange = posRange,
      .uniformRange = cache.uniformRange,
      .immediateData = immediates,
#ifdef __EMSCRIPTEN__
      .immediateRange = gfx::push_uniform(immediates),
#endif
      .vtxCount = vtxCount,
      .indexCount = numIndices,
      .instanceCount = instanceCount,
      .bindGroups = cache.bindGroups,
      .dstAlpha = state.dstAlpha,
      .tag = sMarkerTag | (sLinkTag << 16),
      .xrIdxRange = xrIdxRange,
      .xrIndexCount = xrIndexCount,
      .xrSolidPipeline = xrSolidPipeline,
      .xrSolidIdxRange = xrSolidIdxRange,
      .xrSolidIndexCount = xrSolidIndexCount,
      .xrMotionPipeline = xrMotionPipeline,
#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
      .xrBatchPipeline = xrPipeline != 0 && xrSolidIndexCount == 0 && cls_inside ? xr_batch_pipeline(false) : 0,
      .xrMotionBatchPipeline =
          xrMotionPipeline != 0 && cls_inside ? xr_batch_pipeline(true, xrMotionKind) : 0,
      .xrMotionMaterial = xrMotionKind != XrMotionKind::Plain,
      .xrReorderable = [&] {
        const auto& c = cache.config;
        const bool opaque = c.blendMode == GX_BM_NONE ||
                            (c.blendMode == GX_BM_BLEND && c.blendFacSrc == GX_BL_ONE && c.blendFacDst == GX_BL_ZERO);
        return opaque && c.depthCompare && c.depthUpdate && c.colorUpdate && !c.shaderConfig.alphaCompare &&
               (c.depthFunc == GX_LESS || c.depthFunc == GX_LEQUAL) && c.polygonOffsetBits == 0 &&
               c.polygonOffsetScaleBits == 0 && c.shaderConfig.lineMode == 0;
      }(),
#else
      // xr_batch_pipeline() is XR-only (it needs the multiview/motion layouts),
      // so off XR the batch pipelines stay 0. The DrawData fields exist either
      // way and batching is disabled, so this is the same as before the XR
      // branch, not a behaviour change.
      .xrBatchPipeline = 0,
      .xrMotionBatchPipeline = 0,
#endif
      .uniformSharedKey = cache.uniformSharedKey,
  });
}

static void handle_draw_unmerged(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, gfx::Range vertRange,
                                 const u8* raw) noexcept {
  ZoneScoped;
  u32 numIndices = 0;
  gfx::Range idxRange;

  if (prim != GX_TRIANGLES) {
    ZoneScopedN("build idx buffer");
    static ByteBuffer idxBuf;
    numIndices = prepare_idx_buffer(idxBuf, prim, 0, vtxCount);
    idxRange = gfx::push_indices(idxBuf.data(), idxBuf.size(), 4);
    idxBuf.clear();
  }

  push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, numIndices, raw);
}

static void draw_prim(GXPrimitive prim, GXVtxFmt fmt, u16 vtxCount, ByteReader& reader) noexcept {
  ZoneScoped;
  sDrawPrim = prim;
  u32 vtxSize;
  if (g_gxState.lastVtxFmt == fmt)
    LIKELY { vtxSize = g_gxState.lastVtxSize; }
  else
    UNLIKELY { vtxSize = calc_vtx_size(fmt); }

  u32 totalVtxBytes = vtxCount * vtxSize;
  if (totalVtxBytes > reader.remaining())
    UNLIKELY { handle_draw_overrun(totalVtxBytes, reader); }

  const bool cleanState = g_gxState.dirty == 0 && fmt == sDrawCache.lastDrawFmt && sDrawCache.lineMode == 0 &&
                          prim != GX_LINES && prim != GX_LINESTRIP && prim != GX_POINTS;
  auto* lastDraw = cleanState ? gfx::get_last_draw_command<DrawData>() : nullptr;
  // A draw with decoded positions merges only while its range can grow in
  // place: nothing else may have been pushed to the storage pool since.
  bool canMerge = lastDraw != nullptr && lastDraw->instanceCount == 1 &&
                  (lastDraw->posRange.size == 0 ||
                   gfx::storage_tail() == size_t{lastDraw->posRange.offset} + lastDraw->posRange.size);

  // Push raw vertex data to buffer. Merged draws must remain contiguous with the previous range.
  const auto vertexData = reader.take(totalVtxBytes);
#if defined(AURORA_ENABLE_OPENXR) && !defined(__EMSCRIPTEN__)
  // Clipped world draws merge only with draws of their own class (inside,
  // straddling, outside), so a merged draw never needs the dithering
  // pipeline just because some of it straddles.
  // Straddling draws don't merge: each keeps its own eye index list. Except
  // under an opacity-only fade (off-screen fighters, Brinstar's rising
  // acid): every draw straddles there and none builds an index list (no
  // plane cuts a triangle). Unmerged, a faded fighter reached the eye and
  // motion passes as its hundreds of tiny GX draws: forcing all four
  // fighters to fade took the eyes from 4.2 to 16.2 ms on the Quest 3,
  // dithered, blended or drawn opaque alike.
  if (canMerge && gfx::xr_soft_clip_active()) {
    XrClipClass cls;
    canMerge = xr_world_pipeline(vertexData.data(), vtxCount, &cls) == lastDraw->xrPipeline &&
               (cls != XrClipClass::Straddle || xr_fade_opacity_only()) && lastDraw->xrIndexCount == 0;
  }
#endif
  // AURORA_LOG_QUADPOS=1: log the vertex POSITIONS of untextured 4-vertex
  // quads. Sampling these draws by count is useless -- there are tens of
  // thousands per run and only a handful are the white artifact -- so the
  // artifact has to be picked out by WHERE it is on screen.
  {
    static const bool logPos = [] {
      const char* v = std::getenv("AURORA_LOG_QUADPOS");
      return v != nullptr && *v != '\0' && *v != '0';
    }();
    if (logPos && vtxCount == 4 && g_gxState.vtxDesc[GX_VA_POS] == GX_DIRECT &&
        g_gxState.vtxFmts[fmt].attrs[GX_VA_POS].type == GX_F32) {
      const auto cnt = g_gxState.vtxFmts[fmt].attrs[GX_VA_POS].cnt;
      const u32 nComp = (cnt == GX_POS_XYZ) ? 3u : 2u;
      if (vtxSize >= nComp * 4) {
        float px[4], py[4];
        for (u32 v = 0; v < 4; ++v) {
          u32 off = v * vtxSize;
          // Vertex data is big-endian regardless of host.
          auto be32 = [&](u32 o) {
            return (u32(vertexData[o]) << 24) | (u32(vertexData[o + 1]) << 16) |
                   (u32(vertexData[o + 2]) << 8) | u32(vertexData[o + 3]);
          };
          u32 bx = be32(off), by = be32(off + 4);
          std::memcpy(&px[v], &bx, 4);
          std::memcpy(&py[v], &by, 4);
        }
        fmt::print(stderr, "quad blend={} src={} dst={} v=({:.1f},{:.1f}) "
                           "({:.1f},{:.1f}) ({:.1f},{:.1f}) ({:.1f},{:.1f})\n",
                   underlying(g_gxState.blendMode), underlying(g_gxState.blendFacSrc),
                   underlying(g_gxState.blendFacDst),
                   px[0], py[0], px[1], py[1], px[2], py[2], px[3], py[3]);
      }
    }
  }
  gfx::Range vertRange = gfx::push_verts(vertexData.data(), vertexData.size(), canMerge ? 0 : 4);

  // Try to merge with previous draw call
  if (canMerge) {
    u32 numIndices = 0;
    gfx::Range idxRange;
    static ByteBuffer idxBuf;
    const bool hadIndexRange = lastDraw->idxRange.size != 0;
    if (lastDraw->indexCount == 0 && prim != GX_TRIANGLES) {
      // Generate triangle index buffer for previous draw
      lastDraw->indexCount = prepare_idx_buffer(idxBuf, GX_TRIANGLES, 0, lastDraw->vtxCount);
    }
    if (lastDraw->indexCount != 0) {
      numIndices += prepare_idx_buffer(idxBuf, prim, lastDraw->vtxCount, vtxCount);
      idxRange = gfx::push_indices(idxBuf.data(), idxBuf.size(), hadIndexRange ? 0 : 4);
      idxBuf.clear();
    }
    CHECK(lastDraw->vertRange.offset + lastDraw->vertRange.size == vertRange.offset,
          "Non-consecutive vertex ranges ({} < {})", lastDraw->vertRange.offset + lastDraw->vertRange.size,
          vertRange.offset);
    if (hadIndexRange) {
      CHECK(lastDraw->idxRange.offset + lastDraw->idxRange.size == idxRange.offset,
            "Non-consecutive index ranges ({} < {})", lastDraw->idxRange.offset + lastDraw->idxRange.size,
            idxRange.offset);
    }
    lastDraw->vertRange.size += vertRange.size;
    if (lastDraw->posRange.size != 0) {
      lastDraw->posRange.size +=
          push_decoded_vertices(sDrawCache.config.shaderConfig, vertexData.data(), vtxCount, 0).size;
    }
    if (lastDraw->idxRange.size == 0) {
      lastDraw->idxRange = idxRange;
    } else {
      lastDraw->idxRange.size += idxRange.size;
    }
    lastDraw->vtxCount += vtxCount;
    lastDraw->indexCount += numIndices;
    gfx::detail::increment_merged_draw_count();
    return;
  }

  handle_draw_unmerged(prim, fmt, vtxCount, vertRange, vertexData.data());
}

static void handle_draw(u8 cmd, ByteReader& reader) noexcept {
  const auto fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
  const auto prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
  const u16 vtxCount = reader.read<u16>();
  auto& p = sFifoProbe;
  if (!p.on) {
    draw_prim(prim, fmt, vtxCount, reader);
    return;
  }
  const int64_t start = probe_now();
  draw_prim(prim, fmt, vtxCount, reader);
  const int64_t ns = probe_now() - start;
  p.drawNs += ns;
  ++p.draws;
  if (ns > p.slowestNs) {
    p.slowestNs = ns;
    p.slowestVerts = vtxCount;
    p.slowestCmd = cmd;
    p.slowestLink = sLinkTag;
  }
}

void handle_aurora(ByteReader& reader) noexcept {
  ZoneScoped;
  const u16 subCmd = reader.read<u16>();

  if (subCmd == GX_AURORA_LOAD_VIEWPORT_RENDER) {
    const f32 left = reader.read<f32>();
    const f32 top = reader.read<f32>();
    const f32 width = reader.read<f32>();
    const f32 height = reader.read<f32>();
    const f32 nearZ = reader.read<f32>();
    const f32 farZ = reader.read<f32>();
    set_render_viewport({
        .left = left,
        .top = top,
        .width = width,
        .height = height,
        .znear = nearZ,
        .zfar = farZ,
    });
  } else if (subCmd == GX_AURORA_LOAD_SCISSOR_RENDER) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    set_render_scissor({left, top, width, height});
  } else if (subCmd == GX_AURORA_LOAD_PROJECTION_FULL) {
    auto& proj = g_gxState.proj;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        proj[r][c] = reader.read<f32>();
      }
    }
    // Invalidate projection XF regs
    for (u32 reg = 0x20; reg <= 0x26; ++reg) {
      g_gxState.xfRegValid.reset(reg);
    }
    g_gxState.dirty |= DirtyUniform;
  } else if (subCmd >= GX_AURORA_LOAD_ARRAYBASE && subCmd <= (GX_AURORA_LOAD_ARRAYBASE | 0x0f)) {
    const u32 attrIdx = subCmd - GX_AURORA_LOAD_ARRAYBASE + GX_VA_POS;
    const u64 arrayAddr = reader.read<u64>();
    const u32 arraySize = reader.read<u32>();
    const bool le = reader.read<u8>() == 1;

    auto& array = g_gxState.arrays[attrIdx];
    const auto newData = reinterpret_cast<void*>(arrayAddr);
    if (array.data != newData || array.size != arraySize || array.le != le) {
      if (array.le != le) {
        // Endianness is baked into the shader
        g_gxState.dirty |= DirtyPipeline;
      }
      array.data = newData;
      array.size = arraySize;
      array.le = le;
      array.cachedRange = {};
      array.hashed = false;
      g_gxState.dirty |= DirtyImmediates;
    }
  } else if (subCmd == GX_AURORA_LOAD_TEXOBJ) {
    const auto texMapId = reader.read<u8>();
    CHECK(texMapId < MaxTextures, "invalid texture map id {}", texMapId);
    auto& slot = g_gxState.loadedTextures[texMapId];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const u32 newWidth = reader.read<u32>();
    const u32 newHeight = reader.read<u32>();
    const auto newFormat = static_cast<GXTexFmt>(reader.read<u32>());
    const auto newTlut = static_cast<GXTlut>(reader.read<u32>());
    u8 newFlags = slot.flags & ~0x80u; // Reset no-cache flag
    if (reader.read<u8>() != 0) {
      newFlags |= 1u;
    } else {
      newFlags &= ~1u;
    }
    const u32 newTexObjId = reader.read<u32>();
    const u32 newTexDataVersion = reader.read<u32>();
    if (slot.data != newData || slot.mWidth != newWidth || slot.mHeight != newHeight ||
        slot.mFormat != static_cast<u32>(newFormat) || slot.tlut != newTlut || slot.flags != newFlags ||
        slot.texObjId != newTexObjId || slot.texDataVersion != newTexDataVersion) {
      slot.data = newData;
      slot.mWidth = newWidth;
      slot.mHeight = newHeight;
      slot.mFormat = newFormat;
      slot.tlut = newTlut;
      slot.flags = newFlags;
      slot.texObjId = newTexObjId;
      slot.texDataVersion = newTexDataVersion;
      g_gxState.dirty |= DirtyTextures;
    }
  } else if (subCmd == GX_AURORA_LOAD_TLUT) {
    const auto idx = reader.read<u8>();
    CHECK(idx < MaxTluts, "invalid tlut slot {}", idx);
    auto& slot = g_gxState.loadedTluts[idx];
    const auto newData = reinterpret_cast<const void*>(reader.read<u64>());
    const auto newFormat = static_cast<GXTlutFmt>(reader.read<u32>());
    const u16 newNumEntries = reader.read<u16>();
    const u32 newTlutObjId = reader.read<u32>();
    const u32 newTlutDataVersion = reader.read<u32>();
    const u8 newFlags = slot.flags & ~0x80u; // Reset no-cache flag
    if (slot.data != newData || slot.format != newFormat || slot.numEntries != newNumEntries ||
        slot.tlutObjId != newTlutObjId || slot.tlutDataVersion != newTlutDataVersion || slot.flags != newFlags) {
      if (slot.tlutObjId != newTlutObjId || slot.tlutDataVersion != newTlutDataVersion) {
        texture::invalidate_bindings();
      }
      slot.data = newData;
      slot.format = newFormat;
      slot.numEntries = newNumEntries;
      slot.tlutObjId = newTlutObjId;
      slot.tlutDataVersion = newTlutDataVersion;
      slot.flags = newFlags;
      g_gxState.dirty |= DirtyTextures;
    }
  } else if (subCmd == GX2_SET_POLYGON_OFFSET) {
    const f32 frontOffset = reader.read<f32>();
    const f32 frontScale = reader.read<f32>();
    const f32 backOffset = reader.read<f32>();
    const f32 backScale = reader.read<f32>();
    const f32 clamp = reader.read<f32>();
    if (g_gxState.frontOffset != frontOffset || g_gxState.frontScale != frontScale ||
        g_gxState.backOffset != backOffset || g_gxState.backScale != backScale || g_gxState.clamp != clamp) {
      g_gxState.frontOffset = frontOffset;
      g_gxState.frontScale = frontScale;
      g_gxState.backOffset = backOffset;
      g_gxState.backScale = backScale;
      g_gxState.clamp = clamp;
      g_gxState.dirty |= DirtyPipeline;
    }
  } else if (subCmd == GX_AURORA_LOAD_COPY_SRC) {
    const s32 left = reader.read<s32>();
    const s32 top = reader.read<s32>();
    const s32 width = reader.read<s32>();
    const s32 height = reader.read<s32>();
    g_gxState.texCopySrc = {left, top, width, height};
  } else if (subCmd == GX_AURORA_LOAD_COPY_DST) {
    g_gxState.texCopyDstWidth = reader.read<u32>();
    g_gxState.texCopyDstHeight = reader.read<u32>();
    g_gxState.texCopyFmt = static_cast<GXTexFmt>(reader.read<u32>());
    reader.skip(1); // mipmap is not implemented, but remains part of the command payload
    g_gxState.texCopyDstWide = true;
  } else if (subCmd == GX_AURORA_LOAD_COPY_DEST) {
    g_gxState.texCopyDest = reinterpret_cast<const void*>(reader.read<u64>());
  } else if (subCmd == GX_AURORA_REQUEST_DEPTH_SNAPSHOT) {
    gfx::depth_peek::request_snapshot();
  } else if (subCmd == GX_AURORA_XR_CAMERA) {
    const u32 category = reader.read<u32>();
    const bool hasView = reader.read<u32>() != 0;
    std::array<float, 12> view{};
    for (auto& v : view) {
      v = std::bit_cast<float>(reader.read<u32>());
    }
    if (category == 3) { // aurora_xr_world_transform
      gfx::xr_set_world_transform(hasView ? view.data() : nullptr);
    } else if (category == 4) { // aurora_xr_world_clips4: planes 1-2 (clears the rest)
      gfx::xr_set_world_clip(hasView ? view.data() : nullptr);
    } else if (category == 5 || category == 7 || category == 8) { // planes 3-4, 5-6, 7-8
      gfx::xr_set_world_clip_more(category == 5 ? 1 : category - 5, view.data());
    } else if (category == 9) { // aurora_xr_world_still
      sXrStill = hasView;
    } else if (category == 10) { // aurora_xr_light_probe: the point in the view's first row
      gfx::xr_set_light_probe(hasView ? view.data() : nullptr);
    } else {
      const auto cat = category <= 2 ? static_cast<gfx::XrCategory>(category)
                       : category == 6 ? gfx::XrCategory::Hidden // AURORA_XR_HIDDEN
                                       : gfx::XrCategory::Mono;
      gfx::xr_set_category(cat, hasView ? view.data() : nullptr);
    }
  } else if (subCmd == GX_AURORA_BEGIN_OFFSCREEN) {
    const u32 width = reader.read<u32>();
    const u32 height = reader.read<u32>();
    gfx::begin_offscreen(width, height);
  } else if (subCmd == GX_AURORA_END_OFFSCREEN) {
    gfx::end_offscreen();
  } else if (subCmd == GX_AURORA_DESTROY_TEXOBJ) {
    evict_texture_object(reader.read<u32>());
  } else if (subCmd == GX_AURORA_DESTROY_TLUT) {
    evict_tlut_object(reader.read<u32>());
  } else if (subCmd == GX_AURORA_INVALIDATE_TEX) {
    invalidate_texture_hashes();
  } else if (subCmd == GX_AURORA_DESTROY_COPY_TEX) {
    evict_copy_texture(reinterpret_cast<const void*>(reader.read<u64>()));
  } else if (subCmd == GX_AURORA_DRAW_SIZED) {
    const u8 cmd = reader.read<u8>();
    const u32 byteLen = reader.read<u32>();
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    if (byteLen != 0) {
      u32 vtxSize;
      if (g_gxState.lastVtxFmt == fmt) {
        vtxSize = g_gxState.lastVtxSize;
      } else {
        vtxSize = calc_vtx_size(fmt);
      }
      AURORA_ASSERT(vtxSize != 0 && byteLen % vtxSize == 0,
                    "GX_AURORA_DRAW_SIZED: {} bytes is not a whole number of size-{} vertices", byteLen, vtxSize);
      u32 vtxCount = byteLen / vtxSize;
      AURORA_ASSERT(vtxCount <= 0xFFFF, "GX_AURORA_DRAW_SIZED: too many vertices ({})", vtxCount);
      draw_prim(prim, fmt, static_cast<u16>(vtxCount), reader);
    }
  } else if (subCmd == GX_AURORA_DRAW_INDEXED) {
    ZoneScopedN("DRAW_INDEXED");
    const u8 cmd = reader.read<u8>();
    const u16 vtxCount = reader.read<u16>();
    const u32 indexCount = reader.read<u32>();
    const GXVtxFmt fmt = static_cast<GXVtxFmt>(cmd & CP_VAT_MASK);
    const GXPrimitive prim = static_cast<GXPrimitive>(cmd & CP_OPCODE_MASK);
    AURORA_ASSERT(prim == GX_TRIANGLES, "GX_AURORA_DRAW_INDEXED: primitive must be GX_TRIANGLES, got {}",
                  static_cast<u32>(prim));
    const size_t idxBytes = static_cast<size_t>(indexCount) * sizeof(u16);
    // Index data is always host-endian; push it to the GPU buffer as-is
    const auto indexData = reader.take(idxBytes);
    const gfx::Range idxRange = gfx::push_indices(indexData.data(), indexData.size(), 4);
    u32 vtxSize;
    if (g_gxState.lastVtxFmt == fmt) {
      vtxSize = g_gxState.lastVtxSize;
    } else {
      vtxSize = calc_vtx_size(fmt);
    }
    const u32 totalVtxBytes = vtxCount * vtxSize;
    const auto vertexData = reader.take(totalVtxBytes);
    const gfx::Range vertRange = gfx::push_verts(vertexData.data(), vertexData.size(), 4);
    sDrawPrim = 0;
    if (indexCount != 0) {
      push_gx_draw(prim, fmt, vtxCount, vertRange, idxRange, indexCount, vertexData.data());
    }
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_PUSH) {
    auto label = reader.read_string();
    gfx::push_debug_group(std::move(label));
  } else if (subCmd == GX_AURORA_DEBUG_GROUP_POP) {
    pop_debug_group();
  } else if (subCmd == GX_AURORA_DEBUG_MARKER_INSERT) {
    auto label = reader.read_string();
    // A numeric marker is kept as a breadcrumb for the following draws. This
    // runs on the same worker that records draws, so ordering is exact.
    if (!label.empty() && label[0] == 'L')
      sLinkTag = static_cast<uint32_t>(std::strtoul(label.c_str() + 1, nullptr, 10)) + 1;
    else
      sMarkerTag = static_cast<uint32_t>(std::strtoul(label.c_str(), nullptr, 0));
    gfx::insert_debug_marker(std::move(label));
  }

  else {
    Log.error("Unknown Aurora subcommand: {:04X}", subCmd);
  }
}

void clear_draw_cache() noexcept {
  sDrawCache.bindGeneration = 0;
  sDrawCache.uniformRange = {};
  sDrawCache.fogRange = {};
  sDrawCache.hasFogRange = false;
}

} // namespace aurora::gx::fifo
