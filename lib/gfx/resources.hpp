#pragma once

#include "types.hpp"

namespace aurora::gfx {
inline constexpr bool UseTextureBuffer = true;
/* These pools are non-owning views onto the mapped staging buffer, so
 * overflowing one is a hard abort, not a realloc. */
#if defined(__ANDROID__)
// melee-pc's Quest sizes, kept until Strikers is measured there
// (STRIKERS_LOG_GPUMEM). The XR eye replay and SpaceWarp's motion pass add
// uniforms and decoded vertices on top of the flat frame's: Melee's busiest
// frames reached 24 MiB of uniforms, 1 MiB of indices and 4 MiB of storage.
inline constexpr uint64_t UniformBufferSize = 25165824; // 24 MiB
inline constexpr uint64_t VertexBufferSize = 4194304;   // 4 MiB
inline constexpr uint64_t IndexBufferSize = 4194304;    // 4 MiB
inline constexpr uint64_t StorageBufferSize = 16777216; // 16 MiB
inline constexpr uint64_t TextureUploadSize = 12582912; // 12 MiB
#else
// smstrikers-port: sized from this game's measured peaks rather than upstream's
// defaults (uniform 24, storage 8 MiB). Measured worst case is 1.92, 1.95, 0.52,
// 8.10 and 19.45 MiB in the order below; overflow of the first four is a bare
// abort(), so each carries 2-4x margin. Storage is grown, not shrunk.
inline constexpr uint64_t UniformBufferSize = 8388608;   // 8 MiB
inline constexpr uint64_t VertexBufferSize = 5242880;    // 5 MiB
inline constexpr uint64_t IndexBufferSize = 2097152;     // 2 MiB
inline constexpr uint64_t StorageBufferSize = 16777216;  // 16 MiB
inline constexpr uint64_t TextureUploadSize = 25165824;  // 24 MiB
#endif

namespace detail {
// XR batching: per frame in flight, room for merged indices and records.
inline constexpr uint64_t XrBatchSlotSize = 2097152; // 2 MiB
inline constexpr uint64_t XrBatchSlots = 4;

struct Resources {
  wgpu::Buffer vertexBuffer;
  wgpu::Buffer uniformBuffer;
  wgpu::Buffer indexBuffer;
  wgpu::Buffer storageBuffer;
  // XR batching: merged indices, per-draw records and range tables, written
  // per frame (gx::xr_batch_*); bound as storage in group 0 with the uniform
  // pool, which batched shaders read draws' matrices from.
  wgpu::Buffer xrBatchBuffer;
  wgpu::BindGroupLayout staticBindGroupLayout;
  wgpu::BindGroup staticBindGroup;
  wgpu::BindGroupLayout uniformBindGroupLayout;
  wgpu::BindGroup uniformBindGroup;
  wgpu::Limits limits;
  AuroraStats stats{};
};

Resources& resources() noexcept;
} // namespace detail
} // namespace aurora::gfx
