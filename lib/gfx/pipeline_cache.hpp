#pragma once

#include "types.hpp"

namespace aurora::gfx::clear {
struct PipelineConfig;
} // namespace aurora::gfx::clear

namespace aurora::gx {
struct PipelineConfig;
} // namespace aurora::gx

namespace aurora::rmlui {
struct PipelineConfig;
} // namespace aurora::rmlui

namespace aurora::gfx {

enum class ShaderType : uint8_t {
  Clear = 0,
  GX = 1,
  Rml = 2,
};

void initialize_pipeline_cache();
void shutdown_pipeline_cache();
void begin_pipeline_frame();
void end_pipeline_frame();
void rebuild_pipeline_cache();

// `roomtone`: the draw's Roomtone variant (gx::roomtone_config), recorded for
// the other lighting's pipelines; 0xFF none.
PipelineRef find_pipeline(const gx::PipelineConfig& config, const RenderTargetLayout& layout,
                          uint8_t roomtone = 0xFF);
// The same record for a pipeline asked for some other way.
void note_roomtone_variant(const gx::PipelineConfig& config, const RenderTargetLayout& layout, uint8_t variant);
PipelineRef find_pipeline(const clear::PipelineConfig& config, const RenderTargetLayout& layout);
PipelineRef find_pipeline(const rmlui::PipelineConfig& config);

bool get_pipeline(PipelineRef ref, wgpu::RenderPipeline& pipeline);
/* Blocks up to maxWaitMs while queued pipelines compile; returns how many are
 * still pending (queued or mid-compile). */
uint32_t wait_pipelines(uint32_t maxWaitMs);
// AURORA_PRECOMPILE (on by default in the Quest launcher): pipelines left to
// build for the game process (0: done, or not precompiling), and stopping.
uint32_t precompile_pending();
void precompile_stop();
// Pipelines created so far; any thread. A count still climbing means shaders
// are compiling, which stalls the frame on the CPU (XR dynamic resolution
// leaves such frames out of its costs).
uint32_t pipelines_created() noexcept;

} // namespace aurora::gfx
