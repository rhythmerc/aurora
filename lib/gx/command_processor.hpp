#pragma once

#include "../internal.hpp"

#include <cstdint>

namespace aurora::gx::fifo {

struct ProcessResult {
  uint32_t bytesProcessed;
  bool drawDone;
};

// Process GX FIFO commands until the next draw done event or end of buffer
ProcessResult process(const uint8_t* data, uint32_t size) noexcept;
void clear_draw_cache() noexcept;
// The frame being processed skips its draws (aurora_set_frame_discard).
// Set between frames, before any of the next frame's commands.
void set_discard_draws(bool discard) noexcept;

} // namespace aurora::gx::fifo
