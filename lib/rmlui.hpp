#pragma once

#include "aurora/aurora.h"
#include "webgpu/gpu.hpp"

#include <SDL3/SDL_events.h>
#include <aurora/rmlui.hpp>
#include <dawn/webgpu_cpp.h>

namespace aurora::rmlui {

struct RecordedFrame {
  wgpu::BindGroup bindGroup;
  bool overlay = false;
};

void initialize(const AuroraWindowSize& size) noexcept;
void handle_event(SDL_Event& event) noexcept;
RecordedFrame record_frame(const webgpu::Viewport& presentViewport) noexcept;
// aurora_rmlui_draw_here (game thread, during the game's frame).
void draw_here() noexcept;
void shutdown() noexcept;

} // namespace aurora::rmlui
