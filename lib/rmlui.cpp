#include "rmlui.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

#include <RmlUi/Core.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi_Backend.h>
#include <RmlUi_Platform_SDL.h>
#include <tracy/Tracy.hpp>

#include "window.hpp"
#include "internal.hpp"
#include "imgui.hpp"
#include "rmlui/FileInterface_SDL.h"
#include "rmlui/GlassFilter.hpp"
#include "rmlui/ImageEffects.hpp"
#include "rmlui/SystemInterface_Aurora.h"
#include "rmlui/WebGPURenderInterface.hpp"
#include "webgpu/gpu.hpp"

#include <aurora/gfx.hpp>
#include <mutex>
#include <unordered_map>

namespace aurora::rmlui {
Rml::Context* g_context = nullptr;
FileInterface_SDL* g_fileInterface = nullptr;

namespace {
Module Log("aurora::rmlui");
constexpr size_t MaxTrackedTouches = 16;

struct TrackedTouch {
  SDL_FingerID id = 0;
  Rml::Vector2f position;
  Rml::Vector2f rmlPosition;
  Rml::Vector2f startPosition;
  Rml::ObserverPtr<Rml::Element> target;
  bool active = false;
};

uint32_t s_pressedMouseButtons = 0;
std::array<TrackedTouch, MaxTrackedTouches> s_trackedTouches{};
float s_uiScale = 0.0f;
webgpu::TextureWithSampler s_renderTarget;
wgpu::BindGroup s_renderTargetCopyBindGroup;

// aurora_rmlui_draw_here: the documents drawn inside the game's frame (a
// custom draw at that point of its stream) from the copy bind group of the
// last render, under a lock as the render worker reads it.
std::mutex s_layerMutex;
wgpu::BindGroup s_layerBindGroup;
gfx::DrawTypeId s_layerDraw = gfx::InvalidDrawType;
std::atomic<bool> s_layerThisFrame{false};
std::unordered_map<uint64_t, wgpu::RenderPipeline> s_layerPipelines; // by the pass's layout (render worker)

wgpu::RenderPipeline layer_pipeline(const gfx::RenderTargetLayout& layout) {
  if (auto it = s_layerPipelines.find(layout.key); it != s_layerPipelines.end())
    return it->second;
  wgpu::ShaderSourceWGSL source{};
  source.code = R"""(
@group(0) @binding(0) var layer_sampler: sampler;
@group(0) @binding(1) var layer_texture: texture_2d<f32>;
struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
var<private> pos: array<vec2<f32>, 3> = array<vec2<f32>, 3>(vec2(-1.0, 1.0), vec2(-1.0, -3.0), vec2(3.0, 1.0));
var<private> uvs: array<vec2<f32>, 3> = array<vec2<f32>, 3>(vec2(0.0, 0.0), vec2(0.0, 2.0), vec2(2.0, 0.0));
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(pos[i], 0.0, 1.0);
    out.uv = uvs[i];
    return out;
}
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(layer_texture, layer_sampler, in.uv);
}
)""";
  const wgpu::ShaderModuleDescriptor moduleDesc{.nextInChain = &source, .label = "RmlUi in-frame layer"};
  auto module = webgpu::g_device.CreateShaderModule(&moduleDesc);
  const auto bgl = webgpu::copy_bind_group_layout();
  const wgpu::PipelineLayoutDescriptor layoutDesc{.bindGroupLayoutCount = 1, .bindGroupLayouts = &bgl};
  auto pipelineLayout = webgpu::g_device.CreatePipelineLayout(&layoutDesc);
  // Premultiplied alpha over the scene colour; the other attachments untouched.
  static const wgpu::BlendState blend{
      .color = {.operation = wgpu::BlendOperation::Add, .srcFactor = wgpu::BlendFactor::One,
                .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
      .alpha = {.operation = wgpu::BlendOperation::Add, .srcFactor = wgpu::BlendFactor::One,
                .dstFactor = wgpu::BlendFactor::OneMinusSrcAlpha},
  };
  std::array<wgpu::ColorTargetState, gfx::MaxColorAttachments> targets{};
  for (uint32_t i = 0; i < layout.colorAttachmentCount; ++i) {
    const bool scene = i == gfx::SceneColorAttachmentIndex;
    targets[i] = {.format = layout.colorAttachments[i].format, .blend = scene ? &blend : nullptr,
                  .writeMask = scene ? wgpu::ColorWriteMask::All : wgpu::ColorWriteMask::None};
  }
  const wgpu::FragmentState fragment{
      .module = module, .entryPoint = "fs_main", .targetCount = layout.colorAttachmentCount, .targets = targets.data()};
  wgpu::DepthStencilState depth{};
  const bool hasDepth = layout.depthStencilFormat != wgpu::TextureFormat::Undefined;
  if (hasDepth) {
    depth.format = layout.depthStencilFormat;
    depth.depthWriteEnabled = wgpu::OptionalBool::False;
    depth.depthCompare = wgpu::CompareFunction::Always;
  }
  const wgpu::RenderPipelineDescriptor desc{
      .label = "RmlUi in-frame layer",
      .layout = pipelineLayout,
      .vertex = {.module = module, .entryPoint = "vs_main"},
      .primitive = {.topology = wgpu::PrimitiveTopology::TriangleList},
      .depthStencil = hasDepth ? &depth : nullptr,
      .multisample = {.count = layout.sampleCount, .mask = UINT32_MAX},
      .fragment = &fragment,
  };
  auto pipeline = webgpu::g_device.CreateRenderPipeline(&desc);
  s_layerPipelines.emplace(layout.key, pipeline);
  return pipeline;
}

void layer_draw(const gfx::DrawContext& ctx, const wgpu::RenderPassEncoder& pass, const void*, size_t, void*) {
  const auto& layout = ctx.layout;
  if (layout.viewCount > 1 || layout.colorAttachmentCount == 0)
    return; // not into an XR eye pass
  wgpu::BindGroup bindGroup;
  {
    std::lock_guard lock{s_layerMutex};
    bindGroup = s_layerBindGroup;
  }
  if (!bindGroup)
    return;
  const auto& color = layout.colorAttachments[gfx::SceneColorAttachmentIndex];
  pass.SetPipeline(layer_pipeline(layout));
  pass.SetBindGroup(0, bindGroup);
  pass.SetViewport(0.f, 0.f, static_cast<float>(color.width), static_cast<float>(color.height), 0.f, 1.f);
  pass.SetScissorRect(0, 0, color.width, color.height);
  pass.Draw(3);
}

WebGPURenderInterface* get_render_interface() noexcept {
  return static_cast<WebGPURenderInterface*>(Backend::GetRenderInterface()); // NOLINT(*-pro-type-static-cast-downcast)
}

Rml::Vector2i dimensions_from_viewport(const gfx::Viewport& viewport) noexcept {
  return {
      std::max(1, static_cast<int>(std::lround(viewport.width))),
      std::max(1, static_cast<int>(std::lround(viewport.height))),
  };
}

Rml::Vector2i presentation_dimensions_from_window_size(const AuroraWindowSize& size) noexcept {
  const auto viewport =
      webgpu::calculate_present_viewport(size.native_fb_width, size.native_fb_height, size.fb_width, size.fb_height);
  return dimensions_from_viewport(viewport);
}

void sync_context_metrics(Rml::Vector2i dimensions) noexcept {
  if (g_context == nullptr || dimensions.x <= 0 || dimensions.y <= 0) {
    return;
  }
  if (g_context->GetDimensions() != dimensions) {
    g_context->SetDimensions(dimensions);
  }
  const float ratio = s_uiScale > 0.0f ? s_uiScale : window::get_window_size().scale;
  if (g_context->GetDensityIndependentPixelRatio() != ratio) {
    g_context->SetDensityIndependentPixelRatio(ratio);
  }
}

void ensure_render_target(Rml::Vector2i dimensions) noexcept {
  if (dimensions.x <= 0 || dimensions.y <= 0) {
    return;
  }
  const auto width = static_cast<uint32_t>(dimensions.x);
  const auto height = static_cast<uint32_t>(dimensions.y);
  if (s_renderTarget.view && s_renderTarget.size.width == width && s_renderTarget.size.height == height) {
    return;
  }
  s_renderTarget = webgpu::create_render_texture(width, height, false);
  s_renderTargetCopyBindGroup = webgpu::create_copy_bind_group(s_renderTarget);
  std::lock_guard lock{s_layerMutex};
  s_layerBindGroup = s_renderTargetCopyBindGroup;
}

bool element_has_visible_backdrop_filter(const Rml::Element* element) noexcept {
  if (element == nullptr || !element->IsVisible(true)) {
    return false;
  }

  const auto& computed = element->GetComputedValues();
  if (computed.opacity() > 0.f && computed.has_backdrop_filter()) {
    return true;
  }

  const int childCount = element->GetNumChildren();
  for (int childIndex = 0; childIndex < childCount; ++childIndex) {
    if (element_has_visible_backdrop_filter(element->GetChild(childIndex))) {
      return true;
    }
  }
  return false;
}

bool context_has_visible_backdrop_filter(Rml::Context* context) noexcept {
  if (context == nullptr) {
    return false;
  }
  return element_has_visible_backdrop_filter(context->GetRootElement());
}

struct MappedPoint {
  Rml::Vector2f position;
  bool valid = false;
  bool inside = false;
};

MappedPoint map_native_point_to_content(float nativeX, float nativeY) noexcept {
  if (g_context == nullptr) {
    return {};
  }

  const auto size = window::get_window_size();
  const Rml::Vector2i contentSize = g_context->GetDimensions();
  if (size.native_fb_width == 0 || size.native_fb_height == 0 || contentSize.x <= 0 || contentSize.y <= 0) {
    return {};
  }

  const auto viewport =
      webgpu::calculate_present_viewport(size.native_fb_width, size.native_fb_height,
                                         static_cast<uint32_t>(contentSize.x), static_cast<uint32_t>(contentSize.y));
  const float right = viewport.left + viewport.width;
  const float bottom = viewport.top + viewport.height;
  if (viewport.width <= 0.f || viewport.height <= 0.f) {
    return {};
  }

  return {
      .position =
          {
              (nativeX - viewport.left) * static_cast<float>(contentSize.x) / viewport.width,
              (nativeY - viewport.top) * static_cast<float>(contentSize.y) / viewport.height,
          },
      .valid = true,
      .inside = nativeX >= viewport.left && nativeY >= viewport.top && nativeX < right && nativeY < bottom,
  };
}

MappedPoint map_window_point_to_content(float windowX, float windowY) noexcept {
  const auto size = window::get_window_size();
  if (size.width == 0 || size.height == 0) {
    return {};
  }
  return map_native_point_to_content(
      windowX * static_cast<float>(size.native_fb_width) / static_cast<float>(size.width),
      windowY * static_cast<float>(size.native_fb_height) / static_cast<float>(size.height));
}

MappedPoint map_touch_point_to_content(const SDL_TouchFingerEvent& event) noexcept {
  const auto size = window::get_window_size();
  return map_native_point_to_content(event.x * static_cast<float>(size.native_fb_width),
                                     event.y * static_cast<float>(size.native_fb_height));
}

int rounded_content_coord(float value) noexcept { return static_cast<int>(std::floor(value)); }

bool mouse_button_tracked(uint8_t button) noexcept {
  return button < 32 && (s_pressedMouseButtons & (1u << button)) != 0;
}

void set_mouse_button_tracked(uint8_t button, bool tracked) noexcept {
  if (button >= 32) {
    return;
  }
  const uint32_t mask = 1u << button;
  if (tracked) {
    s_pressedMouseButtons |= mask;
  } else {
    s_pressedMouseButtons &= ~mask;
  }
}

TrackedTouch* find_tracked_touch(SDL_FingerID id) noexcept {
  for (auto& touch : s_trackedTouches) {
    if (touch.active && touch.id == id) {
      return &touch;
    }
  }
  return nullptr;
}

TrackedTouch* find_free_touch() noexcept {
  for (auto& touch : s_trackedTouches) {
    if (!touch.active) {
      return &touch;
    }
  }
  return nullptr;
}

Rml::TouchList touch_list(SDL_FingerID id, Rml::Vector2f position) {
  return {Rml::Touch{static_cast<Rml::TouchId>(id), position}};
}

void dispatch_touch_event(TrackedTouch& touch, const char* type, Rml::Vector2f position, bool inside,
                          Rml::Vector2f delta = {}) noexcept {
  if (touch.target == nullptr) {
    return;
  }

  Rml::Dictionary parameters;
  parameters["finger_id"] = touch.id;
  parameters["x"] = position.x;
  parameters["y"] = position.y;
  parameters["dx"] = delta.x;
  parameters["dy"] = delta.y;
  parameters["start_x"] = touch.startPosition.x;
  parameters["start_y"] = touch.startPosition.y;
  parameters["inside"] = inside;
  touch.target->DispatchEvent(type, parameters, true, true);
}

void handle_mouse_motion(const SDL_MouseMotionEvent& motion) noexcept {
  const MappedPoint mapped = map_window_point_to_content(motion.x, motion.y);
  if (!mapped.inside) {
    g_context->ProcessMouseLeave();
    return;
  }
  g_context->ProcessMouseMove(rounded_content_coord(mapped.position.x), rounded_content_coord(mapped.position.y),
                              RmlSDL::GetKeyModifierState());
}

void handle_mouse_button_down(const SDL_MouseButtonEvent& button) noexcept {
  const MappedPoint mapped = map_window_point_to_content(button.x, button.y);
  if (!mapped.inside) {
    g_context->ProcessMouseLeave();
    return;
  }
  g_context->ProcessMouseMove(rounded_content_coord(mapped.position.x), rounded_content_coord(mapped.position.y),
                              RmlSDL::GetKeyModifierState());
  g_context->ProcessMouseButtonDown(RmlSDL::ConvertMouseButton(button.button), RmlSDL::GetKeyModifierState());
  set_mouse_button_tracked(button.button, true);
  SDL_CaptureMouse(true);
}

void handle_mouse_button_up(const SDL_MouseButtonEvent& button) noexcept {
  if (!mouse_button_tracked(button.button)) {
    return;
  }
  const auto mapped = map_window_point_to_content(button.x, button.y);
  if (mapped.inside) {
    g_context->ProcessMouseMove(rounded_content_coord(mapped.position.x), rounded_content_coord(mapped.position.y),
                                RmlSDL::GetKeyModifierState());
  } else {
    g_context->ProcessMouseLeave();
  }
  g_context->ProcessMouseButtonUp(RmlSDL::ConvertMouseButton(button.button), RmlSDL::GetKeyModifierState());
  set_mouse_button_tracked(button.button, false);
  if (s_pressedMouseButtons == 0) {
    SDL_CaptureMouse(false);
  }
}

void handle_mouse_wheel(const SDL_MouseWheelEvent& wheel) noexcept {
  float mouseX = 0.f;
  float mouseY = 0.f;
  SDL_GetMouseState(&mouseX, &mouseY);
  if (!map_window_point_to_content(mouseX, mouseY).inside) {
    g_context->ProcessMouseLeave();
    return;
  }
  g_context->ProcessMouseWheel(Rml::Vector2f{wheel.x, -wheel.y}, RmlSDL::GetKeyModifierState());
}

void handle_touch_down(const SDL_TouchFingerEvent& finger) noexcept {
  if (find_tracked_touch(finger.fingerID) != nullptr) {
    return;
  }
  const auto mapped = map_touch_point_to_content(finger);
  if (!mapped.inside) {
    return;
  }
  auto* tracked = find_free_touch();
  if (tracked == nullptr) {
    return;
  }
  auto* target = g_context->GetElementAtPoint(mapped.position);
  if (target == nullptr) {
    target = g_context->GetRootElement();
  }
  *tracked = {
      .id = finger.fingerID,
      .position = mapped.position,
      .rmlPosition = mapped.position,
      .startPosition = mapped.position,
      .target = target->GetObserverPtr(),
      .active = true,
  };
  dispatch_touch_event(*tracked, TouchStartEvent, mapped.position, true);
  g_context->ProcessTouchStart(touch_list(finger.fingerID, mapped.position), RmlSDL::GetKeyModifierState());
  g_context->ProcessMouseMove(rounded_content_coord(mapped.position.x), rounded_content_coord(mapped.position.y),
                              RmlSDL::GetKeyModifierState());
  g_context->ProcessMouseButtonDown(RmlSDL::ConvertMouseButton(SDL_BUTTON_LEFT), RmlSDL::GetKeyModifierState());
}

void handle_touch_motion(const SDL_TouchFingerEvent& finger) noexcept {
  auto* tracked = find_tracked_touch(finger.fingerID);
  if (tracked == nullptr) {
    return;
  }
  const auto mapped = map_touch_point_to_content(finger);
  if (!mapped.valid) {
    return;
  }
  const Rml::Vector2f delta = mapped.position - tracked->position;
  tracked->position = mapped.position;
  dispatch_touch_event(*tracked, TouchMoveEvent, mapped.position, mapped.inside, delta);
  if (mapped.inside) {
    tracked->rmlPosition = mapped.position;
    g_context->ProcessTouchMove(touch_list(finger.fingerID, mapped.position), RmlSDL::GetKeyModifierState());
    g_context->ProcessMouseMove(rounded_content_coord(mapped.position.x), rounded_content_coord(mapped.position.y),
                                RmlSDL::GetKeyModifierState());
  }
}

void handle_touch_up(const SDL_TouchFingerEvent& finger) noexcept {
  auto* tracked = find_tracked_touch(finger.fingerID);
  if (tracked == nullptr) {
    return;
  }
  const auto mapped = map_touch_point_to_content(finger);
  const Rml::Vector2f position = mapped.valid ? mapped.position : tracked->position;
  const Rml::Vector2f delta = position - tracked->position;
  dispatch_touch_event(*tracked, TouchEndEvent, position, mapped.valid && mapped.inside, delta);
  const auto rmlPosition = tracked->rmlPosition;
  *tracked = {};
  g_context->ProcessTouchEnd(touch_list(finger.fingerID, rmlPosition), RmlSDL::GetKeyModifierState());
  g_context->ProcessMouseButtonUp(RmlSDL::ConvertMouseButton(SDL_BUTTON_LEFT), RmlSDL::GetKeyModifierState());
}

void handle_touch_cancel(const SDL_TouchFingerEvent& finger) noexcept {
  auto* tracked = find_tracked_touch(finger.fingerID);
  if (tracked == nullptr) {
    return;
  }
  dispatch_touch_event(*tracked, TouchCancelEvent, tracked->position, false);
  const auto rmlPosition = tracked->rmlPosition;
  *tracked = {};
  g_context->ProcessTouchCancel(touch_list(finger.fingerID, rmlPosition));
  g_context->ProcessMouseButtonUp(RmlSDL::ConvertMouseButton(SDL_BUTTON_LEFT), RmlSDL::GetKeyModifierState());
}
} // namespace

void initialize(const AuroraWindowSize& size) noexcept {
  const Rml::Vector2i dim = presentation_dimensions_from_window_size(size);
  if (!Backend::Initialize("Aurora RmlUi Backend", dim.x, dim.y, false)) {
    Log.error("Failed to initialize RmlUI Backend!");
    return;
  }

  g_fileInterface = new FileInterface_SDL();

  auto* renderInterface = get_render_interface();
  Rml::SetSystemInterface(Backend::GetSystemInterface());
  Rml::SetRenderInterface(renderInterface);
  Rml::SetFileInterface(g_fileInterface);

  renderInterface->SetWindowSize(dim);
  renderInterface->SetRenderTargetFormat(webgpu::g_graphicsConfig.surfaceConfiguration.format);
  renderInterface->CreateDeviceObjects();

  Rml::Initialise();
  register_image_effects();

  static GlassFilterInstancer s_glassInstancer;
  Rml::Factory::RegisterFilterInstancer("glass", &s_glassInstancer);

  g_context = Rml::CreateContext("main", dim);

  if (g_context) {
    sync_context_metrics(dim);
  } else {
    Log.error("Failed to initialize RmlUI Context!");
    Rml::Shutdown();
    Backend::Shutdown();
  }
}

Rml::Context* get_context() noexcept { return g_context; }

bool is_initialized() noexcept { return g_context != nullptr; }

void set_ui_scale(float scale) noexcept { s_uiScale = scale > 0.0f ? std::clamp(scale, 0.25f, 4.0f) : 0.0f; }

float get_ui_scale() noexcept { return s_uiScale; }

void set_glass_light_dir(float x, float y) noexcept {
  if (const float len = std::hypot(x, y); len > 1e-4f) {
    g_glassLightDir = {x / len, y / len};
  }
}

void set_input_type(InputType type) noexcept {
  auto* systemInterface = static_cast<SystemInterface_Aurora*>(Backend::GetSystemInterface());
  if (systemInterface != nullptr) {
    systemInterface->SetInputType(type);
  }
}

void handle_event(SDL_Event& event) noexcept {
  if (g_context == nullptr || imgui::wants_capture_event(event)) {
    return;
  }

  switch (event.type) {
  case SDL_EVENT_MOUSE_MOTION:
    handle_mouse_motion(event.motion);
    return;
  case SDL_EVENT_MOUSE_BUTTON_DOWN:
    handle_mouse_button_down(event.button);
    return;
  case SDL_EVENT_MOUSE_BUTTON_UP:
    handle_mouse_button_up(event.button);
    return;
  case SDL_EVENT_MOUSE_WHEEL:
    handle_mouse_wheel(event.wheel);
    return;
  case SDL_EVENT_FINGER_DOWN:
    handle_touch_down(event.tfinger);
    return;
  case SDL_EVENT_FINGER_MOTION:
    handle_touch_motion(event.tfinger);
    return;
  case SDL_EVENT_FINGER_UP:
    handle_touch_up(event.tfinger);
    return;
  case SDL_EVENT_FINGER_CANCELED:
    handle_touch_cancel(event.tfinger);
    return;
  case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
    sync_context_metrics(presentation_dimensions_from_window_size(window::get_window_size()));
    return;
  case SDL_EVENT_WINDOW_MOUSE_LEAVE:
    g_context->ProcessMouseLeave();
    return;
  case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
    sync_context_metrics(presentation_dimensions_from_window_size(window::get_window_size()));
    return;
  default:
    break;
  }

  RmlSDL::InputEventHandler(g_context, window::get_sdl_window(), event);
}

RecordedFrame record_frame(const webgpu::Viewport& presentViewport) noexcept {
  if (g_context == nullptr) {
    return {};
  }

  ZoneScoped;
  const Rml::Vector2i dim = dimensions_from_viewport(presentViewport);
  ensure_render_target(dim);
  if (!s_renderTarget.view) {
    return {};
  }

  sync_context_metrics(dim);
  g_context->Update();
  const bool needsBackdrop = context_has_visible_backdrop_filter(g_context);

  auto* renderInterface = get_render_interface();
  renderInterface->SetWindowSize(g_context->GetDimensions());
  renderInterface->BeginFrame(s_renderTarget, webgpu::present_source(),
                              needsBackdrop ? BaseLayerContent::Scene : BaseLayerContent::Transparent);

  Backend::BeginFrame();
  g_context->Render();
  Backend::PresentFrame();

  const bool drawn = renderInterface->EndFrame();
  {
    // What the next aurora_rmlui_draw_here draws: this render, or nothing.
    std::lock_guard lock{s_layerMutex};
    s_layerBindGroup = drawn && !needsBackdrop ? s_renderTargetCopyBindGroup : wgpu::BindGroup{};
  }
  if (!drawn || s_layerThisFrame.exchange(false)) {
    // Nothing rendered, or the game drew the documents inside its frame
    // already (aurora_rmlui_draw_here): no composite at present.
    return {};
  }
  return {
      .bindGroup = s_renderTargetCopyBindGroup,
      .overlay = !needsBackdrop,
  };
}

void draw_here() noexcept {
  if (g_context == nullptr)
    return;
  if (s_layerDraw == gfx::InvalidDrawType)
    s_layerDraw = gfx::register_draw_type({.label = "RmlUi in-frame layer", .draw = layer_draw});
  if (gfx::push_custom_draw(s_layerDraw, nullptr, 0))
    s_layerThisFrame = true;
}

void shutdown() noexcept {
  if (g_context == nullptr) {
    return;
  }

  s_trackedTouches = {};
  Rml::Shutdown();
  Backend::Shutdown();
  g_context = nullptr;
  s_renderTarget = {};
  s_renderTargetCopyBindGroup = {};
  {
    std::lock_guard lock{s_layerMutex};
    s_layerBindGroup = {};
  }
  s_layerPipelines.clear();
}
} // namespace aurora::rmlui
