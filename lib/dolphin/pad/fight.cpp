#include "fight.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <mutex>

namespace aurora::pad {
namespace {

constexpr PADFightOutput kNone{0, PAD_FIGHT_STICK_NONE};
constexpr PADFightOutput kMain{0, PAD_FIGHT_STICK_MAIN};
constexpr PADFightOutput kC{0, PAD_FIGHT_STICK_C};
constexpr PADFightOutput button(u16 b) { return {b, PAD_FIGHT_STICK_NONE}; }

// The usual mappings: g_defaultButtonsStandard's (pad.cpp) and update_input's
// (lib/xr/xr.cpp). Left bumper and the stick clicks give nothing there.
constexpr std::array<PADFightOutput, PAD_FIGHT_PAD_COUNT> kStandardGamepad{{
    kMain,
    kC,
    kNone,
    kNone,
    button(PAD_TRIGGER_L),
    button(PAD_TRIGGER_R),
    button(PAD_BUTTON_A),
    button(PAD_BUTTON_B),
    button(PAD_BUTTON_X),
    button(PAD_BUTTON_Y),
    kNone,
    button(PAD_TRIGGER_Z),
    button(PAD_BUTTON_UP),
    button(PAD_BUTTON_DOWN),
    button(PAD_BUTTON_LEFT),
    button(PAD_BUTTON_RIGHT),
}};
constexpr std::array<PADFightOutput, PAD_FIGHT_TOUCH_COUNT> kStandardTouch{{
    kMain,
    kC,
    kNone,
    kNone,
    button(PAD_TRIGGER_L),
    button(PAD_TRIGGER_R),
    button(PAD_BUTTON_A),
    button(PAD_BUTTON_B),
    button(PAD_BUTTON_X),
    button(PAD_BUTTON_Y),
    button(PAD_TRIGGER_Z),
    button(PAD_TRIGGER_Z),
}};

constexpr u32 input_count(PADFightDevice device) {
  return device == PAD_FIGHT_DEVICE_TOUCH ? PAD_FIGHT_TOUCH_COUNT : PAD_FIGHT_PAD_COUNT;
}

std::mutex g_mutex;
std::array<std::array<PADFightOutput, PAD_FIGHT_INPUT_MAX>, PAD_FIGHT_DEVICE_COUNT> g_tables = [] {
  std::array<std::array<PADFightOutput, PAD_FIGHT_INPUT_MAX>, PAD_FIGHT_DEVICE_COUNT> t{};
  std::copy(kStandardGamepad.begin(), kStandardGamepad.end(), t[PAD_FIGHT_DEVICE_GAMEPAD].begin());
  std::copy(kStandardTouch.begin(), kStandardTouch.end(), t[PAD_FIGHT_DEVICE_TOUCH].begin());
  return t;
}();
std::atomic<bool> g_active{false};
// Bumped each time the mapping turns on or off: readers then take what's
// held as suppressed.
std::atomic<uint32_t> g_gen{0};

bool same(const PADFightOutput& a, const PADFightOutput& b) { return a.button == b.button && a.stick == b.stick; }

// What the raw inputs hold, a bit per input: a stick out of its middle, a
// trigger pressed in at all.
uint32_t held(const PADFightRaw& raw) {
  uint32_t h = raw.pressed;
  for (int s = 0; s < 2; ++s)
    if (raw.stick[s][0] * raw.stick[s][0] + raw.stick[s][1] * raw.stick[s][1] > 0.09f)
      h |= 1u << (PAD_FIGHT_LEFT_STICK + s);
  for (int t = 0; t < 2; ++t)
    if (raw.analog[t] > 32)
      h |= 1u << (PAD_FIGHT_LEFT_TRIGGER + t);
  return h;
}

} // namespace

const PADFightOutput* fight_standard(PADFightDevice device) {
  return device == PAD_FIGHT_DEVICE_TOUCH ? kStandardTouch.data() : kStandardGamepad.data();
}

void fight_filter(PADFightDevice device, FightReader& reader, PADFightRaw& raw) {
  const uint32_t h = held(raw);
  const uint32_t gen = g_gen.load(std::memory_order_acquire);
  if (reader.gen != gen) {
    reader.gen = gen;
    uint32_t changed = 0;
    {
      std::lock_guard lock{g_mutex};
      const PADFightOutput* standard = fight_standard(device);
      for (u32 i = 0; i < input_count(device); ++i)
        if (!same(g_tables[device][i], standard[i]))
          changed |= 1u << i;
    }
    reader.suppressed = h & changed;
  }
  reader.suppressed &= h;
  if (reader.suppressed == 0)
    return;
  raw.pressed &= ~reader.suppressed;
  for (int s = 0; s < 2; ++s)
    if (reader.suppressed & (1u << (PAD_FIGHT_LEFT_STICK + s)))
      raw.stick[s][0] = raw.stick[s][1] = 0.f;
  for (int t = 0; t < 2; ++t)
    if (reader.suppressed & (1u << (PAD_FIGHT_LEFT_TRIGGER + t)))
      raw.analog[t] = 0;
}

void fight_compose(PADFightDevice device, const PADFightRaw& raw, const PADFightOutput* table, float mainScale,
                   float cScale, uint8_t digitalAnalog, PADStatus& out) {
  std::array<PADFightOutput, PAD_FIGHT_INPUT_MAX> t{};
  if (table != nullptr) {
    std::copy(table, table + input_count(device), t.begin());
  } else {
    std::lock_guard lock{g_mutex};
    t = g_tables[device];
  }
  out.button = raw.start ? PAD_BUTTON_START : 0;
  out.stickX = out.stickY = out.substickX = out.substickY = 0;
  out.triggerLeft = out.triggerRight = 0;
  // Several sticks on one GameCube stick: the one tilted furthest.
  float best[2] = {-1.f, -1.f};
  for (int s = 0; s < 2; ++s) {
    const PADFightOutput& o = t[PAD_FIGHT_LEFT_STICK + s];
    if (o.stick == PAD_FIGHT_STICK_NONE)
      continue;
    const int k = o.stick == PAD_FIGHT_STICK_MAIN ? 0 : 1;
    const float x = raw.stick[s][0], y = raw.stick[s][1];
    const float m = x * x + y * y; // no libm here: the game's own interposes some
    if (m <= best[k])
      continue;
    best[k] = m;
    const float scale = k == 0 ? mainScale : cScale;
    const auto axis = [scale](float v) {
      return static_cast<s8>(std::lround(std::clamp(v, -1.f, 1.f) * scale));
    };
    if (k == 0)
      out.stickX = axis(x), out.stickY = axis(y);
    else
      out.substickX = axis(x), out.substickY = axis(y);
  }
  for (u32 i = PAD_FIGHT_LEFT_STICK_CLICK; i < input_count(device); ++i) {
    const u16 b = t[i].button;
    if (b == 0)
      continue;
    // A trigger carries its analog value (light shield) into L or R.
    if (i == PAD_FIGHT_LEFT_TRIGGER || i == PAD_FIGHT_RIGHT_TRIGGER) {
      const u8 a = raw.analog[i - PAD_FIGHT_LEFT_TRIGGER];
      if (b & PAD_TRIGGER_L)
        out.triggerLeft = std::max(out.triggerLeft, a);
      if (b & PAD_TRIGGER_R)
        out.triggerRight = std::max(out.triggerRight, a);
    }
    if ((raw.pressed & (1u << i)) == 0)
      continue;
    out.button |= b;
  }
  if (digitalAnalog != 0) {
    if (out.button & PAD_TRIGGER_L)
      out.triggerLeft = digitalAnalog;
    if (out.button & PAD_TRIGGER_R)
      out.triggerRight = digitalAnalog;
  }
}

} // namespace aurora::pad

void PADSetFightMapping(PADFightDevice device, const PADFightOutput* outputs, u32 count) {
  using namespace aurora::pad;
  if (device < 0 || device >= PAD_FIGHT_DEVICE_COUNT || outputs == nullptr)
    return;
  std::lock_guard lock{g_mutex};
  const u32 n = std::min(count, input_count(device));
  std::copy(outputs, outputs + n, g_tables[device].begin());
}

void PADSetFightMappingActive(bool active) {
  using namespace aurora::pad;
  if (g_active.exchange(active, std::memory_order_acq_rel) != active)
    g_gen.fetch_add(1, std::memory_order_release);
}

bool PADFightMappingActive(void) { return aurora::pad::g_active.load(std::memory_order_acquire); }

const PADFightOutput* PADFightStandardMapping(PADFightDevice device) { return aurora::pad::fight_standard(device); }
