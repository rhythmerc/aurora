#pragma once

#include <dolphin/pad.h>

#include <cstdint>

// The fight mapping (PADSetFightMapping), shared by PADRead's gamepads and
// the XR controllers (aurora_xr_get_pad).
namespace aurora::pad {

// One reader of a device (a port's gamepad, the XR controllers): what it
// held when the mapping last turned on or off, until released.
struct FightReader {
  uint32_t gen = 0;
  uint32_t suppressed = 0;
};

// The device's usual GameCube mapping, as a fight table.
const PADFightOutput* fight_standard(PADFightDevice device);

// Clears from `raw` what the reader still holds from before the mapping
// turned on or off, if it means something else now.
void fight_filter(PADFightDevice device, FightReader& reader, PADFightRaw& raw);

// A GameCube status from raw inputs through `table` (the active fight table
// when null). Sticks come out at `mainScale` and `cScale` at full tilt; a
// pressed input that gives L or R sets that trigger's analog value to
// `digitalAnalog` (PADRead's 180 for a gamepad; 0 leaves the analog value).
void fight_compose(PADFightDevice device, const PADFightRaw& raw, const PADFightOutput* table, float mainScale,
                   float cScale, uint8_t digitalAnalog, PADStatus& out);

} // namespace aurora::pad
