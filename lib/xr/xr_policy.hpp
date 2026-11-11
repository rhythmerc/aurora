#pragma once

// The OpenXR presenter's decisions that depend only on what the runtime
// offers and on knobs, kept free of OpenXR and Vulkan types so they can be
// unit-tested (tests/xr_policy_test.cpp). lib/xr/xr.cpp asks the runtime and
// acts on the answers.
//
// The Quest's answers are the tests' golden cases: a runtime offering what
// the Quest offers gets exactly what it got before these were factored out.

#include <cstdint>
#include <string_view>
#include <vector>

namespace aurora::xr::policy {

// ---------------------------------------------------------------- the room

// How the room is shown behind the game:
//   Fb     an XR_FB_passthrough layer under the others (Quest)
//   Alpha  the frame blended over the room by its alpha
//          (XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND: the Steam Frame on the
//          headset, which has no XR_FB_passthrough)
//   None   no room: the frame is opaque (full VR, or a runtime offering
//          neither, like SteamVR streaming to a headset)
enum class Room { None, Fb, Alpha };

// AURORA_XR_ROOM: auto (the default), fb, blend or none.
enum class RoomPref { Auto, Fb, Blend, None };
RoomPref parse_room_pref(const char* env) noexcept;

// Whether to try the XR_FB_passthrough layer at all (blend and none leave
// it alone, so a runtime whose FB layer misbehaves can be tested by blending).
bool try_fb_layer(RoomPref pref) noexcept;

// fbLayer: an XR_FB_passthrough layer was created. alphaOffered: the system
// lists ALPHA_BLEND for the stereo view configuration.
Room room_mode(bool fbLayer, bool alphaOffered, RoomPref pref) noexcept;

// Whether a fight's 3D view goes first, under every other layer. SteamVR
// (blending) reprojects only a frame's first layer for the head: any later
// projection layer is shown as drawn, and a 3D view drawn one to three display
// frames before flickered between three places as the head moved. There the
// 3D view, see-through where it drew nothing, is the room's layer too, with
// the HUD above it. The Quest's room is a layer of its own, under the HUD,
// under the 3D view.
bool fight_layer_first(Room room) noexcept;

// ---------------------------------------------------------------- pacing

// The display's rate in Hz. reportedHz: XR_FB_display_refresh_rate's answer
// (0: none). periodNs: xrWaitFrame's predictedDisplayPeriod (0: none yet).
// The report is believed while the period agrees with it (within 10%); a
// runtime can grant one rate and run at another, and some offer no report at
// all, so otherwise the period decides. With neither, 60 Hz.
float display_rate(float reportedHz, int64_t periodNs) noexcept;

// Display frames per game frame when the display runs at a whole multiple of
// 60 Hz (the game is paced to it); 0 otherwise (the game keeps its own 60 Hz
// timer).
int per_game_frame(float rateHz) noexcept;

// xrWaitFrame's predictedDisplayPeriod, watched for a change that lasts: a
// period over 10% off the current one counts once it has held for
// `settleFrames` frames running. SteamVR on the Steam Frame reports two or
// three display frames' period now and then for a single frame, and pacing
// re-planned on each flipped the game between one and two display frames per
// game frame some 20 times a second. The first period (current 0) counts at
// once.
struct PeriodWatch {
  int64_t candidate = 0;
  int frames = 0;
};
bool period_changed(PeriodWatch& watch, int64_t period, int64_t current, int settleFrames) noexcept;

// The refresh rate to request from those offered: wantedHz (AURORA_XR_REFRESH,
// 0: none) if offered, else 60, else 120. 0: request nothing. A runtime that
// lists just one rate (SteamVR on the Steam Frame lists the one it runs at,
// its auto-adjust moving it among 72, 90, 108, 120 and 144 Hz) is asked for
// wantedHz, else 120, even when already there (to hold it): at the others the
// game's 60 frames a second land unevenly on the display frames, and the 3D
// view shook with every head movement. Steady at 120 it didn't.
float pick_refresh(const std::vector<float>& rates, float wantedHz) noexcept;

// ---------------------------------------------------------------- the queue

// Separate: the runtime and the XR thread get a queue of their own in Dawn's
// queue family. Shared: the family has only Dawn's (Turnip, RADV), so all of
// them use it, one at a time (lib/webgpu/queue_guard.hpp).
enum class QueueMode { Separate, Shared };
QueueMode queue_mode(uint32_t extraQueues, bool forceShared) noexcept;

// ---------------------------------------------------------------- input

// What a hand holds, from its current interaction profile.
enum class Profile { None, Touch, Frame, Simple, HandInteraction, OtherController };
Profile classify_profile(std::string_view path) noexcept;

// Whether a hand's tracked joints are a bare hand (pinches, a laser from the
// knuckle) rather than a controller in it. SteamVR reports joints for a hand
// holding a controller, from the controller; those aren't a bare hand. The
// Quest tracks hands only with the controller put down, so its profiles keep
// today's answer: joints are a hand whenever they're tracked.
bool joints_are_hand(Profile current) noexcept;

// The pad's actions (lib/xr/xr.cpp maps them to XrActions).
enum class Act {
  Stick,
  CStick,
  A,
  B,
  X,
  Y,
  GripL,
  GripR,
  StickClickL,
  StickClickR,
  Start,
  L,
  R,
  AimL,
  AimR,
  GrabL,
  GrabR,
  DUp,
  DDown,
  DLeft,
  DRight,
  View, // the Steam Frame's left View button: room or skybox, in a paused fight
  RumbleL,
  RumbleR,
};
struct Binding {
  Act act;
  const char* path;
};

// Touch controllers, as on the Quest: the grips and stick clicks as fight
// inputs (the grips Z by default), the left menu button Start. With haptics, rumble on both.
std::vector<Binding> touch_bindings(bool haptics);

// The Steam Frame's controllers (XR_VALVE_frame_controller_interaction), in
// the order to try: a suggestion with one path the runtime doesn't know is
// rejected whole, and SteamVR renamed the shoulder buttons (bumper, before
// 2.17.10). The shoulders fill the Touch fight table's grip inputs (Z by
// default), the grips only grab, A/B/X/Y on the right controller, the left
// D-pad the GameCube's, the right menu button Start, the left View button
// View (the first variants; the rest without it, in case it's refused); the
// first variants with haptics.
std::vector<std::vector<Binding>> frame_binding_variants();

// ---------------------------------------------------------------- swapchains

// Whether another swapchain may be created. live: those alive now. budget:
// AURORA_XR_MAX_SWAPCHAINS (0: no limit; the Steam Frame allows 16 a
// session). A required one (the screen, the 3D view, the HUD, the lasers) is
// always tried; an optional one (the shadow, placing cards, the dev panel) is
// skipped at the budget.
bool swapchain_admit(int live, int budget, bool required) noexcept;

} // namespace aurora::xr::policy
