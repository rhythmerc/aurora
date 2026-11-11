#include "xr/xr_policy.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <string>

using namespace aurora::xr::policy;

namespace {

bool has(const std::vector<Binding>& b, Act act, const char* path) {
  return std::any_of(b.begin(), b.end(), [&](const Binding& x) { return x.act == act && !std::strcmp(x.path, path); });
}

int count(const std::vector<Binding>& b, Act act) {
  return static_cast<int>(std::count_if(b.begin(), b.end(), [&](const Binding& x) { return x.act == act; }));
}

} // namespace

// ---------------------------------------------------------------- the room

TEST(XrPolicyRoom, QuestKeepsItsPassthroughLayer) {
  // The Quest: an XR_FB_passthrough layer. It also lists ALPHA_BLEND for
  // nothing; the layer wins.
  EXPECT_EQ(room_mode(true, false, RoomPref::Auto), Room::Fb);
  EXPECT_EQ(room_mode(true, true, RoomPref::Auto), Room::Fb);
  EXPECT_TRUE(try_fb_layer(RoomPref::Auto));
  // Its room is a layer of its own, under the HUD, under the 3D view.
  EXPECT_FALSE(fight_layer_first(Room::Fb));
  EXPECT_FALSE(fight_layer_first(Room::None));
}

TEST(XrPolicyRoom, AlphaBlendWithoutTheLayer) {
  // The Steam Frame on the headset: no XR_FB_passthrough, ALPHA_BLEND offered.
  EXPECT_EQ(room_mode(false, true, RoomPref::Auto), Room::Alpha);
  // SteamVR reprojects only the first layer: the 3D view goes there.
  EXPECT_TRUE(fight_layer_first(Room::Alpha));
  // SteamVR streaming to a headset, or Monado: neither.
  EXPECT_EQ(room_mode(false, false, RoomPref::Auto), Room::None);
}

TEST(XrPolicyRoom, Preferences) {
  EXPECT_EQ(parse_room_pref(nullptr), RoomPref::Auto);
  EXPECT_EQ(parse_room_pref(""), RoomPref::Auto);
  EXPECT_EQ(parse_room_pref("auto"), RoomPref::Auto);
  EXPECT_EQ(parse_room_pref("fb"), RoomPref::Fb);
  EXPECT_EQ(parse_room_pref("blend"), RoomPref::Blend);
  EXPECT_EQ(parse_room_pref("none"), RoomPref::None);
  EXPECT_EQ(room_mode(true, true, RoomPref::Blend), Room::Alpha);
  EXPECT_EQ(room_mode(true, false, RoomPref::Blend), Room::None);
  EXPECT_EQ(room_mode(false, true, RoomPref::Fb), Room::None);
  EXPECT_EQ(room_mode(true, true, RoomPref::None), Room::None);
  EXPECT_FALSE(try_fb_layer(RoomPref::Blend));
  EXPECT_FALSE(try_fb_layer(RoomPref::None));
}

// ---------------------------------------------------------------- pacing

TEST(XrPolicyPacing, QuestAt120) {
  // XR_FB_display_refresh_rate says 120 and frames come every 8.33 ms: the
  // game steps every second display frame, as it always has.
  EXPECT_FLOAT_EQ(display_rate(120.f, 8333333), 120.f);
  EXPECT_FLOAT_EQ(display_rate(120.f, 0), 120.f); // before the first frame
  EXPECT_EQ(per_game_frame(120.f), 2);
  EXPECT_EQ(per_game_frame(60.f), 1);
  EXPECT_EQ(per_game_frame(180.f), 3);
}

TEST(XrPolicyPacing, ThePeriodWhenThereIsNoReport) {
  // A 90 Hz runtime without the extension: 90, not the 60 once assumed
  // (which ran the game at 90 steps a second).
  EXPECT_NEAR(display_rate(0.f, 11111111), 90.f, 0.01f);
  EXPECT_EQ(per_game_frame(display_rate(0.f, 11111111)), 0);
  EXPECT_NEAR(display_rate(0.f, 8333333), 120.f, 0.01f);
  EXPECT_EQ(per_game_frame(display_rate(0.f, 8333333)), 2);
  EXPECT_FLOAT_EQ(display_rate(0.f, 0), 60.f);
}

TEST(XrPolicyPacing, ThePeriodWhenTheReportIsWrong) {
  // SteamVR on the Steam Frame can grant 90 Hz and run at 72.
  EXPECT_NEAR(display_rate(90.f, 13888889), 72.f, 0.01f);
  // Within 10% the report stands (a period's jitter).
  EXPECT_FLOAT_EQ(display_rate(90.f, 11300000), 90.f);
}

TEST(XrPolicyPacing, NotMultiplesOf60) {
  for (float hz : {72.f, 80.f, 90.f, 144.f, 30.f, 0.f})
    EXPECT_EQ(per_game_frame(hz), 0) << hz;
}

TEST(XrPolicyPacing, PeriodChangeMustLast) {
  constexpr int64_t k120 = 8333333, k60 = 16666667, k40 = 25000000;
  PeriodWatch w;
  EXPECT_TRUE(period_changed(w, k120, 0, 12)); // the first period counts at once
  // SteamVR: one long frame now and then (seen at 120 Hz on the Steam Frame).
  for (int i = 0; i < 50; ++i) {
    EXPECT_FALSE(period_changed(w, i % 6 == 5 ? k60 : k120, k120, 12));
    EXPECT_FALSE(period_changed(w, i % 9 == 8 ? k40 : k120, k120, 12));
  }
  // A change that lasts (SpaceWarp halving the loop) counts after 12 frames.
  for (int i = 0; i < 11; ++i)
    EXPECT_FALSE(period_changed(w, k60, k120, 12));
  EXPECT_TRUE(period_changed(w, k60, k120, 12));
  // Within 10%: no change.
  EXPECT_FALSE(period_changed(w, k120 + 400000, k120, 1));
  // Settling of 1: every change at once (as before).
  EXPECT_TRUE(period_changed(w, k60, k120, 1));
}

TEST(XrPolicyPacing, RefreshPick) {
  // The Quest 3: 60 when offered, else 120.
  EXPECT_FLOAT_EQ(pick_refresh({72.f, 80.f, 90.f, 120.f}, 0.f), 120.f);
  EXPECT_FLOAT_EQ(pick_refresh({60.f, 72.f, 90.f, 120.f}, 0.f), 60.f);
  EXPECT_FLOAT_EQ(pick_refresh({72.f, 90.f, 120.f}, 90.f), 90.f);
  EXPECT_FLOAT_EQ(pick_refresh({72.f, 90.f}, 0.f), 0.f);
  // The Steam Frame lists only the rate it runs at: asked for 120 (held when
  // already there), or for AURORA_XR_REFRESH.
  EXPECT_FLOAT_EQ(pick_refresh({72.f}, 0.f), 120.f);
  EXPECT_FLOAT_EQ(pick_refresh({90.f}, 0.f), 120.f);
  EXPECT_FLOAT_EQ(pick_refresh({144.f}, 0.f), 120.f);
  EXPECT_FLOAT_EQ(pick_refresh({120.f}, 0.f), 120.f);
  EXPECT_FLOAT_EQ(pick_refresh({120.f}, 120.f), 120.f);
  EXPECT_FLOAT_EQ(pick_refresh({72.f}, 90.f), 90.f);
  EXPECT_FLOAT_EQ(pick_refresh({}, 0.f), 0.f);
}

// ---------------------------------------------------------------- the queue

TEST(XrPolicyQueue, Modes) {
  EXPECT_EQ(queue_mode(1, false), QueueMode::Separate); // the Quest
  EXPECT_EQ(queue_mode(0, false), QueueMode::Shared);   // Turnip, RADV
  EXPECT_EQ(queue_mode(1, true), QueueMode::Shared);    // AURORA_XR_SHARED_QUEUE=1
}

// ---------------------------------------------------------------- input

TEST(XrPolicyInput, Profiles) {
  EXPECT_EQ(classify_profile(""), Profile::None);
  EXPECT_EQ(classify_profile("/interaction_profiles/oculus/touch_controller"), Profile::Touch);
  EXPECT_EQ(classify_profile("/interaction_profiles/valve/frame_controller_valve"), Profile::Frame);
  EXPECT_EQ(classify_profile("/interaction_profiles/khr/simple_controller"), Profile::Simple);
  EXPECT_EQ(classify_profile("/interaction_profiles/ext/hand_interaction_ext"), Profile::HandInteraction);
  EXPECT_EQ(classify_profile("/interaction_profiles/valve/index_controller"), Profile::OtherController);
}

TEST(XrPolicyInput, HandJoints) {
  // The Quest's answers are unchanged: tracked joints are a bare hand.
  for (auto p : {Profile::None, Profile::Touch, Profile::Simple, Profile::HandInteraction})
    EXPECT_TRUE(joints_are_hand(p));
  // SteamVR's joints for a held controller are the controller.
  EXPECT_FALSE(joints_are_hand(Profile::Frame));
  EXPECT_FALSE(joints_are_hand(Profile::OtherController));
}

TEST(XrPolicyInput, TouchBindingsAsBefore) {
  // The table the Quest has always had.
  const auto b = touch_bindings(false);
  EXPECT_EQ(b.size(), 17u);
  EXPECT_TRUE(has(b, Act::Stick, "/user/hand/left/input/thumbstick"));
  EXPECT_TRUE(has(b, Act::CStick, "/user/hand/right/input/thumbstick"));
  EXPECT_TRUE(has(b, Act::A, "/user/hand/right/input/a/click"));
  EXPECT_TRUE(has(b, Act::B, "/user/hand/right/input/b/click"));
  EXPECT_TRUE(has(b, Act::X, "/user/hand/left/input/x/click"));
  EXPECT_TRUE(has(b, Act::Y, "/user/hand/left/input/y/click"));
  EXPECT_TRUE(has(b, Act::GripL, "/user/hand/left/input/squeeze/value"));
  EXPECT_TRUE(has(b, Act::GripR, "/user/hand/right/input/squeeze/value"));
  EXPECT_TRUE(has(b, Act::StickClickL, "/user/hand/left/input/thumbstick/click"));
  EXPECT_TRUE(has(b, Act::StickClickR, "/user/hand/right/input/thumbstick/click"));
  EXPECT_TRUE(has(b, Act::Start, "/user/hand/left/input/menu/click"));
  EXPECT_TRUE(has(b, Act::L, "/user/hand/left/input/trigger/value"));
  EXPECT_TRUE(has(b, Act::R, "/user/hand/right/input/trigger/value"));
  EXPECT_TRUE(has(b, Act::AimL, "/user/hand/left/input/aim/pose"));
  EXPECT_TRUE(has(b, Act::AimR, "/user/hand/right/input/aim/pose"));
  EXPECT_TRUE(has(b, Act::GrabL, "/user/hand/left/input/squeeze/value"));
  EXPECT_TRUE(has(b, Act::GrabR, "/user/hand/right/input/squeeze/value"));
  EXPECT_EQ(count(b, Act::DUp), 0);
  const auto h = touch_bindings(true);
  EXPECT_EQ(h.size(), 19u);
  EXPECT_TRUE(has(h, Act::RumbleL, "/user/hand/left/output/haptic"));
  EXPECT_TRUE(has(h, Act::RumbleR, "/user/hand/right/output/haptic"));
}

TEST(XrPolicyInput, FrameBindings) {
  const auto variants = frame_binding_variants();
  ASSERT_GE(variants.size(), 2u);
  const auto& best = variants.front();
  // The shoulders are the fight table's grip inputs (Z); the grips only grab.
  EXPECT_TRUE(has(best, Act::GripL, "/user/hand/left/input/bumper/click"));
  EXPECT_TRUE(has(best, Act::GripR, "/user/hand/right/input/bumper/click"));
  EXPECT_FALSE(has(best, Act::GripL, "/user/hand/left/input/squeeze/value"));
  EXPECT_TRUE(has(best, Act::StickClickL, "/user/hand/left/input/thumbstick/click"));
  EXPECT_TRUE(has(best, Act::GrabL, "/user/hand/left/input/squeeze/value"));
  EXPECT_TRUE(has(best, Act::GrabR, "/user/hand/right/input/squeeze/value"));
  // A/B/X/Y and Start on the right controller, the D-pad on the left.
  for (auto [act, path] :
       {std::pair{Act::A, "/user/hand/right/input/a/click"}, std::pair{Act::B, "/user/hand/right/input/b/click"},
        std::pair{Act::X, "/user/hand/right/input/x/click"}, std::pair{Act::Y, "/user/hand/right/input/y/click"},
        std::pair{Act::Start, "/user/hand/right/input/menu/click"},
        std::pair{Act::DUp, "/user/hand/left/input/dpad_up/click"},
        std::pair{Act::DDown, "/user/hand/left/input/dpad_down/click"},
        std::pair{Act::DLeft, "/user/hand/left/input/dpad_left/click"},
        std::pair{Act::DRight, "/user/hand/left/input/dpad_right/click"}})
    EXPECT_TRUE(has(best, act, path)) << path;
  // The left View button switches room and skybox; without it next.
  EXPECT_TRUE(has(best, Act::View, "/user/hand/left/input/view/click"));
  EXPECT_EQ(count(variants[2], Act::View), 0);
  EXPECT_TRUE(has(variants[2], Act::DUp, "/user/hand/left/input/dpad_up/click"));
  EXPECT_EQ(count(best, Act::RumbleL), 1);
  // The rename: shoulder after bumper.
  EXPECT_TRUE(has(variants[1], Act::GripL, "/user/hand/left/input/shoulder/click"));
  // The last resort binds nothing optional.
  const auto& last = variants.back();
  EXPECT_EQ(count(last, Act::GripL), 0);
  EXPECT_EQ(count(last, Act::DUp), 0);
  EXPECT_EQ(count(last, Act::RumbleL), 0);
  EXPECT_EQ(count(last, Act::A), 1);
  // Every variant has the pad and the pointers.
  for (const auto& v : variants)
    for (auto act : {Act::Stick, Act::CStick, Act::A, Act::B, Act::X, Act::Y, Act::Start, Act::L, Act::R, Act::AimL,
                     Act::AimR, Act::GrabL, Act::GrabR})
      EXPECT_EQ(count(v, act), 1);
}

// ---------------------------------------------------------------- swapchains

TEST(XrPolicySwapchains, Budget) {
  EXPECT_TRUE(swapchain_admit(40, 0, false)); // no budget: as before
  EXPECT_TRUE(swapchain_admit(15, 16, false));
  EXPECT_FALSE(swapchain_admit(16, 16, false));
  EXPECT_TRUE(swapchain_admit(16, 16, true)); // required ones are always tried
}
