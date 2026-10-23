#ifndef AURORA_XR_H
#define AURORA_XR_H

/* Headset controller input for XR builds (AURORA_ENABLE_OPENXR).
 * The XR thread reads the controllers through OpenXR each display frame; the
 * game thread merges the newest state into its pad input. */

#include <dolphin/pad.h>

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Latest controller state as a GameCube pad for port 1. Returns false when
 * there is none (no XR session, or the app is not focused), in which case
 * *out is left untouched. Safe to call from any thread. */
bool aurora_xr_get_pad(PADStatus* out);

/* Where a laser points on the virtual screen's picture (menus), in the
 * game frame's own coordinates: 0..1 across the whole frame, x right, y
 * down. `pressed`: the trigger or a pinch is held, having started on the
 * picture. Returns false when no laser is on the picture (also during a 3D
 * fight, or while the screen is being moved); the outputs are then left
 * untouched. Any thread. */
bool aurora_xr_screen_pointer(float* x, float* y, bool* pressed);

/* Whether a hand is pressing the Back button beside the virtual screen's
 * bar (shown while hands, not controllers, are tracked): B, for menus. Any
 * thread. */
bool aurora_xr_screen_back(void);

/* Draw categories for 3D fights (docs/xr-3d-plan.md). Everything drawn after
 * this call, until the next one, is tagged with the category.
 *   AURORA_XR_MONO   only the normal flat frame (the default every frame)
 *   AURORA_XR_WORLD  fight geometry, re-drawn per eye in 3D. Pass the world
 *                    camera's view matrix (world -> camera, GX Mtx layout);
 *                    NULL keeps the frame's current one.
 *   AURORA_XR_HUD    HUD, re-drawn onto its own flat plane
 *   AURORA_XR_HIDDEN fight geometry left out of 3D. Only in the flat frame,
 *                    and dropped from it with the world while nothing reads
 *                    the pass (an EFB copy, such as a stage's screen feed).
 *                    6: 3 to 5 are the transform and clip markers below.
 * Call from the thread that issues GX commands. No-op unless an XR session
 * is presenting. */
enum { AURORA_XR_MONO = 0, AURORA_XR_WORLD = 1, AURORA_XR_HUD = 2, AURORA_XR_HIDDEN = 6 };

/* Lock-step pacing: when the headset runs at a multiple of 60 Hz, blocks
 * until it is time for the next game frame (every display frame at 60 Hz,
 * every other at 120 Hz) and returns true; the caller then skips its own
 * 60 Hz timer. Returns false at once when there is nothing to pace to (no
 * session, a non-multiple rate, AURORA_XR_LOCKSTEP=0). */
bool aurora_xr_pace(void);

/* The same pacing by tick count, for a caller that keeps its own schedule
 * of whole ticks (netplay's time sync, src/pc/vi.c): one tick per game
 * frame. aurora_xr_pace_count is the number of ticks so far, or 0 when there
 * is nothing to pace to (as aurora_xr_pace returning false); the frame
 * recorded next then carries no tick. aurora_xr_pace_until blocks until
 * tick `count` has come (at once if it has), the frame recorded next
 * carrying the newest tick, and returns false if pacing went away
 * meanwhile. */
uint64_t aurora_xr_pace_count(void);
bool aurora_xr_pace_until(uint64_t count);

/* Whether the fight on screen is paused. Call every frame the fight camera
 * draws. While paused, the controllers show lasers and the grips grab the
 * arena to move, turn and scale it instead of pressing Z. */
void aurora_xr_set_paused(bool paused);

/* The fight is held before it starts so the player can place the stage:
 * the hands grab the arena as in a pause. 0: not held; 1: held, showing
 * the button legend; 2: held, showing the how-to cards and the legend.
 * Any thread. */
void aurora_xr_set_placing(int mode);

/* Puts the current stage back at its default placement for this session.
 * Any thread. */
void aurora_xr_reset_stage(void);

/* Takes the current stage's placement as where every stage starts from now
 * on (its own size applied), and forgets the per-stage placements made so
 * far. A reset then returns a stage there instead of in front of the head.
 * Any thread. */
void aurora_xr_adopt_stage(void);
/* Whether aurora_xr_adopt_stage has been called. Any thread. */
bool aurora_xr_stage_adopted(void);

/* The pictures shown while placing (aurora_xr_set_placing): 0 the how-to
 * cards, 1 the button legend, for Touch controllers; 2 and 3 the same for
 * tracked hands, shown instead while a hand is tracked. `rgba` is width x height straight-alpha
 * RGBA8, copied before returning. Once each, before the first placing. */
void aurora_xr_set_placing_image(int which, int width, int height, const unsigned char* rgba);

/* A looping clip shown over a picture on the cards: 0-2 on the
 * controllers' (image 0), 3-5 on the hands' (image 2).
 * `rgba` is a width x height atlas of `frames` frameW x frameH frames,
 * `cols` to a row, played at `fps` forward, then back. x, y, w, h: where it
 * goes on the cards, in their pixels. Opaque. Copied; once each. */
void aurora_xr_set_placing_clip(int which, int width, int height, const unsigned char* rgba, int frames, int cols,
                                int frameW, int frameH, float fps, int x, int y, int w, int h);

/* Whether the room shows behind the game (passthrough): on for mixed
 * reality, off for full VR, where passthrough is stopped, not just hidden.
 * On by default. Any thread. */
void aurora_xr_set_passthrough(bool on);

/* The stage now playing (any id the game keys its stages by), the game
 * point (world units) that sits at the arena position, for stages whose
 * action happens far from the world origin, and its size against the
 * default arena scale (AURORA_XR_ARENA_SCALE). The arena goes back to where
 * the player last put this stage this session, or else to the default
 * placement at this stage's size. Placing one stage never moves another.
 * Any thread. */
void aurora_xr_set_stage(int stage, float x, float y, float z, float scale);
/* Moves the current stage's center (the game point at the arena position)
 * without touching where the arena is: for stages whose camera roams a
 * course bigger than the arena, which then slides through it. Any thread. */
void aurora_xr_set_stage_center(float x, float y, float z);
/* Zooms the 3D view of the current stage about its center by `zoom`, on top
 * of the arena's scale, leaving where the player put the stage as it is
 * (1: none; aurora_xr_set_stage resets it). For a stretch of a course worth
 * a closer look. Any thread. */
void aurora_xr_set_stage_zoom(float zoom);
/* Keeps the HUD over the 3D view in mixed reality too, for the current
 * stage (aurora_xr_set_stage resets it): for stages whose scenery reaches
 * the HUD's height, which would otherwise hide it. Any thread. */
void aurora_xr_set_hud_on_top(bool on);

/* The height (world units) of the stage's highest floor, so the HUD can
 * float clear of it. NaN: unknown (the HUD keeps a fixed height). Any
 * thread. */
void aurora_xr_set_stage_top(float y);

/* The stage's footprint (world units), for the soft shadow it casts on the
 * real surface under it in mixed reality: the x and z span of what stands
 * on that surface, and its underside (its lowest point in the 3D view, after
 * any cuts). xmin >= xmax: none (a stage whose course slides through the
 * arena). aurora_xr_set_stage clears it. Any thread. */
void aurora_xr_set_stage_footprint(float xmin, float xmax, float zmin, float zmax, float ybottom);
/* The box the hands grab the current stage by, in its game units (as the
 * footprint): a hand in it touches the stage rather than pointing, and a
 * laser on it takes hold. Call after aurora_xr_set_stage; without it, a box
 * sized for Melee's stages around the arena center. Any thread. */
void aurora_xr_set_grab_box(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax);

/* A stage that is a board standing on a real surface (a table, the floor):
 * its proxy, in meters at the default arena scale, drawn while it's placed.
 * With AURORA_XR_PLACE_AT_START=1 the player places it once the headset is
 * tracked, before anything else shows: a spot on a level surface, a size, a
 * height on a stand. That placement becomes every stage's start (as
 * aurora_xr_adopt_stage); the arena position is the board's foot. Any
 * thread. */
typedef struct AuroraXrBoardProxy {
  float pitchHalfX, pitchHalfZ; /* the pitch drawn on the top */
  float topHalfX, topHalfZ;     /* the board's top */
  float footHalfX, footHalfZ;   /* its foot, what stands on the stand or the surface */
  float depth;                  /* from the top down to the foot */
  float standHalfX, standHalfZ; /* the stand's column */
} AuroraXrBoardProxy;
void aurora_xr_set_board_proxy(const AuroraXrBoardProxy* proxy);
/* Whether the board is being placed now (the game sees an idle pad). */
bool aurora_xr_board_placing(void);
/* The stand under the board: from the arena position down to the surface the
 * board was placed on, in the current stage's game units; 0 when there's
 * none (not placed, or resting on the surface). Any thread. */
float aurora_xr_stand_height(void);

/* The room's colour grade for the current stage's fights in mixed reality:
 * passthrough tinted toward the stage's palette, a gain per channel (linear
 * light) and a saturation; 1, 1, 1, 1 is none. It fades in as a fight starts
 * (or its placing hold is let go) and out as the fight ends. It may change
 * during a fight, as often as every frame (a stage whose own lighting
 * changes): the room follows within about 100 ms.
 * aurora_xr_set_stage resets it to none. Any thread. */
void aurora_xr_set_room_grade(float r, float g, float b, float saturation);

/* The game: a heavy effect starts this frame (a KO's explosion). Dynamic
 * resolution drops ahead of it, as far as the last one needed, instead of
 * after its first late frames, and comes back once it has passed. Any
 * thread. */
void aurora_xr_heavy_effect(void);

/* Dev tools (AURORA_XR_DEV_TOOLS=1): a panel in paused fights. Its Audio
 * page's settings, which the game applies: spatial audio on, or bypassed
 * (every sound plain stereo, as without it), and the music's level (0 to 1,
 * times its usual: 0 mutes it, to hear the rest). */
typedef struct AuroraXrDevAudio {
  bool spatial;
  float music;
} AuroraXrDevAudio;
/* The game: its audio settings as they stand without the panel, which it
 * shows and Reset goes back to. Any thread. */
void aurora_xr_dev_audio_defaults(const AuroraXrDevAudio* audio);
/* The game: the panel's audio settings; false, leaving `audio` untouched,
 * without dev tools or before the defaults came. Any thread; cheap. */
bool aurora_xr_dev_audio(AuroraXrDevAudio* audio);

/* Dev tools' Stage page: a mixer for the current stage's own sounds, a few
 * channels a stage, each a level and a three-band parametric EQ (a low
 * shelf, a peak, a high shelf), which the game applies. */
enum { AURORA_XR_MIX_CHANNELS = 4, AURORA_XR_MIX_BANDS = 3 };
#define AURORA_XR_MIX_OFF (-40.f) /* a level at or under it: muted */
typedef struct AuroraXrMixBand {
  float freq; /* Hz: the shelf's corner, the peak's centre */
  float gain; /* dB */
  float q;
} AuroraXrMixBand;
typedef struct AuroraXrMixChannel {
  float level; /* dB over the usual */
  AuroraXrMixBand band[AURORA_XR_MIX_BANDS];
} AuroraXrMixChannel;
/* The game, each fight frame: the stage now playing (any key) and its
 * channels, `count` names (upper case) and their settings without the
 * panel, which Reset goes back to; count 0: none to mix. A stage's settings
 * on the panel are kept for the session. Any thread. */
void aurora_xr_dev_mix_stage(int stage, int count, const char* const names[],
                             const AuroraXrMixChannel defaults[]);
/* The game: the panel's settings for `stage`'s channel `channel`; false,
 * leaving `out` untouched, without dev tools or before the stage came. Any
 * thread; cheap. */
bool aurora_xr_dev_mix(int stage, int channel, AuroraXrMixChannel* out);
/* The game: the panel shows its Stage page now, so the pause should not
 * duck what's being mixed. Any thread; cheap. */
bool aurora_xr_dev_mix_listening(void);

/* Dev tools' Game page: the game's own options, a row each, a name and up
 * to AURORA_XR_DEV_CHOICES choices (upper case, short), `initial` picked
 * at first and after Reset. Returns the option, for aurora_xr_dev_option;
 * -1 without dev tools or past AURORA_XR_DEV_OPTIONS. Any thread; once an
 * option. */
enum { AURORA_XR_DEV_OPTIONS = 5, AURORA_XR_DEV_CHOICES = 4 };
int aurora_xr_dev_option_add(const char* name, int count, const char* const choices[], int initial);
/* The game: the choice picked for `option`, or -1 for -1. Any thread;
 * cheap. */
int aurora_xr_dev_option(int option);
/* A slider row on the Game page, from `min` to `max` in steps of `step`
 * (0: smooth), `initial` at first and after Reset. Returns the row, for
 * aurora_xr_dev_slider, or -1 as aurora_xr_dev_option_add. Any thread; once
 * a slider. */
int aurora_xr_dev_slider_add(const char* name, float min, float max, float step, float initial);
/* The game: the slider's value, or `fallback` for -1. Any thread; cheap. */
float aurora_xr_dev_slider(int option, float fallback);

/* True while an OpenXR session is presenting the game. Any thread. */
bool aurora_xr_active(void);
/* For positional audio: the transform from the current stage's game world
 * (game units) to the head's own space (meters; x right, y up, -z ahead),
 * from the newest head pose and the arena's placement, in GX Mtx layout.
 * False, leaving `m` untouched, while no session presents or the head pose
 * is not tracked. Any thread; cheap (two short locks), so an audio thread
 * may call it every audio frame. */
bool aurora_xr_audio_listener(float m[3][4]);
/* For audio in the menus: while the flat screen is shown (not a fight in 3D),
 * its center, its unit right axis (both in the head's own space, as for
 * aurora_xr_audio_listener) and its width in meters. False otherwise, or
 * while the head pose is not tracked. Any thread; cheap. */
bool aurora_xr_audio_screen(float center[3], float right[3], float* width);
void aurora_xr_camera(int category, const float view[3][4]);

/* Extra placement for the following AURORA_XR_WORLD draws in the 3D view
 * only: a world-space transform (GX Mtx layout, game units) applied before
 * the arena's, e.g. to pull a far background piece in. NULL = none. Up to
 * three distinct transforms a frame; the rest draw unmoved. Same thread and
 * FIFO ordering as aurora_xr_camera. */
void aurora_xr_world_transform(const float m[3][4]);

/* Clip plane for the following AURORA_XR_WORLD draws in the 3D view only, in
 * game world units: geometry where a*x + b*y + c*z + d < 0 is cut away (e.g.
 * {0, 1, 0, -y0} keeps everything above y0). Applies with or without an
 * aurora_xr_world_transform, before it. NULL = none. Needs multiview;
 * otherwise ignored. */
void aurora_xr_world_clip(const float plane[4]);
/* Same, faded: geometry within `fade` units above the plane dissolves out
 * (ordered dither) instead of ending in a hard edge. */
void aurora_xr_world_clip_soft(const float plane[4], float fade);
/* Two planes at once (both must pass); plane2 may be NULL. */
void aurora_xr_world_clips(const float plane1[4], float fade1, const float plane2[4], float fade2);
/* Up to AURORA_XR_MAX_CLIPS planes (all must pass), each with its fade
 * band; count 0 = none, past the maximum = the first ones. (Named for its
 * first limit, four.) */
enum { AURORA_XR_MAX_CLIPS = 8 };
void aurora_xr_world_clips4(const float planes[][4], const float fades[], int count);
/* The same, with the clipped geometry also dissolved as a whole: opacity 1
 * is solid, lower values dither more of it away (0: none left). With count
 * 0 and opacity below 1, only the dissolve applies. */
void aurora_xr_world_clips4_fade(const float planes[][4], const float fades[], int count, float opacity);

#ifdef __cplusplus
}
#endif

#endif
