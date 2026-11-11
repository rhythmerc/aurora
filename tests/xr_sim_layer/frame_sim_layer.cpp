// XR_APILAYER_AURORA_frame_sim: an OpenXR API layer that makes a desktop
// runtime (Monado) look like SteamVR on the Steam Frame, for testing the
// presenter's paths that only that headset takes, on a desktop. Test only;
// never shipped. Each part is switched on by an environment variable:
//
//   AURORA_SIM_BLEND=alpha     ALPHA_BLEND is offered (and taken off frames
//                              before they reach the runtime); frames that
//                              ask for it without a projection layer first
//                              are counted (the Frame shows the room only
//                              behind one).
//   AURORA_SIM_MAX_SWAPCHAINS=n  xrCreateSwapchain fails with
//                              XR_ERROR_LIMIT_REACHED past n alive (the
//                              Frame allows 16).
//   AURORA_SIM_FRAME=1         XR_VALVE_frame_controller_interaction: the
//                              Frame controller's bindings are checked
//                              against its documented inputs and accepted,
//                              and both hands report its profile.
//   AURORA_SIM_FRAME_SHOULDER=bumper|shoulder|both  which name the shoulder
//                              buttons go by (both: SteamVR 2.17.10 on).
//   AURORA_SIM_PRESS=path@first-last;...  Frame inputs held down during
//                              those frames (counted by xrEndFrame), e.g.
//                              /user/hand/left/input/dpad_up/click@600-640.
//   AURORA_SIM_HANDS=1         hand tracking, with joints wherever a hand is
//                              (as SteamVR derives them from a held
//                              controller), thumb and index pinched.
//   AURORA_SIM_PERIOD_MS=x     xrWaitFrame's predictedDisplayPeriod.
//   AURORA_SIM_REPORTED_HZ=x   XR_FB_display_refresh_rate's rate (0: none).
//   AURORA_SIM_RATES=a,b       the rates it lists.
//
// At xrDestroyInstance it logs (stderr, "frame_sim:") what it saw: frames,
// frames blended without a projection layer first, frames with a projection
// layer after the first layer (SteamVR reprojects only the first for the
// head: a later one flickers as the head moves), the most swapchains alive,
// refused creations, Frame suggestions accepted and refused.

#include <openxr/openxr.h>
#include <openxr/openxr_loader_negotiation.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#if defined(_WIN32)
#define SIM_EXPORT extern "C" __declspec(dllexport)
#else
#define SIM_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr const char* kLayerName = "XR_APILAYER_AURORA_frame_sim";
constexpr const char* kFrameExt = "XR_VALVE_frame_controller_interaction";
constexpr const char* kFrameProfile = "/interaction_profiles/valve/frame_controller_valve";

std::string env(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr ? v : "";
}
bool env_on(const char* name) {
  const std::string v = env(name);
  return !v.empty() && v != "0";
}

struct Press {
  std::string path;
  uint64_t first = 0, last = 0;
};

struct State {
  PFN_xrGetInstanceProcAddr next = nullptr;
  XrInstance instance = XR_NULL_HANDLE;
  XrSession session = XR_NULL_HANDLE;
  // Next layer's (or the runtime's) functions.
  PFN_xrDestroyInstance destroyInstance = nullptr;
  PFN_xrEnumerateEnvironmentBlendModes enumerateBlendModes = nullptr;
  PFN_xrCreateSession createSession = nullptr;
  PFN_xrEndFrame endFrame = nullptr;
  PFN_xrWaitFrame waitFrame = nullptr;
  PFN_xrCreateSwapchain createSwapchain = nullptr;
  PFN_xrDestroySwapchain destroySwapchain = nullptr;
  PFN_xrSuggestInteractionProfileBindings suggest = nullptr;
  PFN_xrGetCurrentInteractionProfile currentProfile = nullptr;
  PFN_xrPollEvent pollEvent = nullptr;
  PFN_xrPathToString pathToString = nullptr;
  PFN_xrStringToPath stringToPath = nullptr;
  PFN_xrGetActionStateBoolean actionBool = nullptr;
  PFN_xrGetActionStateFloat actionFloat = nullptr;
  PFN_xrGetSystemProperties systemProperties = nullptr;
  PFN_xrGetDisplayRefreshRateFB getRefreshRate = nullptr;
  PFN_xrEnumerateDisplayRefreshRatesFB enumerateRates = nullptr;

  bool blend = false, frame = false, hands = false;
  int maxSwapchains = 0;
  std::string shoulder = "both";
  double periodMs = 0.0;
  bool reportHz = false;
  float reportedHz = 0.f;
  std::vector<float> rates;
  std::vector<Press> presses;

  std::mutex mutex;
  std::set<XrSwapchain> swapchains;
  int peakSwapchains = 0, refusedSwapchains = 0;
  uint64_t frames = 0, blendedFrames = 0, blendedWithoutProjection = 0, laterProjectionFrames = 0;
  int frameAccepted = 0, frameRejected = 0;
  std::map<XrAction, std::vector<std::string>> frameBindings; // action -> Frame input paths
  bool profileEventPending = false;
};
State S;

void load(const char* name, PFN_xrVoidFunction* out) {
  if (S.next(S.instance, name, out) != XR_SUCCESS)
    *out = nullptr;
}

std::string path_string(XrPath p) {
  char buf[XR_MAX_PATH_LENGTH];
  uint32_t n = 0;
  if (S.pathToString == nullptr || S.pathToString(S.instance, p, sizeof buf, &n, buf) != XR_SUCCESS)
    return {};
  return buf;
}

// The Frame controller's documented inputs (partner.steamgames.com: Steam
// Frame input); shoulder per AURORA_SIM_FRAME_SHOULDER.
bool frame_path_known(const std::string& path) {
  static const std::set<std::string> known = [] {
    std::set<std::string> k;
    for (const char* hand : {"left", "right"}) {
      const std::string h = std::string{"/user/hand/"} + hand;
      for (const char* c : {"system/touch", "system/click", "squeeze/touch", "squeeze/click", "squeeze/value",
                            "trigger/touch", "trigger/click", "trigger/value", "thumbstick", "thumbstick/x",
                            "thumbstick/y", "thumbstick/touch", "thumbstick/click", "grip/pose", "aim/pose"})
        k.insert(h + "/input/" + c);
      k.insert(h + "/output/haptic");
      for (const char* s : {"bumper", "shoulder"})
        for (const char* t : {"touch", "click", "value"})
          k.insert(h + "/input/" + s + "/" + t);
    }
    for (const char* c : {"a", "b", "x", "y", "menu"})
      for (const char* t : {"touch", "click"})
        k.insert(std::string{"/user/hand/right/input/"} + c + "/" + t);
    for (const char* c : {"dpad_up", "dpad_down", "dpad_left", "dpad_right", "view"})
      for (const char* t : {"touch", "click"})
        k.insert(std::string{"/user/hand/left/input/"} + c + "/" + t);
    return k;
  }();
  if (!known.contains(path))
    return false;
  const bool isBumper = path.find("/bumper/") != std::string::npos;
  const bool isShoulder = path.find("/shoulder/") != std::string::npos;
  if (isBumper && S.shoulder == "shoulder")
    return false;
  if (isShoulder && S.shoulder == "bumper")
    return false;
  return true;
}

bool pressed(const std::string& path) {
  for (const auto& p : S.presses)
    if (p.path == path && S.frames >= p.first && S.frames <= p.last)
      return true;
  return false;
}

// ---------------------------------------------------------------- intercepts

XRAPI_ATTR XrResult XRAPI_CALL sim_DestroyInstance(XrInstance instance) {
  std::fprintf(stderr,
               "frame_sim: %llu frames; %llu blended, %llu of them without a projection layer first; %llu with "
               "a projection layer after the first; swapchains "
               "at most %d alive, %d refused; Frame bindings accepted %d, refused %d\n",
               static_cast<unsigned long long>(S.frames), static_cast<unsigned long long>(S.blendedFrames),
               static_cast<unsigned long long>(S.blendedWithoutProjection),
               static_cast<unsigned long long>(S.laterProjectionFrames), S.peakSwapchains, S.refusedSwapchains,
               S.frameAccepted, S.frameRejected);
  return S.destroyInstance(instance);
}

XRAPI_ATTR XrResult XRAPI_CALL sim_EnumerateEnvironmentBlendModes(XrInstance instance, XrSystemId systemId,
                                                                  XrViewConfigurationType type, uint32_t capacity,
                                                                  uint32_t* count, XrEnvironmentBlendMode* modes) {
  std::vector<XrEnvironmentBlendMode> all;
  uint32_t n = 0;
  XrResult r = S.enumerateBlendModes(instance, systemId, type, 0, &n, nullptr);
  if (r != XR_SUCCESS)
    return r;
  all.resize(n);
  r = S.enumerateBlendModes(instance, systemId, type, n, &n, all.data());
  if (r != XR_SUCCESS)
    return r;
  all.resize(n);
  if (S.blend && std::find(all.begin(), all.end(), XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND) == all.end())
    all.push_back(XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND);
  *count = static_cast<uint32_t>(all.size());
  if (capacity == 0)
    return XR_SUCCESS;
  if (capacity < all.size())
    return XR_ERROR_SIZE_INSUFFICIENT;
  std::copy(all.begin(), all.end(), modes);
  return XR_SUCCESS;
}

XRAPI_ATTR XrResult XRAPI_CALL sim_CreateSession(XrInstance instance, const XrSessionCreateInfo* info,
                                                 XrSession* session) {
  const XrResult r = S.createSession(instance, info, session);
  if (r == XR_SUCCESS)
    S.session = *session;
  return r;
}

XRAPI_ATTR XrResult XRAPI_CALL sim_WaitFrame(XrSession session, const XrFrameWaitInfo* info, XrFrameState* state) {
  const XrResult r = S.waitFrame(session, info, state);
  if (r == XR_SUCCESS && S.periodMs > 0.0)
    state->predictedDisplayPeriod = static_cast<XrDuration>(S.periodMs * 1e6);
  return r;
}

XRAPI_ATTR XrResult XRAPI_CALL sim_EndFrame(XrSession session, const XrFrameEndInfo* info) {
  XrFrameEndInfo copy = *info;
  {
    std::lock_guard lock{S.mutex};
    ++S.frames;
    if (info->environmentBlendMode == XR_ENVIRONMENT_BLEND_MODE_ALPHA_BLEND) {
      ++S.blendedFrames;
      if (info->layerCount == 0 || info->layers[0]->type != XR_TYPE_COMPOSITION_LAYER_PROJECTION)
        ++S.blendedWithoutProjection;
      bool later = false;
      for (uint32_t i = 1; i < info->layerCount; ++i)
        later |= info->layers[i]->type == XR_TYPE_COMPOSITION_LAYER_PROJECTION;
      S.laterProjectionFrames += later ? 1 : 0;
      if (S.blend)
        copy.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE; // what the runtime can show
    }
  }
  return S.endFrame(session, &copy);
}

XRAPI_ATTR XrResult XRAPI_CALL sim_CreateSwapchain(XrSession session, const XrSwapchainCreateInfo* info,
                                                   XrSwapchain* swapchain) {
  std::lock_guard lock{S.mutex};
  if (S.maxSwapchains > 0 && static_cast<int>(S.swapchains.size()) >= S.maxSwapchains) {
    ++S.refusedSwapchains;
    return XR_ERROR_LIMIT_REACHED;
  }
  const XrResult r = S.createSwapchain(session, info, swapchain);
  if (r == XR_SUCCESS) {
    S.swapchains.insert(*swapchain);
    S.peakSwapchains = std::max(S.peakSwapchains, static_cast<int>(S.swapchains.size()));
  }
  return r;
}

XRAPI_ATTR XrResult XRAPI_CALL sim_DestroySwapchain(XrSwapchain swapchain) {
  {
    std::lock_guard lock{S.mutex};
    S.swapchains.erase(swapchain);
  }
  return S.destroySwapchain(swapchain);
}

XRAPI_ATTR XrResult XRAPI_CALL
sim_SuggestInteractionProfileBindings(XrInstance instance, const XrInteractionProfileSuggestedBinding* suggested) {
  if (!S.frame || path_string(suggested->interactionProfile) != kFrameProfile)
    return S.suggest(instance, suggested);
  // Like a runtime: one path it doesn't know rejects the whole suggestion.
  std::map<XrAction, std::vector<std::string>> bindings;
  for (uint32_t i = 0; i < suggested->countSuggestedBindings; ++i) {
    const std::string p = path_string(suggested->suggestedBindings[i].binding);
    if (!frame_path_known(p)) {
      ++S.frameRejected;
      std::fprintf(stderr, "frame_sim: Frame bindings refused: unknown input %s\n", p.c_str());
      return XR_ERROR_PATH_UNSUPPORTED;
    }
    bindings[suggested->suggestedBindings[i].action].push_back(p);
  }
  ++S.frameAccepted;
  S.frameBindings = std::move(bindings);
  S.profileEventPending = true;
  return XR_SUCCESS;
}

XRAPI_ATTR XrResult XRAPI_CALL sim_GetCurrentInteractionProfile(XrSession session, XrPath topLevelUserPath,
                                                                XrInteractionProfileState* state) {
  if (!S.frame || S.frameBindings.empty())
    return S.currentProfile(session, topLevelUserPath, state);
  XrPath frame = XR_NULL_PATH;
  S.stringToPath(S.instance, kFrameProfile, &frame);
  state->interactionProfile = frame;
  return XR_SUCCESS;
}

XRAPI_ATTR XrResult XRAPI_CALL sim_PollEvent(XrInstance instance, XrEventDataBuffer* event) {
  if (S.profileEventPending && S.session != XR_NULL_HANDLE && S.frames > 0) {
    S.profileEventPending = false;
    auto* e = reinterpret_cast<XrEventDataInteractionProfileChanged*>(event);
    *e = {XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED};
    e->session = S.session;
    return XR_SUCCESS;
  }
  return S.pollEvent(instance, event);
}

// A Frame input held by AURORA_SIM_PRESS, for an action bound to it.
bool simulated(XrAction action, float& value) {
  if (!S.frame)
    return false;
  const auto it = S.frameBindings.find(action);
  if (it == S.frameBindings.end())
    return false;
  value = 0.f;
  for (const auto& p : it->second)
    if (pressed(p))
      value = 1.f;
  return true;
}

XRAPI_ATTR XrResult XRAPI_CALL sim_GetActionStateBoolean(XrSession session, const XrActionStateGetInfo* info,
                                                         XrActionStateBoolean* state) {
  float v = 0.f;
  if (!simulated(info->action, v))
    return S.actionBool(session, info, state);
  state->currentState = v > 0.5f ? XR_TRUE : XR_FALSE;
  state->isActive = XR_TRUE;
  state->changedSinceLastSync = XR_FALSE;
  state->lastChangeTime = 0;
  return XR_SUCCESS;
}

XRAPI_ATTR XrResult XRAPI_CALL sim_GetActionStateFloat(XrSession session, const XrActionStateGetInfo* info,
                                                       XrActionStateFloat* state) {
  float v = 0.f;
  if (!simulated(info->action, v))
    return S.actionFloat(session, info, state);
  state->currentState = v;
  state->isActive = XR_TRUE;
  state->changedSinceLastSync = XR_FALSE;
  state->lastChangeTime = 0;
  return XR_SUCCESS;
}

// ---- hands: joints wherever a hand is, pinched, as from a held controller.
struct FakeTracker {
  XrHandEXT hand;
};

XRAPI_ATTR XrResult XRAPI_CALL sim_GetSystemProperties(XrInstance instance, XrSystemId systemId,
                                                       XrSystemProperties* properties) {
  const XrResult r = S.systemProperties(instance, systemId, properties);
  if (r == XR_SUCCESS && S.hands)
    for (auto* p = static_cast<XrBaseOutStructure*>(properties->next); p != nullptr; p = p->next)
      if (p->type == XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT)
        reinterpret_cast<XrSystemHandTrackingPropertiesEXT*>(p)->supportsHandTracking = XR_TRUE;
  return r;
}

XRAPI_ATTR XrResult XRAPI_CALL sim_CreateHandTrackerEXT(XrSession, const XrHandTrackerCreateInfoEXT* info,
                                                        XrHandTrackerEXT* tracker) {
  *tracker = reinterpret_cast<XrHandTrackerEXT>(new FakeTracker{info->hand});
  return XR_SUCCESS;
}

XRAPI_ATTR XrResult XRAPI_CALL sim_DestroyHandTrackerEXT(XrHandTrackerEXT tracker) {
  delete reinterpret_cast<FakeTracker*>(tracker);
  return XR_SUCCESS;
}

XRAPI_ATTR XrResult XRAPI_CALL sim_LocateHandJointsEXT(XrHandTrackerEXT tracker, const XrHandJointsLocateInfoEXT*,
                                                       XrHandJointLocationsEXT* locations) {
  const float side = reinterpret_cast<FakeTracker*>(tracker)->hand == XR_HAND_LEFT_EXT ? -0.2f : 0.2f;
  locations->isActive = XR_TRUE;
  for (uint32_t i = 0; i < locations->jointCount; ++i) {
    auto& j = locations->jointLocations[i];
    j.locationFlags = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT |
                      XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;
    j.pose = {{0.f, 0.f, 0.f, 1.f}, {side, -0.3f, -0.4f}};
    j.radius = 0.01f;
  }
  // Thumb and index tips a centimetre apart: a pinch.
  if (locations->jointCount > XR_HAND_JOINT_INDEX_TIP_EXT) {
    locations->jointLocations[XR_HAND_JOINT_THUMB_TIP_EXT].pose.position.x += 0.005f;
    locations->jointLocations[XR_HAND_JOINT_INDEX_TIP_EXT].pose.position.x -= 0.005f;
    locations->jointLocations[XR_HAND_JOINT_INDEX_PROXIMAL_EXT].pose.position.z += 0.05f;
  }
  return XR_SUCCESS;
}

// ---- refresh rates
XRAPI_ATTR XrResult XRAPI_CALL sim_GetDisplayRefreshRateFB(XrSession session, float* rate) {
  if (!S.reportHz)
    return S.getRefreshRate(session, rate);
  *rate = S.reportedHz;
  return XR_SUCCESS;
}

XRAPI_ATTR XrResult XRAPI_CALL sim_EnumerateDisplayRefreshRatesFB(XrSession session, uint32_t capacity, uint32_t* count,
                                                                  float* rates) {
  if (S.rates.empty())
    return S.enumerateRates(session, capacity, count, rates);
  *count = static_cast<uint32_t>(S.rates.size());
  if (capacity == 0)
    return XR_SUCCESS;
  if (capacity < S.rates.size())
    return XR_ERROR_SIZE_INSUFFICIENT;
  std::copy(S.rates.begin(), S.rates.end(), rates);
  return XR_SUCCESS;
}

// ---------------------------------------------------------------- dispatch

XRAPI_ATTR XrResult XRAPI_CALL sim_GetInstanceProcAddr(XrInstance instance, const char* name,
                                                       PFN_xrVoidFunction* function) {
  const std::string n = name;
#define SIM_FN(api, fn)                                                                                                \
  if (n == api) {                                                                                                      \
    *function = reinterpret_cast<PFN_xrVoidFunction>(fn);                                                              \
    return XR_SUCCESS;                                                                                                 \
  }
  SIM_FN("xrGetInstanceProcAddr", sim_GetInstanceProcAddr)
  SIM_FN("xrDestroyInstance", sim_DestroyInstance)
  SIM_FN("xrEnumerateEnvironmentBlendModes", sim_EnumerateEnvironmentBlendModes)
  SIM_FN("xrCreateSession", sim_CreateSession)
  SIM_FN("xrWaitFrame", sim_WaitFrame)
  SIM_FN("xrEndFrame", sim_EndFrame)
  SIM_FN("xrCreateSwapchain", sim_CreateSwapchain)
  SIM_FN("xrDestroySwapchain", sim_DestroySwapchain)
  SIM_FN("xrSuggestInteractionProfileBindings", sim_SuggestInteractionProfileBindings)
  SIM_FN("xrGetCurrentInteractionProfile", sim_GetCurrentInteractionProfile)
  SIM_FN("xrPollEvent", sim_PollEvent)
  SIM_FN("xrGetActionStateBoolean", sim_GetActionStateBoolean)
  SIM_FN("xrGetActionStateFloat", sim_GetActionStateFloat)
  SIM_FN("xrGetSystemProperties", sim_GetSystemProperties)
  if (S.hands) {
    SIM_FN("xrCreateHandTrackerEXT", sim_CreateHandTrackerEXT)
    SIM_FN("xrDestroyHandTrackerEXT", sim_DestroyHandTrackerEXT)
    SIM_FN("xrLocateHandJointsEXT", sim_LocateHandJointsEXT)
  }
  if (S.getRefreshRate != nullptr)
    SIM_FN("xrGetDisplayRefreshRateFB", sim_GetDisplayRefreshRateFB)
  if (S.enumerateRates != nullptr)
    SIM_FN("xrEnumerateDisplayRefreshRatesFB", sim_EnumerateDisplayRefreshRatesFB)
#undef SIM_FN
  return S.next(instance, name, function);
}

XRAPI_ATTR XrResult XRAPI_CALL sim_CreateApiLayerInstance(const XrInstanceCreateInfo* info,
                                                          const XrApiLayerCreateInfo* layerInfo, XrInstance* instance) {
  S.next = layerInfo->nextInfo->nextGetInstanceProcAddr;
  S.blend = env("AURORA_SIM_BLEND") == "alpha";
  S.frame = env_on("AURORA_SIM_FRAME");
  S.hands = env_on("AURORA_SIM_HANDS");
  S.maxSwapchains = std::atoi(env("AURORA_SIM_MAX_SWAPCHAINS").c_str());
  if (const std::string s = env("AURORA_SIM_FRAME_SHOULDER"); !s.empty())
    S.shoulder = s;
  S.periodMs = std::atof(env("AURORA_SIM_PERIOD_MS").c_str());
  if (const char* hz = std::getenv("AURORA_SIM_REPORTED_HZ"); hz != nullptr && *hz != '\0') {
    S.reportHz = true;
    S.reportedHz = static_cast<float>(std::atof(hz));
  }
  for (std::string rates = env("AURORA_SIM_RATES"); !rates.empty();) {
    const auto comma = rates.find(',');
    S.rates.push_back(static_cast<float>(std::atof(rates.substr(0, comma).c_str())));
    rates = comma == std::string::npos ? "" : rates.substr(comma + 1);
  }
  for (std::string presses = env("AURORA_SIM_PRESS"); !presses.empty();) {
    const auto semi = presses.find(';');
    const std::string p = presses.substr(0, semi);
    presses = semi == std::string::npos ? "" : presses.substr(semi + 1);
    const auto at = p.find('@'), dash = p.find('-', at);
    if (at != std::string::npos && dash != std::string::npos)
      S.presses.push_back({p.substr(0, at), std::strtoull(p.c_str() + at + 1, nullptr, 10),
                           std::strtoull(p.c_str() + dash + 1, nullptr, 10)});
  }

  // The runtime doesn't know the Frame's extension: it's this layer's.
  std::vector<const char*> exts;
  for (uint32_t i = 0; i < info->enabledExtensionCount; ++i)
    if (std::strcmp(info->enabledExtensionNames[i], kFrameExt) != 0)
      exts.push_back(info->enabledExtensionNames[i]);
  XrInstanceCreateInfo forwarded = *info;
  forwarded.enabledExtensionCount = static_cast<uint32_t>(exts.size());
  forwarded.enabledExtensionNames = exts.data();
  XrApiLayerCreateInfo nextLayer = *layerInfo;
  nextLayer.nextInfo = layerInfo->nextInfo->next;
  const XrResult r = layerInfo->nextInfo->nextCreateApiLayerInstance(&forwarded, &nextLayer, instance);
  if (r != XR_SUCCESS)
    return r;
  S.instance = *instance;
#define SIM_LOAD(member, api) load(api, reinterpret_cast<PFN_xrVoidFunction*>(&S.member))
  SIM_LOAD(destroyInstance, "xrDestroyInstance");
  SIM_LOAD(enumerateBlendModes, "xrEnumerateEnvironmentBlendModes");
  SIM_LOAD(createSession, "xrCreateSession");
  SIM_LOAD(endFrame, "xrEndFrame");
  SIM_LOAD(waitFrame, "xrWaitFrame");
  SIM_LOAD(createSwapchain, "xrCreateSwapchain");
  SIM_LOAD(destroySwapchain, "xrDestroySwapchain");
  SIM_LOAD(suggest, "xrSuggestInteractionProfileBindings");
  SIM_LOAD(currentProfile, "xrGetCurrentInteractionProfile");
  SIM_LOAD(pollEvent, "xrPollEvent");
  SIM_LOAD(pathToString, "xrPathToString");
  SIM_LOAD(stringToPath, "xrStringToPath");
  SIM_LOAD(actionBool, "xrGetActionStateBoolean");
  SIM_LOAD(actionFloat, "xrGetActionStateFloat");
  SIM_LOAD(systemProperties, "xrGetSystemProperties");
  SIM_LOAD(getRefreshRate, "xrGetDisplayRefreshRateFB");
  SIM_LOAD(enumerateRates, "xrEnumerateDisplayRefreshRatesFB");
#undef SIM_LOAD
  std::fprintf(stderr, "frame_sim: on (blend %s, frame %d/%s, hands %d, max swapchains %d, period %.3f ms)\n",
               S.blend ? "alpha" : "as is", S.frame, S.shoulder.c_str(), S.hands, S.maxSwapchains, S.periodMs);
  return XR_SUCCESS;
}

} // namespace

SIM_EXPORT XRAPI_ATTR XrResult XRAPI_CALL xrNegotiateLoaderApiLayerInterface(const XrNegotiateLoaderInfo* loaderInfo,
                                                                             const char* layerName,
                                                                             XrNegotiateApiLayerRequest* request) {
  if (layerName == nullptr || std::strcmp(layerName, kLayerName) != 0 || loaderInfo == nullptr || request == nullptr ||
      loaderInfo->structType != XR_LOADER_INTERFACE_STRUCT_LOADER_INFO ||
      request->structType != XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST)
    return XR_ERROR_INITIALIZATION_FAILED;
  request->layerInterfaceVersion = XR_CURRENT_LOADER_API_LAYER_VERSION;
  request->layerApiVersion = XR_CURRENT_API_VERSION;
  request->getInstanceProcAddr = sim_GetInstanceProcAddr;
  request->createApiLayerInstance = sim_CreateApiLayerInstance;
  return XR_SUCCESS;
}
