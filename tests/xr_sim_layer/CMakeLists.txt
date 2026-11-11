# Test only: an OpenXR API layer giving a desktop runtime (Monado) the Steam
# Frame's SteamVR personality (frame_sim_layer.cpp). Enable it with
#   XR_API_LAYER_PATH=<this build directory> XR_ENABLE_API_LAYERS=XR_APILAYER_AURORA_frame_sim
add_library(XrApiLayer_aurora_frame_sim MODULE frame_sim_layer.cpp)
target_link_libraries(XrApiLayer_aurora_frame_sim PRIVATE OpenXR::headers)
set_target_properties(XrApiLayer_aurora_frame_sim PROPERTIES CXX_VISIBILITY_PRESET hidden)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/XrApiLayer_aurora_frame_sim.json"
     INPUT "${CMAKE_CURRENT_SOURCE_DIR}/XrApiLayer_aurora_frame_sim.json.in")
