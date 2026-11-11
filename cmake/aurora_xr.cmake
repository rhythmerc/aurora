# OpenXR presentation (lib/xr): the game's frame goes to a virtual screen in
# the headset. Linux uses the system OpenXR loader when its SDK is new enough
# (e.g. Monado for desktop testing, enabled at runtime with AURORA_XR=1), else
# builds Khronos' at the Android loader's version. Android fetches Khronos'
# loader AAR, links it, stages libopenxr_loader.so next to the build output
# for packaging. XR stays off unless AURORA_XR=1 (melee-pc's immersive
# activity sets it, platforms/android/.../MeleeXrActivity.java), or unless the
# build is configured with AURORA_XR_DEFAULT_ON (a headset's own build, like
# the Steam Frame's, where AURORA_XR=0 turns it off).
include("${CMAKE_CURRENT_LIST_DIR}/AuroraDependencyVersions.cmake")

option(AURORA_XR_DEFAULT_ON "Present to OpenXR unless AURORA_XR=0 (a build for a headset)" OFF)
set(AURORA_OPENXR_PROVIDER "auto" CACHE STRING
  "Desktop OpenXR SDK: auto (the system's if at least AURORA_OPENXR_SDK_VERSION, else vendor), system, vendor (build Khronos' OpenXR-SDK)")
set_property(CACHE AURORA_OPENXR_PROVIDER PROPERTY STRINGS auto system vendor)

target_sources(aurora_core PRIVATE lib/xr/xr.cpp lib/xr/xr_policy.cpp lib/webgpu/queue_guard.cpp)
target_compile_definitions(aurora_core PUBLIC AURORA_ENABLE_OPENXR)
if (AURORA_XR_DEFAULT_ON)
  target_compile_definitions(aurora_core PRIVATE AURORA_XR_DEFAULT_ON)
endif ()

# One Vulkan queue shared by Dawn and the OpenXR runtime, where the GPU's
# graphics family has no spare (Turnip on the Steam Frame, RADV): every Dawn
# entry point that can reach the queue holds lib/webgpu/queue_guard.hpp's
# lock, for every caller (aurora, the ImGui and RmlUi backends, the game),
# by wrapping them at link time. Linux only; elsewhere a GPU without a spare
# queue keeps presenting to the window.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT ANDROID)
  target_compile_definitions(aurora_core PRIVATE AURORA_QUEUE_GUARD_WRAP)
  foreach (_sym IN ITEMS
      wgpuQueueSubmit wgpuQueueWriteBuffer wgpuQueueWriteTexture wgpuQueueOnSubmittedWorkDone
      wgpuBufferMapAsync wgpuBufferUnmap wgpuBufferDestroy
      wgpuInstanceProcessEvents wgpuInstanceWaitAny wgpuDeviceTick wgpuDeviceDestroy wgpuDeviceRelease
      wgpuSurfacePresent wgpuSurfaceGetCurrentTexture wgpuSurfaceConfigure wgpuSurfaceUnconfigure
      wgpuSharedTextureMemoryBeginAccess wgpuSharedTextureMemoryEndAccess)
    target_link_options(aurora_core INTERFACE "LINKER:--wrap=${_sym}")
  endforeach ()
endif ()

# Roomtone: the room capture's light estimation (github.com/rhythmerc/roomtone).
# ROOMTONE_SOURCE_DIR builds a local checkout instead.
if (NOT TARGET roomtone)
  set(ROOMTONE_SOURCE_DIR "" CACHE PATH "A local Roomtone checkout (empty: fetch the pinned one)")
  if (ROOMTONE_SOURCE_DIR)
    add_subdirectory("${ROOMTONE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_deps/roomtone-build" EXCLUDE_FROM_ALL)
  else ()
    include(FetchContent)
    FetchContent_Declare(roomtone
      GIT_REPOSITORY https://github.com/rhythmerc/roomtone.git
      GIT_TAG ${AURORA_ROOMTONE_COMMIT}
      EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(roomtone)
  endif ()
endif ()
target_link_libraries(aurora_core PRIVATE roomtone)

if (ANDROID)
  set(_xr_dir "${CMAKE_BINARY_DIR}/_deps/openxr_loader_android-${AURORA_OPENXR_ANDROID_VERSION}")
  set(_xr_aar "${_xr_dir}.aar")
  if (NOT EXISTS "${_xr_dir}/prefab")
    set(_xr_url "https://repo1.maven.org/maven2/org/khronos/openxr/openxr_loader_for_android/${AURORA_OPENXR_ANDROID_VERSION}/openxr_loader_for_android-${AURORA_OPENXR_ANDROID_VERSION}.aar")
    message(STATUS "aurora: fetching OpenXR loader ${AURORA_OPENXR_ANDROID_VERSION} for Android")
    file(DOWNLOAD "${_xr_url}" "${_xr_aar}" EXPECTED_HASH SHA256=${AURORA_OPENXR_ANDROID_SHA256} STATUS _xr_status)
    list(GET _xr_status 0 _xr_code)
    if (NOT _xr_code EQUAL 0)
      message(FATAL_ERROR "aurora: downloading ${_xr_url} failed: ${_xr_status}")
    endif ()
    file(ARCHIVE_EXTRACT INPUT "${_xr_aar}" DESTINATION "${_xr_dir}")
  endif ()
  set(_xr_so "${_xr_dir}/prefab/modules/openxr_loader/libs/android.${ANDROID_ABI}/libopenxr_loader.so")
  add_library(aurora_openxr_loader SHARED IMPORTED)
  set_target_properties(aurora_openxr_loader PROPERTIES
          IMPORTED_LOCATION "${_xr_so}"
          IMPORTED_NO_SONAME TRUE
          INTERFACE_INCLUDE_DIRECTORIES "${_xr_dir}/prefab/modules/headers/include")
  # tools/build_android.sh copies it into jniLibs beside libmelee.so.
  configure_file("${_xr_so}" "${CMAKE_BINARY_DIR}/libopenxr_loader.so" COPYONLY)
  target_link_libraries(aurora_core PRIVATE aurora_openxr_loader vulkan nativewindow android)
else ()
  # xr.cpp uses extensions newer than some distributions' SDKs (Nix has
  # 1.1.54, without XR_META_environment_raycast's types): the system's only
  # when it is at least the Android loader's version.
  if (NOT AURORA_OPENXR_PROVIDER STREQUAL "vendor")
    find_package(OpenXR ${AURORA_OPENXR_SDK_VERSION} CONFIG QUIET)
  endif ()
  if (NOT TARGET OpenXR::openxr_loader)
    if (AURORA_OPENXR_PROVIDER STREQUAL "system")
      message(FATAL_ERROR "aurora: AURORA_OPENXR_PROVIDER=system, but no OpenXR SDK ${AURORA_OPENXR_SDK_VERSION} or newer was found")
    endif ()
    message(STATUS "aurora: building Khronos' OpenXR loader ${AURORA_OPENXR_SDK_VERSION} (provider=${AURORA_OPENXR_PROVIDER})")
    include(FetchContent)
    FetchContent_Declare(openxr_sdk
      URL "https://github.com/KhronosGroup/OpenXR-SDK/archive/refs/tags/release-${AURORA_OPENXR_SDK_VERSION}.tar.gz"
      URL_HASH SHA256=${AURORA_OPENXR_SDK_SHA256}
      DOWNLOAD_EXTRACT_TIMESTAMP FALSE
      EXCLUDE_FROM_ALL
    )
    block()
      # The loader as a shared library (libopenxr_loader.so.1, shipped beside
      # the game for systems without one) with its own jsoncpp.
      set(DYNAMIC_LOADER ON)
      set(BUILD_WITH_SYSTEM_JSONCPP OFF)
      FetchContent_MakeAvailable(openxr_sdk)
    endblock()
  endif ()
  find_package(Vulkan REQUIRED)
  target_link_libraries(aurora_core PRIVATE OpenXR::openxr_loader Vulkan::Vulkan)
endif ()
