# SDL lists the Wii U / Switch GameCube adapter (057e:0337) as reachable only
# through libusb, and every platform HID backend skips it
# (SDL_HIDAPI_ShouldIgnoreDevice with libusb=false). That holds where libusb
# is compiled in, but Android never has libusb: its Java HIDDeviceManager opens
# the adapter's HID interface and streams its reports, and then the native side
# drops it from every enumeration, so nothing can see the adapter at all.
# Keep the skip only when HAVE_LIBUSB is there to take the adapter instead.
# Narrowed to the adapter: the other libusb-only devices (Switch 2 pads, the
# Xbox 360 Big Button receiver) keep SDL's behaviour.
if (NOT DEFINED SDL_SOURCE_DIR)
  message(FATAL_ERROR "SDL_SOURCE_DIR is required")
endif ()

set(_hidapi "${SDL_SOURCE_DIR}/src/hidapi/SDL_hidapi.c")
if (NOT EXISTS "${_hidapi}")
  message(FATAL_ERROR "SDL hidapi source is missing: ${_hidapi}")
endif ()

file(READ "${_hidapi}" _source)

set(_old "    } else {
        if (RequiresLibUSB(vendor_id, product_id, libusb_xbox)) {
            return true;
        }
    }
")
set(_new "    } else {
        if (RequiresLibUSB(vendor_id, product_id, libusb_xbox)
#ifndef HAVE_LIBUSB
            // Without libusb the platform backend is the adapter's only way in
            && !(vendor_id == USB_VENDOR_NINTENDO && product_id == USB_PRODUCT_NINTENDO_GAMECUBE_ADAPTER)
#endif
        ) {
            return true;
        }
    }
")

string(FIND "${_source}" "${_new}" _already_patched)
if (NOT _already_patched EQUAL -1)
  message(STATUS "SDL3 GameCube adapter without libusb patch already applied")
  return()
endif ()

string(FIND "${_source}" "${_old}" _patch_site)
if (_patch_site EQUAL -1)
  message(FATAL_ERROR "Failed to apply SDL3 GameCube adapter without libusb patch: expected SDL_HIDAPI_ShouldIgnoreDevice branch not found")
endif ()

string(REPLACE "${_old}" "${_new}" _patched_source "${_source}")
file(WRITE "${_hidapi}" "${_patched_source}")
message(STATUS "Applied SDL3 GameCube adapter without libusb patch")
