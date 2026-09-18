#ifndef AURORA_REPLACEMENT_H
#define AURORA_REPLACEMENT_H

/* smstrikers-port: the replacement folders and texture dumps of <aurora/texture.hpp>, for C callers. */

#include <dolphin/gx/GXStruct.h>

#ifdef __cplusplus
#include <cstdint>

extern "C" {
#else
#include <stdint.h>
#endif

/* Registers every replacement file under `root`, a UTF-8 path that must exist; returns how many. */
uint32_t aurora_replacement_load(const char* root, int32_t priority);
void aurora_replacement_clear(void);
void aurora_replacement_set_cache_budget(uint64_t bytes);
/* aurora::texture::dump_texture: 1 written, 0 already there, -1 failed. */
int aurora_replacement_dump(const GXTexObj* obj, const GXTlutObj* tlut, const char* dir);

#ifdef __cplusplus
}
#endif

#endif
