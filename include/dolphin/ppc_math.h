#ifndef DOLPHIN_PPC_MATH_H
#define DOLPHIN_PPC_MATH_H

#include <float.h>
#include <math.h>
#include <stdint.h>

// frsqrte matching courtesy of Geotale, with reference to https://achurch.org/cpu-tests/ppc750cl.s

struct BaseAndDec32 {
    uint32_t base;
    int32_t dec;
};

struct BaseAndDec64 {
    uint64_t base;
    int64_t dec;
};

union c32 {
    uint32_t u;
    float f;
};

union c64 {
    uint64_t u;
    double f;
};

#define EXPONENT_SHIFT_F64 ((uint64_t)52)
#define MANTISSA_MASK_F64  ((uint64_t)0x000fffffffffffffULL)
#define EXPONENT_MASK_F64  ((uint64_t)0x7ff0000000000000ULL)
#define SIGN_MASK_F64      ((uint64_t)0x8000000000000000ULL)

static const struct BaseAndDec64 RSQRTE_TABLE[32] = {
    {0x69fa000000000ULL, -0x15a0000000LL},
    {0x5f2e000000000ULL, -0x13cc000000LL},
    {0x554a000000000ULL, -0x1234000000LL},
    {0x4c30000000000ULL, -0x10d4000000LL},
    {0x43c8000000000ULL, -0x0f9c000000LL},
    {0x3bfc000000000ULL, -0x0e88000000LL},
    {0x34b8000000000ULL, -0x0d94000000LL},
    {0x2df0000000000ULL, -0x0cb8000000LL},
    {0x2794000000000ULL, -0x0bf0000000LL},
    {0x219c000000000ULL, -0x0b40000000LL},
    {0x1bfc000000000ULL, -0x0aa0000000LL},
    {0x16ae000000000ULL, -0x0a0c000000LL},
    {0x11a8000000000ULL, -0x0984000000LL},
    {0x0ce6000000000ULL, -0x090c000000LL},
    {0x0862000000000ULL, -0x0898000000LL},
    {0x0416000000000ULL, -0x082c000000LL},
    {0xffe8000000000ULL, -0x1e90000000LL},
    {0xf0a4000000000ULL, -0x1c00000000LL},
    {0xe2a8000000000ULL, -0x19c0000000LL},
    {0xd5c8000000000ULL, -0x17c8000000LL},
    {0xc9e4000000000ULL, -0x1610000000LL},
    {0xbedc000000000ULL, -0x1490000000LL},
    {0xb498000000000ULL, -0x1330000000LL},
    {0xab00000000000ULL, -0x11f8000000LL},
    {0xa204000000000ULL, -0x10e8000000LL},
    {0x9994000000000ULL, -0x0fe8000000LL},
    {0x91a0000000000ULL, -0x0f08000000LL},
    {0x8a1c000000000ULL, -0x0e38000000LL},
    {0x8304000000000ULL, -0x0d78000000LL},
    {0x7c48000000000ULL, -0x0cc8000000LL},
    {0x75e4000000000ULL, -0x0c28000000LL},
    {0x6fd0000000000ULL, -0x0b98000000LL},
};

#ifdef _MSC_VER
#include <intrin.h>
static inline uint32_t ppc_clz64(uint64_t x) {
    unsigned long idx;
    _BitScanReverse64(&idx, x);
    return 63u - (uint32_t)idx;
}
#else
static inline uint32_t ppc_clz64(uint64_t x) {
    return (uint32_t)__builtin_clzll(x);
}
#endif

static inline double frsqrte(double val) {
    union c64 bits;
    uint64_t mantissa;
    int64_t exponent;
    int sign;
    uint32_t key;
    uint64_t new_exp;
    const struct BaseAndDec64 *entry;
    union c64 result;

    bits.f = val;
    mantissa = bits.u & MANTISSA_MASK_F64;
    exponent = (int64_t)(bits.u & EXPONENT_MASK_F64);
    sign = (bits.u & SIGN_MASK_F64) != 0;

    if (mantissa == 0 && exponent == 0) {
        return copysign(INFINITY, bits.f);
    }

    if ((uint64_t)exponent == EXPONENT_MASK_F64) {
        if (mantissa == 0) {
            return sign ? NAN : 0.0;
        }
        return val;
    }

    if (sign) {
        return NAN;
    }

    if (exponent == 0) {
        uint32_t shift = ppc_clz64(mantissa) - (uint32_t)(63 - EXPONENT_SHIFT_F64);
        mantissa <<= shift;
        mantissa &= MANTISSA_MASK_F64;
        exponent -= (int64_t)(shift - 1) << EXPONENT_SHIFT_F64;
    }

    key = (uint32_t)(((uint64_t)exponent | mantissa) >> 37);
    new_exp = (((0xbfcULL << EXPONENT_SHIFT_F64) - (uint64_t)exponent) >> 1) & EXPONENT_MASK_F64;

    entry = &RSQRTE_TABLE[0x1fu & (key >> 11)];
    result.u = new_exp | (uint64_t)(entry->base + entry->dec * (int64_t)(key & 0x7ffu));
    return result.f;
}

// fres: the hardware's reciprocal estimate, as Dolphin computes it
// (Common::ApproximateReciprocal), a value with single-precision bits.
static const struct BaseAndDec32 FRES_TABLE[32] = {
    {0x7ff800, 0x3e1}, {0x783800, 0x3a7}, {0x70ea00, 0x371}, {0x6a0800, 0x340},
    {0x638800, 0x313}, {0x5d6200, 0x2ea}, {0x579000, 0x2c4}, {0x520800, 0x2a0},
    {0x4cc800, 0x27f}, {0x47ca00, 0x261}, {0x430800, 0x245}, {0x3e8000, 0x22a},
    {0x3a2c00, 0x212}, {0x360800, 0x1fb}, {0x321400, 0x1e5}, {0x2e4a00, 0x1d1},
    {0x2aa800, 0x1be}, {0x272c00, 0x1ac}, {0x23d600, 0x19b}, {0x209e00, 0x18b},
    {0x1d8800, 0x17c}, {0x1a9000, 0x16e}, {0x17ae00, 0x15b}, {0x14f800, 0x15b},
    {0x124400, 0x143}, {0x0fbe00, 0x143}, {0x0d3800, 0x12d}, {0x0ade00, 0x12d},
    {0x088400, 0x11a}, {0x065000, 0x11a}, {0x041c00, 0x108}, {0x020c00, 0x106},
};

static inline double ppc_fres(double val) {
    union c64 bits;
    uint64_t mantissa, sign, exponent;
    bits.f = val;
    mantissa = bits.u & MANTISSA_MASK_F64;
    sign = bits.u & SIGN_MASK_F64;
    exponent = bits.u & EXPONENT_MASK_F64;
    if (mantissa == 0 && exponent == 0) {
        return sign ? -INFINITY : INFINITY;
    }
    if (exponent == EXPONENT_MASK_F64) {
        if (mantissa == 0) {
            return sign ? -0.0 : 0.0;
        }
        return 0.0 + val;
    }
    if (exponent < (895ULL << EXPONENT_SHIFT_F64)) {
        return sign ? -FLT_MAX : FLT_MAX;
    }
    if (exponent >= (1149ULL << EXPONENT_SHIFT_F64)) {
        return sign ? -0.0 : 0.0;
    }
    exponent = (0x7FDULL << EXPONENT_SHIFT_F64) - exponent;
    {
        const int i = (int)(mantissa >> 37);
        const struct BaseAndDec32* entry = &FRES_TABLE[i / 1024];
        union c64 r;
        r.u = sign | exponent;
        r.u |= (uint64_t)(entry->base - (entry->dec * (i % 1024) + 1) / 2) << 29;
        return r.f;
    }
}

// Gekko's single-precision ops as Dolphin computes them: in double, then
// rounded to single. A multiply's second operand (frC) held in double
// precision goes in rounded to the 25 bits the hardware's multiplier uses
// (Dolphin's Force25Bit); a single-precision value passes unchanged. The
// fused ops round the sum to double, then to single (an fma in double,
// then ForceSingle; with single inputs the product is exact in double).
// Where the JIT and the hardware part (the sign of a zero), the JIT's.
static inline double ppc_force25(double d) {
    union c64 b;
    b.f = d;
    b.u = (b.u & 0xFFFFFFFFF8000000ULL) + (b.u & 0x8000000ULL);
    return b.f;
}
static inline float ppc_fmadds(float a, float c, float b) { // a*c + b
    return (float)((double)a * (double)c + (double)b);
}
static inline float ppc_fmsubs(float a, float c, float b) { // a*c - b
    return (float)((double)a * (double)c - (double)b);
}
static inline float ppc_fnmadds(float a, float c, float b) { // -(a*c + b)
    return (float)-((double)a * (double)c + (double)b);
}
// -(a*c - b), computed as b - a*c: what Dolphin's JIT does (x86's fnmadd,
// AArch64's fmsub), which a Slippi peer runs. The two part only in the
// sign of a zero (0 - 0*k is +0 here, -(0*k - 0) is -0 on the hardware), and
// replays show Dolphin's: an airborne knockback y of 0 stays +0 as it
// decays, so atan2f of it later sees +0.
static inline float ppc_fnmsubs(float a, float c, float b) {
    return (float)((double)b - (double)a * (double)c);
}

// The double-precision ones (fmadd, fmsub, fnmadd, fnmsub) are one fma:
// Dolphin computes them so, rounding once to double.
static inline double ppc_fmadd(double a, double c, double b) { return fma(a, c, b); }
static inline double ppc_fmsub(double a, double c, double b) { return fma(a, c, -b); }
static inline double ppc_fnmadd(double a, double c, double b) { return -fma(a, c, b); }
static inline double ppc_fnmsub(double a, double c, double b) { return -fma(a, c, -b); }

// The SDK's reciprocal square root (PSVECNormalize, PSVECMag): frsqrte and
// one Newton-Raphson step, as the disc's code has it (0x80342DD4..DE4):
// fmuls e*e (frC = e, the estimate, in double), fmuls e*0.5, fnmsubs
// 3 - e^2*x, fmuls.
static inline float ppc_rsqrte(float x) {
    double rsqrt_d = frsqrte((double)x);
    float nwork0 = (float)(rsqrt_d * ppc_force25(rsqrt_d));
    float nwork1 = (float)(rsqrt_d * 0.5);
    nwork0 = ppc_fnmsubs(nwork0, x, 3.0f);
    return nwork0 * nwork1;
}

#endif // DOLPHIN_PPC_MATH_H
