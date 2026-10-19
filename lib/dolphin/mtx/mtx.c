#include <dolphin/ppc_math.h>
#include <dolphin/mtx.h>

#include <assert.h>
#include <math.h>

void C_MTXIdentity(Mtx m) {
  assert(m && "MtxIdentity():  NULL Mtx 'm' ");
  m[0][0] = 1;
  m[0][1] = 0;
  m[0][2] = 0;
  m[0][3] = 0;
  m[1][0] = 0;
  m[1][1] = 1;
  m[1][2] = 0;
  m[1][3] = 0;
  m[2][0] = 0;
  m[2][1] = 0;
  m[2][2] = 1;
  m[2][3] = 0;
}

void C_MTXCopy(const Mtx src, Mtx dst) {
  assert(src && "MTXCopy():  NULL MtxPtr 'src' ");
  assert(dst && "MTXCopy():  NULL MtxPtr 'dst' ");
  if (src != dst) {
    dst[0][0] = src[0][0];
    dst[0][1] = src[0][1];
    dst[0][2] = src[0][2];
    dst[0][3] = src[0][3];
    dst[1][0] = src[1][0];
    dst[1][1] = src[1][1];
    dst[1][2] = src[1][2];
    dst[1][3] = src[1][3];
    dst[2][0] = src[2][0];
    dst[2][1] = src[2][1];
    dst[2][2] = src[2][2];
    dst[2][3] = src[2][3];
  }
}

void C_MTXConcat(const Mtx a, const Mtx b, Mtx ab) {
  Mtx mTmp;
  MtxPtr m;

  assert(a && "MTXConcat():  NULL MtxPtr 'a'  ");
  assert(b && "MTXConcat():  NULL MtxPtr 'b'  ");
  assert(ab && "MTXConcat():  NULL MtxPtr 'ab' ");

  if (ab == a || ab == b) {
    m = mTmp;
  } else {
    m = ab;
  }

  m[0][0] =       0 +  a[0][2] * b[2][0] + ((a[0][0] * b[0][0]) + (a[0][1] * b[1][0]));
  m[0][1] =       0 +  a[0][2] * b[2][1] + ((a[0][0] * b[0][1]) + (a[0][1] * b[1][1]));
  m[0][2] =       0 +  a[0][2] * b[2][2] + ((a[0][0] * b[0][2]) + (a[0][1] * b[1][2]));
  m[0][3] = a[0][3] + (a[0][2] * b[2][3] +  (a[0][0] * b[0][3]  + (a[0][1] * b[1][3])));

  m[1][0] =       0 +  a[1][2] * b[2][0] + ((a[1][0] * b[0][0]) + (a[1][1] * b[1][0]));
  m[1][1] =       0 +  a[1][2] * b[2][1] + ((a[1][0] * b[0][1]) + (a[1][1] * b[1][1]));
  m[1][2] =       0 +  a[1][2] * b[2][2] + ((a[1][0] * b[0][2]) + (a[1][1] * b[1][2]));
  m[1][3] = a[1][3] + (a[1][2] * b[2][3] +  (a[1][0] * b[0][3]  + (a[1][1] * b[1][3])));

  m[2][0] =       0 +  a[2][2] * b[2][0] + ((a[2][0] * b[0][0]) + (a[2][1] * b[1][0]));
  m[2][1] =       0 +  a[2][2] * b[2][1] + ((a[2][0] * b[0][1]) + (a[2][1] * b[1][1]));
  m[2][2] =       0 +  a[2][2] * b[2][2] + ((a[2][0] * b[0][2]) + (a[2][1] * b[1][2]));
  m[2][3] = a[2][3] + (a[2][2] * b[2][3] +  (a[2][0] * b[0][3]  + (a[2][1] * b[1][3])));

  if (m == mTmp) {
    C_MTXCopy(mTmp, ab);
  }
}

void C_MTXConcatArray(const Mtx a, const Mtx* srcBase, Mtx* dstBase, u32 count) {
  u32 i;

  assert(a != 0 && "MTXConcatArray(): NULL MtxPtr 'a' ");
  assert(srcBase != 0 && "MTXConcatArray(): NULL MtxPtr 'srcBase' ");
  assert(dstBase != 0 && "MTXConcatArray(): NULL MtxPtr 'dstBase' ");
  assert(count > 1 && "MTXConcatArray(): count must be greater than 1.");

  for (i = 0; i < count; i++) {
    C_MTXConcat(a, *srcBase, *dstBase);
    srcBase++;
    dstBase++;
  }
}

void C_MTXTranspose(const Mtx src, Mtx xPose) {
  Mtx mTmp;
  MtxPtr m;

  assert(src && "MTXTranspose():  NULL MtxPtr 'src' ");
  assert(xPose && "MTXTranspose():  NULL MtxPtr 'xPose' ");

  if (src == xPose) {
    m = mTmp;
  } else {
    m = xPose;
  }

  m[0][0] = src[0][0];
  m[0][1] = src[1][0];
  m[0][2] = src[2][0];
  m[0][3] = 0;
  m[1][0] = src[0][1];
  m[1][1] = src[1][1];
  m[1][2] = src[2][1];
  m[1][3] = 0;
  m[2][0] = src[0][2];
  m[2][1] = src[1][2];
  m[2][2] = src[2][2];
  m[2][3] = 0;
  if (m == mTmp) {
    C_MTXCopy(mTmp, xPose);
  }
}

u32 C_MTXInverse(const Mtx src, Mtx inv) {
  Mtx mTmp;
  MtxPtr m;
  f32 det;

  assert(src && "MTXInverse():  NULL MtxPtr 'src' ");
  assert(inv && "MTXInverse():  NULL MtxPtr 'inv' ");

  if (src == inv) {
    m = mTmp;
  } else {
    m = inv;
  }
  det = ((((src[2][1] * (src[0][2] * src[1][0]))
        + ((src[2][2] * (src[0][0] * src[1][1]))
         + (src[2][0] * (src[0][1] * src[1][2]))))
         - (src[0][2] * (src[2][0] * src[1][1])))
         - (src[2][2] * (src[1][0] * src[0][1])))
         - (src[1][2] * (src[0][0] * src[2][1]));
  if (0 == det) {
    return 0;
  }
  det = 1 / det;
  m[0][0] = (det * +((src[1][1] * src[2][2]) - (src[2][1] * src[1][2])));
  m[0][1] = (det * -((src[0][1] * src[2][2]) - (src[2][1] * src[0][2])));
  m[0][2] = (det * +((src[0][1] * src[1][2]) - (src[1][1] * src[0][2])));

  m[1][0] = (det * -((src[1][0] * src[2][2]) - (src[2][0] * src[1][2])));
  m[1][1] = (det * +((src[0][0] * src[2][2]) - (src[2][0] * src[0][2])));
  m[1][2] = (det * -((src[0][0] * src[1][2]) - (src[1][0] * src[0][2])));

  m[2][0] = (det * +((src[1][0] * src[2][1]) - (src[2][0] * src[1][1])));
  m[2][1] = (det * -((src[0][0] * src[2][1]) - (src[2][0] * src[0][1])));
  m[2][2] = (det * +((src[0][0] * src[1][1]) - (src[1][0] * src[0][1])));

  m[0][3] = ((-m[0][0] * src[0][3]) - (m[0][1] * src[1][3])) - (m[0][2] * src[2][3]);
  m[1][3] = ((-m[1][0] * src[0][3]) - (m[1][1] * src[1][3])) - (m[1][2] * src[2][3]);
  m[2][3] = ((-m[2][0] * src[0][3]) - (m[2][1] * src[1][3])) - (m[2][2] * src[2][3]);

  if (m == mTmp) {
    C_MTXCopy(mTmp, inv);
  }
  return 1;
}

u32 C_MTXInvXpose(const Mtx src, Mtx invX) {
  Mtx mTmp;
  MtxPtr m;
  f32 det;

  assert(src && "MTXInvXpose(): NULL MtxPtr 'src' ");
  assert(invX && "MTXInvXpose(): NULL MtxPtr 'invX' ");

  if (src == invX) {
    m = mTmp;
  } else {
    m = invX;
  }
  det = ((((src[2][1] * (src[0][2] * src[1][0]))
        + ((src[2][2] * (src[0][0] * src[1][1]))
        +  (src[2][0] * (src[0][1] * src[1][2]))))
        -  (src[0][2] * (src[2][0] * src[1][1])))
        -  (src[2][2] * (src[1][0] * src[0][1])))
        -  (src[1][2] * (src[0][0] * src[2][1]));
  if (0 == det) {
    return 0;
  }
  det = 1 / det;
  m[0][0] = (det * +((src[1][1] * src[2][2]) - (src[2][1] * src[1][2])));
  m[0][1] = (det * -((src[1][0] * src[2][2]) - (src[2][0] * src[1][2])));
  m[0][2] = (det * +((src[1][0] * src[2][1]) - (src[2][0] * src[1][1])));

  m[1][0] = (det * -((src[0][1] * src[2][2]) - (src[2][1] * src[0][2])));
  m[1][1] = (det * +((src[0][0] * src[2][2]) - (src[2][0] * src[0][2])));
  m[1][2] = (det * -((src[0][0] * src[2][1]) - (src[2][0] * src[0][1])));

  m[2][0] = (det * +((src[0][1] * src[1][2]) - (src[1][1] * src[0][2])));
  m[2][1] = (det * -((src[0][0] * src[1][2]) - (src[1][0] * src[0][2])));
  m[2][2] = (det * +((src[0][0] * src[1][1]) - (src[1][0] * src[0][1])));

  m[0][3] = 0;
  m[1][3] = 0;
  m[2][3] = 0;

  if (m == mTmp) {
    C_MTXCopy(mTmp, invX);
  }
  return 1;
}

void C_MTXTrans(Mtx m, f32 xT, f32 yT, f32 zT) {
  assert(m && "MTXTrans():  NULL MtxPtr 'm' ");
  m[0][0] = 1;
  m[0][1] = 0;
  m[0][2] = 0;
  m[0][3] = xT;
  m[1][0] = 0;
  m[1][1] = 1;
  m[1][2] = 0;
  m[1][3] = yT;
  m[2][0] = 0;
  m[2][1] = 0;
  m[2][2] = 1;
  m[2][3] = zT;
}

void C_MTXFrustum(Mtx44 m, f32 t, f32 b, f32 l, f32 r, f32 n, f32 f) {
  f32 tmp;

  assert(m && "MTXFrustum():  NULL Mtx44Ptr 'm' ");
  assert(t != b && "MTXFrustum():  't' and 'b' clipping planes are equal ");
  assert(l != r && "MTXFrustum():  'l' and 'r' clipping planes are equal ");
  assert(n != f && "MTXFrustum():  'n' and 'f' clipping planes are equal ");
  tmp = 1 / (r - l);
  m[0][0] = (2 * n * tmp);
  m[0][1] = 0;
  m[0][2] = (tmp * (r + l));
  m[0][3] = 0;
  tmp = 1 / (t - b);
  m[1][0] = 0;
  m[1][1] = (2 * n * tmp);
  m[1][2] = (tmp * (t + b));
  m[1][3] = 0;
  m[2][0] = 0;
  m[2][1] = 0;
  tmp = 1 / (f - n);
  m[2][2] = (-n * tmp);
  m[2][3] = (tmp * -(f * n));
  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = -1;
  m[3][3] = 0;
}

void C_MTXOrtho(Mtx44 m, f32 t, f32 b, f32 l, f32 r, f32 n, f32 f) {
  f32 tmp;

  assert(m && "MTXOrtho():  NULL Mtx44Ptr 'm' ");
  assert(t != b && "MTXOrtho():  't' and 'b' clipping planes are equal ");
  assert(l != r && "MTXOrtho():  'l' and 'r' clipping planes are equal ");
  assert(n != f && "MTXOrtho():  'n' and 'f' clipping planes are equal ");
  tmp = 1 / (r - l);
  m[0][0] = 2 * tmp;
  m[0][1] = 0;
  m[0][2] = 0;
  m[0][3] = (tmp * -(r + l));
  tmp = 1 / (t - b);
  m[1][0] = 0;
  m[1][1] = 2 * tmp;
  m[1][2] = 0;
  m[1][3] = (tmp * -(t + b));
  m[2][0] = 0;
  m[2][1] = 0;
  tmp = 1 / (f - n);
  m[2][2] = (-1 * tmp);
  m[2][3] = (-f * tmp);
  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = 0;
  m[3][3] = 1;
}

void C_MTXQuat(Mtx m, const Quaternion* q) {
  f32 s;
  f32 xs;
  f32 ys;
  f32 zs;
  f32 wx;
  f32 wy;
  f32 wz;
  f32 xx;
  f32 xy;
  f32 xz;
  f32 yy;
  f32 yz;
  f32 zz;

  assert(m && "MTXQuat():  NULL MtxPtr 'm' ");
  assert(q && "MTXQuat():  NULL QuaternionPtr 'q' ");
  assert(q->x || q->y || q->z || q->w && "MTXQuat():  zero-value quaternion ");
  s = 2 / ((q->w * q->w) + ((q->z * q->z) + ((q->x * q->x) + (q->y * q->y))));
  xs = q->x * s;
  ys = q->y * s;
  zs = q->z * s;
  wx = q->w * xs;
  wy = q->w * ys;
  wz = q->w * zs;
  xx = q->x * xs;
  xy = q->x * ys;
  xz = q->x * zs;
  yy = q->y * ys;
  yz = q->y * zs;
  zz = q->z * zs;
  m[0][0] = (1 - (yy + zz));
  m[0][1] = (xy - wz);
  m[0][2] = (xz + wy);
  m[0][3] = 0;
  m[1][0] = (xy + wz);
  m[1][1] = (1 - (xx + zz));
  m[1][2] = (yz - wx);
  m[1][3] = 0;
  m[2][0] = (xz - wy);
  m[2][1] = (yz + wx);
  m[2][2] = (1 - (xx + yy));
  m[2][3] = 0;
}

void C_MTXReflect(Mtx m, const Vec* p, const Vec* n) {
  f32 vxy;
  f32 vxz;
  f32 vyz;
  f32 pdotn;

  vxy = -2 * n->x * n->y;
  vxz = -2 * n->x * n->z;
  vyz = -2 * n->y * n->z;
  pdotn = 2 * C_VECDotProduct(p, n);
  m[0][0] = (1 - (2 * n->x * n->x));
  m[0][1] = vxy;
  m[0][2] = vxz;
  m[0][3] = (pdotn * n->x);
  m[1][0] = vxy;
  m[1][1] = (1 - (2 * n->y * n->y));
  m[1][2] = vyz;
  m[1][3] = (pdotn * n->y);
  m[2][0] = vxz;
  m[2][1] = vyz;
  m[2][2] = (1 - (2 * n->z * n->z));
  m[2][3] = (pdotn * n->z);
}

void C_MTXTransApply(const Mtx src, Mtx dst, f32 xT, f32 yT, f32 zT) {
  assert(src && "MTXTransApply(): NULL MtxPtr 'src' ");
  assert(dst && "MTXTransApply(): NULL MtxPtr 'src' "); //! wrong assert string

  if (src != dst) {
    dst[0][0] = src[0][0];
    dst[0][1] = src[0][1];
    dst[0][2] = src[0][2];
    dst[1][0] = src[1][0];
    dst[1][1] = src[1][1];
    dst[1][2] = src[1][2];
    dst[2][0] = src[2][0];
    dst[2][1] = src[2][1];
    dst[2][2] = src[2][2];
  }

  dst[0][3] = (src[0][3] + xT);
  dst[1][3] = (src[1][3] + yT);
  dst[2][3] = (src[2][3] + zT);
}

void C_MTXScale(Mtx m, f32 xS, f32 yS, f32 zS) {
  assert(m && "MTXScale():  NULL MtxPtr 'm' ");
  m[0][0] = xS;
  m[0][1] = 0;
  m[0][2] = 0;
  m[0][3] = 0;
  m[1][0] = 0;
  m[1][1] = yS;
  m[1][2] = 0;
  m[1][3] = 0;
  m[2][0] = 0;
  m[2][1] = 0;
  m[2][2] = zS;
  m[2][3] = 0;
}

void C_MTXScaleApply(const Mtx src, Mtx dst, f32 xS, f32 yS, f32 zS) {
  assert(src && "MTXScaleApply(): NULL MtxPtr 'src' ");
  assert(dst && "MTXScaleApply(): NULL MtxPtr 'dst' ");
  dst[0][0] = (src[0][0] * xS);
  dst[0][1] = (src[0][1] * xS);
  dst[0][2] = (src[0][2] * xS);
  dst[0][3] = (src[0][3] * xS);
  dst[1][0] = (src[1][0] * yS);
  dst[1][1] = (src[1][1] * yS);
  dst[1][2] = (src[1][2] * yS);
  dst[1][3] = (src[1][3] * yS);
  dst[2][0] = (src[2][0] * zS);
  dst[2][1] = (src[2][1] * zS);
  dst[2][2] = (src[2][2] * zS);
  dst[2][3] = (src[2][3] * zS);
}

void C_MTXRotRad(Mtx m, char axis, f32 rad) {
  f32 sinA;
  f32 cosA;

  assert(m && "MTXRotRad():  NULL MtxPtr 'm' ");
  sinA = sinf(rad);
  cosA = cosf(rad);
  C_MTXRotTrig(m, axis, sinA, cosA);
}

void C_MTXRotTrig(Mtx m, char axis, f32 sinA, f32 cosA) {
  assert(m && "MTXRotTrig():  NULL MtxPtr 'm' ");
  switch(axis) {
  case 'x':
  case 'X':
    m[0][0] = 1;
    m[0][1] = 0;
    m[0][2] = 0;
    m[0][3] = 0;
    m[1][0] = 0;
    m[1][1] = cosA;
    m[1][2] = -sinA;
    m[1][3] = 0;
    m[2][0] = 0;
    m[2][1] = sinA;
    m[2][2] = cosA;
    m[2][3] = 0;
    break;
  case 'y':
  case 'Y':
    m[0][0] = cosA;
    m[0][1] = 0;
    m[0][2] = sinA;
    m[0][3] = 0;
    m[1][0] = 0;
    m[1][1] = 1;
    m[1][2] = 0;
    m[1][3] = 0;
    m[2][0] = -sinA;
    m[2][1] = 0;
    m[2][2] = cosA;
    m[2][3] = 0;
    break;
  case 'z':
  case 'Z':
    m[0][0] = cosA;
    m[0][1] = -sinA;
    m[0][2] = 0;
    m[0][3] = 0;
    m[1][0] = sinA;
    m[1][1] = cosA;
    m[1][2] = 0;
    m[1][3] = 0;
    m[2][0] = 0;
    m[2][1] = 0;
    m[2][2] = 1;
    m[2][3] = 0;
    break;
  default:
    assert(0 && "MTXRotTrig():  invalid 'axis' value ");
    break;
  }
}

void C_MTXRotAxisRad(Mtx m, const Vec* axis, f32 rad) {
  Vec vN;
  f32 s;
  f32 c;
  f32 t;
  f32 x;
  f32 y;
  f32 z;
  f32 xSq;
  f32 ySq;
  f32 zSq;

  assert(m && "MTXRotAxisRad():  NULL MtxPtr 'm' ");
  assert(axis && "MTXRotAxisRad():  NULL VecPtr 'axis' ");

  s = sinf(rad);
  c = cosf(rad);
  t = 1 - c;
  C_VECNormalize(axis, &vN);
  x = vN.x;
  y = vN.y;
  z = vN.z;
  xSq = (x * x);
  ySq = (y * y);
  zSq = (z * z);
  m[0][0] = (c + (t * xSq));
  m[0][1] = (y * (t * x)) - (s * z);
  m[0][2] = (z * (t * x)) + (s * y);
  m[0][3] = 0;
  m[1][0] = ((y * (t * x)) + (s * z));
  m[1][1] = (c + (t * ySq));
  m[1][2] = ((z * (t * y)) - (s * x));
  m[1][3] = 0;
  m[2][0] = ((z * (t * x)) - (s * y));
  m[2][1] = ((z * (t * y)) + (s * x));
  m[2][2] = (c + (t * zSq));
  m[2][3] = 0;
}

void C_MTXReorder(const Mtx src, ROMtx dst)
{
  u32 i;
  u32 j;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 4; j++) {
      dst[j][i] = src[i][j];
    }
  }
}

void C_MTXLookAt(Mtx m, const Point3d* camPos, const Vec* camUp, const Point3d* target) {
  Vec vLook;
  Vec vRight;
  Vec vUp;

  assert(m && "MTXLookAt():  NULL MtxPtr 'm' ");
  assert(camPos && "MTXLookAt():  NULL VecPtr 'camPos' ");
  assert(camUp && "MTXLookAt():  NULL VecPtr 'camUp' ");
  assert(target && "MTXLookAt():  NULL Point3dPtr 'target' ");

  vLook.x = camPos->x - target->x;
  vLook.y = camPos->y - target->y;
  vLook.z = camPos->z - target->z;
  VECNormalize(&vLook, &vLook);
  VECCrossProduct(camUp, &vLook, &vRight);
  VECNormalize(&vRight, &vRight);
  VECCrossProduct(&vLook, &vRight, &vUp);
  m[0][0] = vRight.x;
  m[0][1] = vRight.y;
  m[0][2] = vRight.z;
  m[0][3] = -((camPos->z * vRight.z) + ((camPos->x * vRight.x) + (camPos->y * vRight.y)));
  m[1][0] = vUp.x;
  m[1][1] = vUp.y;
  m[1][2] = vUp.z;
  m[1][3] = -((camPos->z * vUp.z) + ((camPos->x * vUp.x) + (camPos->y * vUp.y)));
  m[2][0] = vLook.x;
  m[2][1] = vLook.y;
  m[2][2] = vLook.z;
  m[2][3] = -((camPos->z * vLook.z) + ((camPos->x * vLook.x) + (camPos->y * vLook.y)));
}

void C_MTXLightFrustum(Mtx m, f32 t, f32 b, f32 l, f32 r, f32 n, f32 scaleS, f32 scaleT, f32 transS, f32 transT) {
  f32 tmp;

  assert(m && "MTXLightFrustum():  NULL MtxPtr 'm' ");
  assert((t != b) && "MTXLightFrustum():  't' and 'b' clipping planes are equal ");
  assert((l != r) && "MTXLightFrustum():  'l' and 'r' clipping planes are equal ");

  tmp = 1 / (r - l);
  m[0][0] = (scaleS * (2 * n * tmp));
  m[0][1] = 0;
  m[0][2] = (scaleS * (tmp * (r + l))) - transS;
  m[0][3] = 0;
  tmp = 1 / (t - b);
  m[1][0] = 0;
  m[1][1] = (scaleT * (2 * n * tmp));
  m[1][2] = (scaleT * (tmp * (t + b))) - transT;
  m[1][3] = 0;
  m[2][0] = 0;
  m[2][1] = 0;
  m[2][2] = -1;
  m[2][3] = 0;
}

void C_MTXLightPerspective(Mtx m, f32 fovY, f32 aspect, f32 scaleS, f32 scaleT, f32 transS, f32 transT) {
  f32 angle;
  f32 cot;

  assert(m && "MTXLightPerspective():  NULL MtxPtr 'm' ");
  assert((fovY > 0.0) && (fovY < 180.0) && "MTXLightPerspective():  'fovY' out of range ");
  assert(0 != aspect && "MTXLightPerspective():  'aspect' is 0 ");

  angle = (0.5f * fovY);
  angle = MTXDegToRad(angle);
  cot = 1 / tanf(angle);
  m[0][0] = (scaleS * (cot / aspect));
  m[0][1] = 0;
  m[0][2] = -transS;
  m[0][3] = 0;
  m[1][0] = 0;
  m[1][1] = (cot * scaleT);
  m[1][2] = -transT;
  m[1][3] = 0;
  m[2][0] = 0;
  m[2][1] = 0;
  m[2][2] = -1;
  m[2][3] = 0;
}

void C_MTXLightOrtho(Mtx m, f32 t, f32 b, f32 l, f32 r, f32 scaleS, f32 scaleT, f32 transS, f32 transT) {
  f32 tmp;

  assert(m && "MTXLightOrtho():  NULL MtxPtr 'm' ");
  assert((t != b) && "MTXLightOrtho():  't' and 'b' clipping planes are equal ");
  assert((l != r) && "MTXLightOrtho():  'l' and 'r' clipping planes are equal ");
  tmp = 1 / (r - l);
  m[0][0] = (2 * tmp * scaleS);
  m[0][1] = 0;
  m[0][2] = 0;
  m[0][3] = (transS + (scaleS * (tmp * -(r + l))));
  tmp = 1/ (t - b);
  m[1][0] = 0;
  m[1][1] = (2 * tmp * scaleT);
  m[1][2] = 0;
  m[1][3] = (transT + (scaleT * (tmp * -(t + b))));
  m[2][0] = 0;
  m[2][1] = 0;
  m[2][2] = 0;
  m[2][3] = 1;
}

/* PSMTXConcat as the paired-single code on the disc computes it
 * (0x80342204): each element of ab's row i a ps_muls0 then two ps_madds
 * (fused) down b's column, a*b ordered as the instructions have them, and
 * the translation column adding a's own last (a ps_madds1 by 1.0: a plain
 * add). All of a and b is read before ab is written, so ab may alias
 * either. */
void PC_PSMTXConcat(const Mtx a, const Mtx b, Mtx ab) {
  Mtx r;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      const f32 t = ppc_fmadds(b[2][j], a[i][2], ppc_fmadds(b[1][j], a[i][1], b[0][j] * a[i][0]));
      /* the last ps_madds1 runs over columns 2 and 3 by (0.0, 1.0): 0 *
       * a[i][3] onto column 2 (which can only change a zero's sign) */
      r[i][j] = j == 3 ? a[i][3] + t : j == 2 ? ppc_fmadds(0.0f, a[i][3], t) : t;
    }
  }
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 4; j++)
      ab[i][j] = r[i][j];
}

/* PSMTXQuat as the paired-single code on the disc computes it (0x80342690):
 * 2/|q|^2 from fres and one Newton step (not a division), the sums paired
 * as the instructions pair them. Each line is one instruction, both slots
 * of a register written out; ps_ fused ops round once (Dolphin's). */
void PC_PSMTXQuat(Mtx m, const Quaternion* q) {
  const f32 x = q->x, y = q->y, z = q->z, w = q->w;
  /* f6 = ps_mul q.xy*q.xy; f9 = merge10 (y, x) */
  const f32 f6a = x * x, f6b = y * y;
  /* f8 = ps_madd q.zw*q.zw + f6 */
  f32 f8a = ppc_fmadds(z, z, f6a), f8b = ppc_fmadds(w, w, f6b);
  /* f7 = ps_mul q.zw*q.zw */
  f32 f7a = z * z;
  /* f3 = ps_sum0 f8: |q|^2 */
  const f32 n = f8a + f8b;
  /* f10 = ps_muls1 (y, x) * w */
  const f32 f10a = y * w, f10b = x * w;
  /* f11 = fres n; f8 = ps_sum1 (f8a, f7a + f6b) */
  const f32 est = (f32)ppc_fres(n);
  f8b = f7a + f6b;
  /* f3 = ps_nmsub -(n*est - 2); f3 = est * f3; f3 = fmuls f3 * 2 */
  const f32 s = (est * ppc_fnmsubs(n, est, 2.0f)) * 2.0f;
  /* f7 = ps_muls1 (z, w) * w */
  const f32 f7a2 = z * w, f7b2 = w * w;
  /* f6 = ps_sum0: xx + yy */
  const f32 f6s = f6a + f6b;
  /* f11 = ps_madd (x, y)*(y, x) + f7; f7 = ps_msub likewise */
  const f32 f11a = ppc_fmadds(x, y, f7a2), f11b = ppc_fmadds(y, x, f7b2);
  const f32 f7ma = ppc_fmsubs(x, y, f7a2), f7mb = ppc_fmsubs(y, x, f7b2);
  /* f6 = ps_nmsub -(f6*s - 1); f8 likewise */
  const f32 m22 = ppc_fnmsubs(f6s, s, 1.0f);
  const f32 m11 = ppc_fnmsubs(f8a, s, 1.0f), m00 = ppc_fnmsubs(f8b, s, 1.0f);
  (void)f11b;
  /* f11 = ps_mul f11*s; f7 = ps_mul f7*s */
  const f32 m10 = f11a * s, m01 = f7ma * s;
  (void)f7mb;
  /* f9 = ps_madds0 (x, y)*z + f10 */
  const f32 f9a = ppc_fmadds(x, z, f10a), f9b = ppc_fmadds(y, z, f10b);
  /* f10 = ps_nmsub -(f10*2 - f9) */
  const f32 f10na = ppc_fnmsubs(f10a, 2.0f, f9a), f10nb = ppc_fnmsubs(f10b, 2.0f, f9b);
  /* f9 = ps_mul f9*s; f10 = ps_mul f10*s */
  const f32 m02 = f9a * s, m21 = f9b * s;
  const f32 m20 = f10na * s, m12 = f10nb * s;
  m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = 0.0f;
  m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = 0.0f;
  m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = 0.0f;
}
