
#include "textflag.h"

// func Sqrt(x float32) float32
TEXT ·Sqrt(SB),NOSPLIT,$0
  MOVF x+0(FP), F0
  SQRTF F0, F0
  MOVF F0, ret+4(FP)
  RET

