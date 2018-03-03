//+build amd64,!noasm

#include "textflag.h"

//func Sset(A float32, X []float32)
TEXT ·Sset(SB), 7, $0
  MOVSS	alpha+0(FP), X0
  MOVQ	X_data+8(FP), SI
  MOVQ	X_len+16(FP), BP

  SUBQ	$4, BP
  JL		rest	// There are less than 4 pairs to process

  // Setup four alphas in X0
  SHUFPS	$0, X0, X0
  simd_loop:
    MOVUPS	X0, (SI)
    // Update data pointers
    ADDQ	$16, SI
    SUBQ	$4, BP
    JGE		simd_loop	// There are 4 or more pairs to process
  JMP	rest

rest:
  // Undo last SUBQ
  ADDQ	$4,	BP
  // Check that are there any value to process
  JE	end
  loop:
    MOVSS	X0, (SI)
    // Update data pointers
    ADDQ	$4, SI
    DECQ	BP
    JNE	loop
  RET

end:
  RET

//func SsetAVX(A float32, X []float32)
TEXT ·SsetAVX(SB), 7, $0
  VMOVSS alpha+0(FP), X0
  MOVQ	X_data+8(FP), SI
  MOVQ	X_len+16(FP), BP

  SUBQ	$4, BP
  JL		rest	// There are less than 4 pairs to process

  // Setup four alphas in X0
  SHUFPS	$0, X0, X0
  simd_loop:
    MOVUPS	X0, (SI)
    // Update data pointers
    ADDQ	$16, SI
    SUBQ	$4, BP
    JGE		simd_loop	// There are 4 or more pairs to process
  JMP	rest

rest:
  // Undo last SUBQ
  ADDQ	$4,	BP
  // Check that are there any value to process
  JE	end
  loop:
    MOVSS	X0, (SI)
    // Update data pointers
    ADDQ	$4, SI
    DECQ	BP
    JNE	loop
  RET

end:
  RET
