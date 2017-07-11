// func l1asm(X []float32) float32
TEXT ·l1asm(SB), 7, $0
  MOVQ	X_data+0(FP), SI
  MOVQ	X_len+8(FP), BP

  // Clear accumulator
  XORPS	X0, X0
  // Check that there are 4 or more pairs for SIMD calculations
  SUBQ	$4, BP
  JL		rest	// There are less than 4 pairs to process

  simd_loop:
    //PREFETCHNTA (128*4)(SI)
    // Clear sign on all four values
    MOVUPS	(SI), X1
    ADDPS	X1, X0

    // Update data pointer
    ADDQ	$16, SI
    SUBQ	$4, BP
    JGE		simd_loop	// There are 4 or more pairs to process

  // Horizontal sum
  MOVHLPS X0, X1
  ADDPS	X0, X1
  MOVSS	X1, X0
  SHUFPS	$0xe1, X1, X1
  ADDSS	X1, X0

rest:
  // Undo last SUBQ
  ADDQ	$4,	BP

  // Check that are there any value to process
  JE	end

  loop:
    // Multiply one value
    MOVSS	(SI), X1
    // Update data pointers
    ADDQ	$4, SI
    // Accumulate the results of multiplication
    ADDSS	X1, X0
    DECQ	BP
    JNE		loop

end:
  // Return
  MOVSS	X0, r+24(FP)
  RET
