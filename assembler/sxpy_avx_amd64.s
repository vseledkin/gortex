//+build amd64,!noasm

#include "textflag.h"

//func SxpyAvx(X, Y []float32)
TEXT ·SxpyAvx(SB), 7, $0
	MOVQ	X_data+0(FP), SI
	MOVQ	X_len+8(FP), BP
	MOVQ	Y_data+24(FP), DI

	SUBQ	$8, BP
	JL		simd	// There are less than 8 pairs to process
	avx_loop:
		// Add 8 pairs
		VMOVUPS	(SI), Y1
		// Save sum
		VADDPS	(DI), Y1, Y1
		VMOVUPS	Y1, (DI)

		// Update data pointers
		ADDQ	$32, SI
		ADDQ	$32, DI

		SUBQ	$8, BP
		JGE		avx_loop	// There are 8 or more pairs to process

	simd:
	// Undo half of last SUBQ
    ADDQ	$4,	BP
    JL		rest	// There are less than 4 pairs to process
	simd_loop:
		// Load four pairs and scale
		VMOVUPS	(SI), X1
		// Save sum
		VADDPS	(DI), X1, X1
		VMOVUPS	X1, (DI)

		// Update data pointers
		ADDQ	$16, SI
		ADDQ	$16, DI

		SUBQ	$4, BP
		JGE		simd_loop	// There are 4 or more pairs to process
rest:
	// Undo last SUBQ
	ADDQ	$4,	BP
	// Check that are there any value to process
	JE	end
	loop:
		// Load from X and scale
		VMOVSS	(SI), X1
		// Save sum in Y
		VADDSS	(DI), X1, X1
		VMOVSS	X1, (DI)

		// Update data pointers
		ADDQ	$4, SI
		ADDQ	$4, DI

		DECQ	BP
		JNE	loop
	RET

end:
	RET
