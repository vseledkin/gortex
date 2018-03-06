//+build amd64,!noasm

#include "textflag.h"

//func SxmyAvx(X, Y []float32)
TEXT ·SxmyAvx(SB), 7, $0
	MOVQ	X_data+0(FP), SI
	MOVQ	X_len+8(FP), BP
	MOVQ	Y_data+24(FP), DI

	SUBQ	$8, BP
	JL		sse4	// There are less than 8 pairs to process
	avx_loop:
		// Load four pairs and scale
		VMOVUPS	(SI), Y1
		// Save product
		VMULPS	(DI), Y1, Y1
		VMOVUPS	Y1, (DI)

		// Update data pointers
		ADDQ	$32, SI
		ADDQ	$32, DI

		SUBQ	$8, BP
		JGE		avx_loop	// There are 8 or more pairs to process
sse4:
    // we can have remain > 4
	// Undo half of last SUBQ
    ADDQ	$4,	BP
    JL		rest	// There are less than 4 pairs to process
	simd_loop:
		// Load four pairs and scale
		// do not use X1 X2 to avoid AVX upper registers load
		VMOVUPS	(SI), X1
		// Save sum
		VMULPS	(DI), X1, X1
		MOVUPS	X1, (DI)

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
		VMULSS	(DI), X1, X1
		VMOVSS	X1, (DI)

		// Update data pointers
		ADDQ	$4, SI
		ADDQ	$4, DI

		DECQ	BP
		JNE	loop
	RET

end:
	RET
