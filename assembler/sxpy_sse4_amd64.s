//+build amd64,!noasm

#include "textflag.h"

//func SxpySSE4(X []float32, Y []float32)
TEXT ·SxpySSE4(SB), 7, $0
	MOVQ	X_data+0(FP), SI
	MOVQ	X_len+8(FP), BP
	MOVQ	Y_data+24(FP), DI

	SUBQ	$4, BP
	JL		rest	// There are less than 4 pairs to process
	simd_loop:
		// Load four pairs and scale
		MOVUPS	(SI), X1
		// sum
		ADDPS	(DI), X1
		MOVUPS	X1, (DI)

		// Update data pointers
		ADDQ	$16, SI
		ADDQ	$16, DI

		SUBQ	$4, BP
		JGE		simd_loop	// There are 4 or more pairs to process
	JMP	rest

rest:
	// Undo last SUBQ
	ADDQ	$4,	BP
	// Check that are there any value to process
	JE	end
	loop:
		MOVSS	(SI), X2
		// Save sum in Y
		ADDSS	(DI), X2
		MOVSS	X2, (DI)

		// Update data pointers
		ADDQ	$4, SI
		ADDQ	$4, DI

		DECQ	BP
		JNE	loop
	RET

end:
	RET
