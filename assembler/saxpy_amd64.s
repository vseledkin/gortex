//+build amd64,!noasm

#include "textflag.h"


//func SaxpySSE4(alpha float32, X []float32, Y []float32)
TEXT ·SaxpySSE4(SB), 7, $0
	MOVSS	alpha+0(FP), X0
	MOVQ	X_data+8(FP), SI
	MOVQ	X_len+16(FP), BP
	MOVQ	Y_data+32(FP), DI

	SUBQ	$4, BP
	JL		rest	// There are less than 4 pairs to process
	// Setup four alphas in X0
	SHUFPS	$0, X0, X0

	simd_loop:
		// Load four pairs and scale
		MOVUPS	(SI), X2
		MOVUPS	(DI), X3
		MULPS	X0, X2
		// Save sum
		ADDPS	X2, X3
		MOVUPS	X3, (DI)

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
		// Load from X and scale
		MOVSS	(SI), X2
		MULSS	X0, X2
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
