//+build amd64,!noasm

#include "textflag.h"

//func SdotSSE4(X, Y []float32) float32
TEXT ·SdotSSE4(SB), 7, $0
	MOVQ	X_data+0(FP), SI
	MOVQ	X_len+8(FP), BP
	MOVQ	Y_data+24(FP), DI

	// Clear accumulator
	XORPS	X0, X0

	// Check that there are 4 or more pairs for SIMD calculations
	SUBQ	$4, BP
	JL		rest	// There are less than 4 pairs to process

	simd_loop:
		// Multiply-add four pairs
		MOVUPS	(SI), X1
		MOVUPS	(DI), X2
		DPPS $0xF1, X2, X1
		ADDSS	X1, X0

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
		// Multiply one pair
		MOVSS	(SI), X3
		MULSS	(DI), X3

		ADDSS	X3, X0


		// Update data pointers
		ADDQ	$4, SI
		ADDQ	$4, DI


		DECQ	BP
		JNE	loop
end:
	MOVSS	X0, r+48(FP)
	RET
