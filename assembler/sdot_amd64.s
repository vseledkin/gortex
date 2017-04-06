//func Sdsot(X, Y []float32) float32
TEXT ·Sdot(SB), 7, $0
	MOVQ	X_data+0(FP), SI
	MOVQ	X_len+8(FP), BP
	MOVQ	Y_data+24(FP), DI

	// Clear accumulator
	XORPS	X0, X0

	// Check that there are 4 or more pairs for SIMD calculations
	SUBQ	$4, BP
	JL		rest	// There are less than 4 pairs to process

	simd_loop:
		// Multiply four pairs
		MOVUPS	(SI), X2
		MOVUPS	(DI), X3
		MULPS	X2, X3

		// Update data pointers
		ADDQ	$16, SI
		ADDQ	$16, DI

		// Accumulate the results of multiplications
		ADDPS	X3, X0

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
		// Multiply one pair
		MOVSS	(SI), X1
		MULSS	(DI), X1


		// Update data pointers
		ADDQ	$4, SI
		ADDQ	$4, DI

		// Save sum in Y
		ADDSS	X1, X0

		DECQ	BP
		JNE	loop
end:
	MOVSS	X0, r+48(FP)
	RET
