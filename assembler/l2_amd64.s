//func L2(X []float32)
TEXT ·L2(SB), 7, $0
	MOVQ	X_data+0(FP), SI
	MOVQ	X_len+8(FP), BP

	// Clear accumulator
	XORPS	X0, X0
	SUBQ	$4, BP
	JL		rest	// There are less than 4 pairs to process

	simd_loop:
		// Load four pairs and scale
		MOVUPS	(SI), X1
		MULPS	X1, X1
		// Save sum
		ADDPS	X1, X0
		// Update data pointers
		ADDQ	$16, SI

		SUBQ	$4, BP
		JGE		simd_loop	// There are 4 or more pairs to process
	JMP	rest

rest:
	// Horizontal sum
	MOVHLPS X0, X1
	ADDPS	X0, X1
	MOVSS	X1, X0
	SHUFPS	$0xe1, X1, X1
	ADDSS	X1, X0
	// Undo last SUBQ
	ADDQ	$4,	BP
	// Check that are there any value to process
	JE	end
	loop:
		// Load from X and scale
		MOVSS	(SI), X1
		MULSS	X1, X1
		// Save sum in Y
		ADDSS	X1, X0

		// Update data pointers
		ADDQ	$4, SI

		DECQ	BP
		JNE	loop
end:
	SQRTSS	X0, X0
	MOVSS	X0, r+24(FP)
	RET
