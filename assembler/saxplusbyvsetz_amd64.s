//func Saxplusbyvsetz(a, float32, X []float32, b float32, Y []float32, V []float32, Z []float32)
TEXT ·Saxplusbyvsetz(SB), 7, $0
    MOVSS	a+0(FP), X0
	MOVQ	X_data+8(FP), SI
	MOVQ	X_len+16(FP), BP
    MOVSS	b+32(FP), X1
	MOVQ	Y_data+40(FP), CX
	MOVQ	V_data+64(FP), BX
	MOVQ	Z_data+88(FP), DI

	SUBQ	$4, BP
	JL		rest	// There are less than 4 pairs to process
	// Setup four a's in X0
	SHUFPS	$0, X0, X0
	// Setup four b's in X1
	SHUFPS	$0, X1, X1


	simd_loop:
		// Load four pairs and scale
		MOVUPS	(SI), X2 // X[SI]
		MULPS	X0, X2 // a*X[SI]

		MOVUPS	(CX), X3 // Y[CX]
		MULPS	X1, X3 // b*Y[CX]
		MOVUPS	(BX), X4 // V[BX]
		MULPS	X4, X3 // b*Y[CX]
		// Save sum
		ADDPS	X2, X3
		MOVUPS	X3, (DI)

		// Update data pointers
		ADDQ	$16, SI
		ADDQ	$16, CX
		ADDQ	$16, BX
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

		MOVSS	(CX), X3
        MULSS	X1, X3
		MOVSS	(BX), X4
        MULSS	X4, X3
		// Save sum in Z
		ADDSS	X2, X3
		MOVSS	X3, (DI)

		// Update data pointers
		ADDQ	$4, SI
		ADDQ	$4, CX
		ADDQ	$4, BX
		ADDQ	$4, DI

		DECQ	BP
		JNE	loop
	RET

end:
	RET
