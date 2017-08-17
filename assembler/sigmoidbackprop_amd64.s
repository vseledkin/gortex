//func Sigmoidbackprop(a float32, X, Y, Z []float32)
TEXT ·Sigmoidbackprop(SB), 7, $0
    MOVSS	alpha+0(FP), X0
	MOVQ	X_data+8(FP), SI
	MOVQ	X_len+16(FP), BP
	MOVQ	Y_data+32(FP), CX
	MOVQ	Z_data+56(FP), DI


	SUBQ	$4, BP
	JL		rest	// There are less than 4 pairs to process

	// Setup four 1 in X0
    SHUFPS	$0, X0, X0
	simd_loop:
		// Load four pairs and scale
		MOVUPS	(SI), X1
                MOVUPS	(SI), X2
                SUBPS	X2, X0
                MULPS	X0, X1

                MOVUPS	alpha+0(FP), X0
                SHUFPS	$0, X0, X0

                MOVUPS	(CX), X2
                MULPS	X2, X1

                MOVUPS	(DI), X2
                ADDPS   X2, X1
                MOVUPS	X1, (DI)

		// Update data pointers
		ADDQ	$16, SI
		ADDQ	$16, CX
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
		MOVSS	(SI), X1
        MOVSS	(SI), X2
        SUBSS	X2, X0
        MULSS	X0, X1

        MOVSS	alpha+0(FP), X0

        MOVSS	(CX), X2
        MULSS	X2, X1

        MOVSS	(DI), X2
        ADDSS   X2, X1
        MOVSS	X1, (DI)

		// Update data pointers
		ADDQ	$4, SI
		ADDQ	$4, CX
		ADDQ	$4, DI

		DECQ	BP
		JNE	loop
	RET

end:
	RET
