//func Sclean(X []float32)
TEXT ·Sclean(SB), 7, $0
	MOVQ	X_data+0(FP), DI
	MOVQ	X_len+8(FP), BP

	// Setup four zeros in X0
	XORPS	X0, X0
	
	SUBQ	$4, BP
	JL		rest	// There are less than 4 pairs to process

	simd_loop:
		MOVUPS	X0, (DI)
		// Update data pointers
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
		MOVSS	X0, (DI)
		// Update data pointers
		ADDQ	$4, DI
		DECQ	BP
		JNE	loop
	RET

end:
	RET
