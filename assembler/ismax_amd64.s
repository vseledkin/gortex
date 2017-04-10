// func ismax_asm(X []float32) int
TEXT ·ismax_asm(SB), 7, $0
	MOVQ	X_data+0(FP), SI
    MOVQ	X_len+8(FP), BP

	// Max value
	XORPS	X0, X0
	// Index
	XORQ	DI, DI
	// Max index
	XORQ	BX, BX


loop:
	CMPQ	BP, DI
	JE	end

	// Load value
	MOVSS	(SI), X1

	// Is loaded value less or equal to max value?
	UCOMISS	X0,	X1
	JBE	less_or_equal

	// Save max index and value
	MOVQ	DI, BX
	MOVSS	X1, X0

less_or_equal:
	// Update data pointers
	ADDQ	$4, SI
	INCQ	DI
	JMP	loop

end:
	// Return the max index
	MOVQ	BX, r+24(FP)
	RET
