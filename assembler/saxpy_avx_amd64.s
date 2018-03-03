//+build amd64,!noasm

#include "textflag.h"


//func SaxpyAvx(alpha float32, X []float32, Y []float32)
TEXT ·SaxpyAvx(SB), 7, $0
    // setup 8 alphas
    VBROADCASTSS alpha+0(FP), Y0
	MOVQ	X_data+8(FP), SI
	MOVQ	X_len+16(FP), BP
	MOVQ	Y_data+32(FP), DI

	SUBQ	$8, BP
	JL		sse4	// There are less than 4 pairs to process
	// Setup four alphas in X0
	//VMOVUPS	Y0, (SI)
	//JMP end

	avx_loop:
		// Load four pairs and scale
		VMOVUPS	(SI), Y1
		//VMOVUPS	(DI), Y2
		VMULPS	Y0, Y1, Y1
		// Save sum
		VADDPS	(DI), Y1, Y1
		VMOVUPS	Y1, (DI)

		// Update data pointers
		ADDQ	$32, SI
		ADDQ	$32, DI

		SUBQ	$8, BP
		JGE		avx_loop	// There are 4 or more pairs to process

sse4:
	ADDQ	$4,	BP
	JL		rest	// There are less than 4 pairs to process
sse4_loop:
		// Load four pairs and scale
		VMOVUPS	(SI), X1
		//VMOVUPS	(DI), Y2
		VMULPS	X0, X1, X1
		// Save sum
		VADDPS	(DI), X1, X1
		VMOVUPS	X1, (DI)

		// Update data pointers
		ADDQ	$16, SI
		ADDQ	$16, DI

		SUBQ	$4, BP
		JGE		sse4_loop	// There are 4 or more pairs to process

rest:
	// Undo last SUBQ
	ADDQ	$4,	BP
	// Check that are there any value to process
	JE	end
	loop:
		// Load from X and scale
		VMOVSS	(SI), X1
		VMULSS	X0, X1, X1
		// Save sum in Y
		VADDSS	(DI), X1, X1
		VMOVSS	X1, (DI)

		// Update data pointers
		ADDQ	$4, SI
		ADDQ	$4, DI

		DECQ	BP
		JNE	loop

end:
    //VZEROUPPER
	RET
