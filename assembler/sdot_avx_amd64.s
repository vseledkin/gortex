//+build amd64,!noasm

#include "textflag.h"

//func SdotAvx(X, Y []float32) float32
TEXT ·SdotAvx(SB), 7, $0
	MOVQ	X_data+0(FP), SI
	MOVQ	X_len+8(FP), BP
	MOVQ	Y_data+24(FP), DI

	VXORPS Y0, Y0, Y0
	// Check that there are 4 or more pairs for SIMD calculations
	SUBQ	$8, BP
	JL		sse4	// There are less than 4 pairs to process

	avx_loop:
		// Multiply-add 8 pairs
		VMOVUPS	(SI), Y1
		VMOVUPS	(DI), Y2
		VDPPS $0xF1, Y2, Y1, Y1
		VADDPS	Y1, Y0, Y0

		// Update data pointers
		ADDQ	$32, SI
		ADDQ	$32, DI


		SUBQ	$8, BP
		JGE		avx_loop	// There are 8 or more pairs to process

    //SUBQ	$64, SI
    // sum up low and hight data
    VEXTRACTF128	$1, Y0,X1
    VEXTRACTF128	$0, Y0,X2
    VADDSS X2, X1, X1
    VMOVSS X1, X0, X0

sse4:
    	// Undo half of last SUBQ
    	ADDQ	$4,	BP
    	// Check that are there any value to process
    	JL	rest
        sse4_loop:
            // Multiply-add four pairs
            VMOVUPS	(SI), X1
            VDPPS $0xF1, (DI), X1, X1
            VADDSS	X1, X0, X0

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
		// Multiply one pair
		VMOVSS	(SI), X1
		VMULSS	(DI), X1, X1
		VADDSS	X1, X0, X0

		// Update data pointers
		ADDQ	$4, SI
		ADDQ	$4, DI

		DECQ	BP
		JNE	loop
end:
	VMOVSS	X0, r+48(FP)
	RET
