

// func supportsAVX2() bool
TEXT ·supportsAVX2(SB), 4, $0-1
    MOVQ runtime·support_avx2(SB), AX
    MOVB AX, ret+0(FP)
    RET

// func supportsAVX() bool
TEXT ·supportsAVX(SB), 4, $0-1
    MOVQ runtime·support_avx(SB), AX
    MOVB AX, ret+0(FP)
    RET

// func supportsSSE4() bool
TEXT ·supportsSSE4(SB), 4, $0-1
    MOVL $1, AX
	CPUID
	SHRL $19, CX  // Bit 19 indicates SSE4 support
	ANDL $1, CX  // CX != 0 if support SSE4
	MOVB CX, ret+0(FP)
	RET
