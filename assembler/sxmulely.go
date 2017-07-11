package assembler

func sxmulely_asm(X, Y []float32)

func sxmulely(X, Y []float32) {
	for i := range X {
		Y[i] *= X[i]
	}
}

