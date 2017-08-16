package assembler

func sxmuleyplusz_asm(X, Y, Z []float32)

func sxmuleyplusz(X, Y, Z []float32) {
	for i := range X {
		Z[i] += X[i] * Y[i]
	}
}
