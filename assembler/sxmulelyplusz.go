package assembler

func sxmulelyplusz_asm(X, Y, Z []float32)

func sxmulelyplusz(X, Y, Z []float32) {
	for i := range X {
		Z[i] += X[i] * Y[i]
	}
}
