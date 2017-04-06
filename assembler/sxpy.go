package assembler

func Sxpy(X, Y []float32)

func sxpy(X, Y []float32) {
	for i := range X {
		Y[i] += X[i]
	}
}
