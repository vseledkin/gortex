package assembler

func Sigmoidbackprop(a float32, X []float32, Y []float32, Z []float32)

func sigmoidbackprop(a float32, X []float32, Y []float32, Z []float32) {
	for i := range X {
		Z[i] += X[i] * (1 - X[i]) * Y[i]
	}
}
