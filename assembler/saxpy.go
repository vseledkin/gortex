package assembler

func Saxpy(a float32, X []float32, Y []float32)

func saxpy(a float32, X []float32, Y []float32) {
	for i := range X {
		Y[i] += a * X[i]
	}
}
