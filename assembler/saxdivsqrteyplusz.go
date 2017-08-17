package assembler

func Saxdivsqrteyplusz(a float32, X []float32, b float32, Y []float32, Z []float32)

func saxdivsqrteyplusz(a float32, X []float32, b float32, Y []float32, Z []float32) {
	for i := range X {
		Z[i] += a * X[i] / Sqrt(b+Y[i])
	}
}
