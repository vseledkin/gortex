package assembler

func Saxplusbysetz(a float32, X []float32, b float32, Y []float32, Z []float32)

func saxplusbysetz(a float32, X []float32, b float32, Y []float32, Z []float32) {
	for i := range X {
		Z[i] = a*X[i] + b*Y[i]
	}
}
