package assembler

func Saxplusbyvsetz(a float32, X []float32, b float32, Y []float32, V []float32, Z []float32)

func saxplusbyvsetz(a float32, X []float32, b float32, Y []float32, V []float32, Z []float32) {
	for i := range X {
		Z[i] = a*X[i] + b*Y[i]*V[i]
	}
}
