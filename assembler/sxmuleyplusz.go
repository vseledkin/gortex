//+build !amd64 noasm

package assembler

func Sxmuleyplusz(X, Y, Z []float32) {
	for i := range X {
		Z[i] += X[i] * Y[i]
	}
}
