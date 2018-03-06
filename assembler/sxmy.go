//+build !amd64 noasm

package assembler

func Sxmuley(X, Y []float32) {
	for i := range X {
		Y[i] *= X[i]
	}
}
