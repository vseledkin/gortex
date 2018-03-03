//+build !amd64 noasm

package assembler

func Sdot(X, Y []float32) (dot float32) {
	for i, x := range X {
		dot += x * Y[i]
	}
	return
}
