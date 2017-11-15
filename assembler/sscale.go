//+build !amd64 noasm

package assembler

func Sscale(a float32, X []float32) {
	for i := range X {
		X[i] *= a
	}
}

