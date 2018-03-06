//+build !amd64 noasm

package assembler

func Sclean(X []float32) { // golang version is actually faster it uses memclr calls
	for i := range X {
		X[i] = .0
	}
}
