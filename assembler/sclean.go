package assembler

func Sclean(X []float32)

func sclean(X []float32) {
	for i := range X {
		X[i] = .0
	}
}
