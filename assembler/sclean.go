package assembler

func Sclean(X []float32)

func sclean(X []float32) { // golang version is actually faster it uses memclr calls
	for i := range X {
		X[i] = .0
	}
}
