package assembler

// L2squared square of the l2
func L2squared(X []float32) float32

func l2squared(X []float32) (nrm float32) {
	for _, x := range X {
		nrm += x * x
	}
	return
}
