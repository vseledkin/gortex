package assembler

import "math"

func L2(X []float32) float32

func l2(X []float32) (nrm2 float32) {
	for _, x := range X {
		nrm2 += x * x
	}
	nrm2 = float32(math.Sqrt(float64(nrm2)))
	return
}
