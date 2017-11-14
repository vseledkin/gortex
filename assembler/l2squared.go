//+build !amd64 noasm

package assembler

// square of l2 norm
func L2Squared(x []float32) (nrm float32) {
	for i := range x {
		nrm += x[i] * x[i]
	}
	return
}
