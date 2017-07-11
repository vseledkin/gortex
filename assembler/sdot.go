package assembler

//Sdot Scalar product: X^T Y
func Sdot(X, Y []float32) float32

func sdot(X, Y []float32) (dot float32) {
	for i, x := range X {
		dot += x * Y[i]
	}
	return
}
