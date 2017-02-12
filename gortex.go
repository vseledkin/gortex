package gortex

import "math"

func Zeros(n int) []float32 {
	return make([]float32, n)
}

//Softmax probability distribution interpretation of any vector/matrix
func Softmax(m *Matrix) *Matrix {
	out := Mat(m.Rows, m.Columns) // probability volume
	var maxval float32 = -math.MaxFloat32
	l := len(m.W)
	for i := 0; i < l; i++ {
		if m.W[i] > maxval {
			maxval = m.W[i]
		}
	}

	var s float32
	for i := 0; i < l; i++ {
		out.W[i] = float32(math.Exp(float64(m.W[i] - maxval)))
		s += out.W[i]
	}

	for i := 0; i < l; i++ {
		out.W[i] /= s
	}

	// no backward pass here needed
	// since we will use the computed probabilities outside
	// to set gradients directly on m
	return out
}
