package gortex

import "math"

func Zeros(n int) []float64 {
	if n <= 0 {
		return []float64{}
	}
	return make([]float64, n)
}

func Softmax(m *Matrix) *Matrix {
	out := Mat(m.n, m.d) // probability volume
	maxval := -math.MaxFloat64
	l := len(m.W)
	for i := 0; i < l; i++ {
		if m.W[i] > maxval {
			maxval = m.W[i]
		}
	}

	s := 0.0
	for i := 0; i < l; i++ {
		out.W[i] = math.Exp(m.W[i] - maxval)
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
