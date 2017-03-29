package gortex

import "math"
import "github.com/vseledkin/gortex/assembler"

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

	for i := 0; i < l; i++ {
		out.W[i] = float32(math.Exp(float64(m.W[i] - maxval)))
	}
	sum := assembler.Ssum(out.W)
	assembler.Sscale(1/sum, out.W)

	// no backward pass here needed
	// since we will use the computed probabilities outside
	// to set gradients directly on m
	return out
}

func Moments(m *Matrix) (mean, variance float32) {
	mean = assembler.Ssum(m.W) / float32(len(m.W))

	var total float32
	var tmp float32
	for i := range m.W {
		tmp = m.W[i] - mean
		total += tmp * tmp
	}
	variance = total / float32(len(m.W))
	return
}

func MaxIV(m *Matrix) (int, float32) {
	var max float32 = -math.MaxFloat32
	maxIndex := -1
	for i, v := range m.W {
		if v > max {
			max = v
			maxIndex = i
		}
	}
	return maxIndex, max
}
