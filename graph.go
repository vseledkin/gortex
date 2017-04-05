package gortex

import (
	"fmt"
	"math"
)

const epsilon = 1e-9

type Graph struct {
	NeedsBackprop bool

	// this will store a list of functions that perform backprop,
	// in their forward pass order. So in backprop we will go
	// backwards and evoke each one
	backprop []func()
}

func (g *Graph) Backward() {
	for i := len(g.backprop) - 1; i >= 0; i-- {
		g.backprop[i]() // tick!
	}
}

func (g *Graph) InstanceNormalization(m *Matrix) *Matrix {
	mean, variance := Moments(m)
	stdDev := float32(math.Sqrt(float64((variance))))
	out := m.SameAs()
	for i := range m.W {
		out.W[i] = (m.W[i] - mean) / (stdDev + epsilon)
	}
	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			scale := (1.0 - 1.0/float32(len(m.W))) / (stdDev + epsilon)
			for i := range m.W {
				m.DW[i] += scale * out.DW[i]
			}
		})
	}
	return out
}

func (g *Graph) Tanh(m *Matrix) *Matrix {
	// tanh nonlinearity
	out := m.SameAs()

	for i := range m.W {
		out.W[i] = float32(math.Tanh(float64(m.W[i])))
	}

	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			for i := range m.W {
				// grad for z = tanh(x) is (1 - z^2)
				m.DW[i] += (1.0 - out.W[i]*out.W[i]) * out.DW[i]
			}
		})
	}
	return out
}

func (g *Graph) Sigmoid(m *Matrix) *Matrix {
	// sigmoid nonlinearity
	out := m.SameAs()

	for i := range m.W {
		out.W[i] = 1.0 / (1.0 + float32(math.Exp(float64(-m.W[i])))) // Sigmoid
	}

	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			for i := range m.W {
				// grad for z = sigmoid(x) is sigmoid(x)(1 - sigmoid(x))
				m.DW[i] += out.W[i] * (1.0 - out.W[i]) * out.DW[i]
			}
		})
	}
	return out
}

func (g *Graph) Add(m1, m2 *Matrix) *Matrix {
	l1 := len(m1.W)
	l2 := len(m2.W)
	if l1 != l2 {
		panic(fmt.Errorf("matadd number of elements must be equal numel(m1)=%d must be equal numel(m2)=%d", l1, l2))
	}

	out := m1.SameAs()
	for i := 0; i < l1; i++ {
		out.W[i] = m1.W[i] + m2.W[i]
	}

	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			for i := 0; i < l1; i++ {
				m1.DW[i] += out.DW[i]
				m2.DW[i] += out.DW[i]
			}
		})
	}
	return out
}

func (g *Graph) Mul(m1, m2 *Matrix) *Matrix {
	// multiply matrices m1 * m2
	if m1.Columns != m2.Rows {
		panic(fmt.Errorf("matmul dimensions misaligned m1.columns=%d must be equal m2.rows=%d", m1.Columns, m2.Rows))
	}

	out := Mat(m1.Rows, m2.Columns)
	for i := 0; i < m1.Rows; i++ { // loop over rows of m1
		for j := 0; j < m2.Columns; j++ { // loop over cols of m2
			var dot float32
			for k := 0; k < m1.Columns; k++ { // dot product loop
				dot += m1.W[m1.Columns*i+k] * m2.W[m2.Columns*k+j]
			}
			out.W[m2.Columns*i+j] = dot
		}
	}

	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			for i := 0; i < m1.Rows; i++ { // loop over rows of m1
				for j := 0; j < m2.Columns; j++ { // loop over cols of m2
					for k := 0; k < m1.Columns; k++ { // dot product loop
						b := out.DW[m2.Columns*i+j]
						m1.DW[m1.Columns*i+k] += m2.W[m2.Columns*k+j] * b
						m2.DW[m2.Columns*k+j] += m1.W[m1.Columns*i+k] * b
					}
				}
			}
		})
	}
	return out
}

// EMul elementwise matrix matrix multiplication
func (g *Graph) EMul(m1, m2 *Matrix) *Matrix {
	l1 := len(m1.W)
	l2 := len(m2.W)
	if l1 != l2 {
		panic(fmt.Errorf("emul number of elements must be equal numel(m1)=%d must be equal numel(m2)=%d", l1, l2))
	}

	out := m1.SameAs()
	for i := range m1.W {
		out.W[i] = m1.W[i] * m2.W[i]
	}
	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			for i := range m1.W {
				m1.DW[i] += m2.W[i] * out.DW[i]
				m2.DW[i] += m1.W[i] * out.DW[i]
			}
		})
	}
	return out
}

func (g *Graph) MSE(m1, t *Matrix) float32 {
	l1 := len(m1.W)
	l2 := len(t.W)
	if l1 != l2 {
		panic(fmt.Errorf("matadd number of elements must be equal numel(m1)=%d must be equal numel(m2)=%d", l1, l2))
	}
	var mse float32
	var tmp float32
	for i := 0; i < l1; i++ {
		tmp = m1.W[i] - t.W[i]
		mse += tmp * tmp
	}
	mse /= float32(l1)

	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			b := 2.0 / float32(l1)
			for i := 0; i < l1; i++ { // loop over rows of m1
				m1.DW[i] += b * (m1.W[i] - t.W[i]) // 1/Columns * sum((x-t)^2) derivative keep it math correct no Ng's
			}
		})
	}

	return mse
}

//Crossentropy takes logits vector and list of label id
func (g *Graph) Crossentropy(m1 *Matrix, label int) (cost, perplexity, probability float32) {
	l1 := len(m1.W)

	if label < 0 || label >= l1 {
		panic(fmt.Errorf("label value must be within range [0;numel(m1)]=[0;%d] but %d given", l1-1, label))
	}
	// compute probabilities
	probabilities := Softmax(m1)
	probability = probabilities.W[label]
	perplexity = float32(-math.Log2(float64(probability)))
	cost = float32(-math.Log(float64(probability)))
	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			for i := range m1.DW {
				m1.DW[i] += probabilities.W[i]
			}
			m1.DW[label] -= 1
		})
	}

	return
}
