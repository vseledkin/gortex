package gortex

import (
	"fmt"
	"math"
)

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

func (g *Graph) Add(m1, m2 *Matrix) *Matrix {
	l1 := len(m1.W)
	l2 := len(m2.W)
	if l1 != l2 {
		panic(fmt.Errorf("matadd number of elements must be equal numel(m1)=%d must be equal numel(m2)=%d", l1, l2))
	}

	out := Mat(m1.n, m1.d)
	for i := 0; i < l1; i++ {
		out.W[i] = m1.W[i] + m2.W[i]
	}

	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			for i := 0; i < l1; i++ {
				m1.dw[i] += out.dw[i]
				m2.dw[i] += out.dw[i]
			}
		})
	}
	return out
}

func (g *Graph) Mul(m1, m2 *Matrix) *Matrix {
	// multiply matrices m1 * m2
	if m1.d != m2.n {
		panic(fmt.Errorf("matmul dimensions misaligned m1.columns=%d must be equal m2.rows=%d", m1.d, m2.n))
	}

	n := m1.n
	d := m2.d
	out := Mat(n, d)
	for i := 0; i < m1.n; i++ { // loop over rows of m1
		for j := 0; j < m2.d; j++ { // loop over cols of m2
			dot := 0.0
			for k := 0; k < m1.d; k++ { // dot product loop
				dot += m1.W[m1.d*i + k] * m2.W[m2.d*k + j]
			}
			out.W[d*i + j] = dot
		}
	}

	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			for i := 0; i < m1.n; i++ { // loop over rows of m1
				for j := 0; j < m2.d; j++ { // loop over cols of m2
					for k := 0; k < m1.d; k++ { // dot product loop
						b := out.dw[d*i + j]
						m1.dw[m1.d*i + k] += m2.W[m2.d*k + j] * b
						m2.dw[m2.d*k + j] += m1.W[m1.d*i + k] * b
					}
				}
			}
		})
	}
	return out
}

func (g *Graph) MSE(m1, t *Matrix) float64 {
	l1 := len(m1.W)
	l2 := len(t.W)
	if l1 != l2 {
		panic(fmt.Errorf("matadd number of elements must be equal numel(m1)=%d must be equal numel(m2)=%d", l1, l2))
	}
	mse := 0.0
	for i := 0; i < l1; i++ {
		mse += math.Pow(m1.W[i]-t.W[i], 2)
	}
	mse /= float64(l1)

	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			b := 2.0 / float64(l1)
			for i := 0; i < l1; i++ { // loop over rows of m1
				m1.dw[i] = b * (m1.W[i] - t.W[i]) // 1/d * sum((x-t)^2) derivative keep it math correct no Ng's
			}
		})
	}

	return mse
}
