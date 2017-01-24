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
		panic(fmt.Errorf("matadd number of elements must be equal numel(m1)=%Columns must be equal numel(m2)=%Columns", l1, l2))
	}

	out := Mat(m1.Rows, m1.Columns)
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
		panic(fmt.Errorf("matmul dimensions misaligned m1.columns=%Columns must be equal m2.rows=%Columns", m1.Columns, m2.Rows))
	}

	n := m1.Rows
	d := m2.Columns
	out := Mat(n, d)
	for i := 0; i < m1.Rows; i++ { // loop over rows of m1
		for j := 0; j < m2.Columns; j++ { // loop over cols of m2
			dot := 0.0
			for k := 0; k < m1.Columns; k++ { // dot product loop
				dot += m1.W[m1.Columns*i + k] * m2.W[m2.Columns*k + j]
			}
			out.W[d*i + j] = dot
		}
	}

	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			for i := 0; i < m1.Rows; i++ { // loop over rows of m1
				for j := 0; j < m2.Columns; j++ { // loop over cols of m2
					for k := 0; k < m1.Columns; k++ { // dot product loop
						b := out.DW[d*i + j]
						m1.DW[m1.Columns*i + k] += m2.W[m2.Columns*k + j] * b
						m2.DW[m2.Columns*k + j] += m1.W[m1.Columns*i + k] * b
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
		panic(fmt.Errorf("matadd number of elements must be equal numel(m1)=%Columns must be equal numel(m2)=%Columns", l1, l2))
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
				m1.DW[i] = b * (m1.W[i] - t.W[i]) // 1/Columns * sum((x-t)^2) derivative keep it math correct no Ng's
			}
		})
	}

	return mse
}
