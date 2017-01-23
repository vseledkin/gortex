package gortex

import "fmt"

type Graph struct {
}

func (g *Graph) Add(x, y *Matrix) *Matrix {
	return new(Matrix)
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
				dot += m1.w[m1.d*i + k] * m2.w[m2.d*k + j]
			}
			out.w[d*i + j] = dot
		}
	}
	return out
}
