package gortex

import (
	"fmt"
	"math"
	"sync"

	"github.com/vseledkin/gortex/assembler"
)

const epsilon = 1e-9

var mux sync.Mutex

type Graph struct {
	NeedsBackprop bool
	Print         bool

	// this will store a list of functions that perform backprop,
	// in their forward pass order. So in backprop we will go
	// backwards and evoke each one
	backprop []func()
}

func (g *Graph) Backward() {
	for i := len(g.backprop) - 1; i >= 0; i-- {
		//mux.Lock()
		g.backprop[i]() // tick!
		//mux.Unlock()
	}
}

func (g *Graph) InstanceNormalization(m *Matrix) *Matrix {
	mean, variance := Moments(m)
	stdDev := assembler.Sqrt(variance)
	out := m.SameAs()
	for i := range m.W {
		out.W[i] = (m.W[i] - mean) / assembler.Sqrt(stdDev*stdDev+epsilon)
	}
	if g.NeedsBackprop {

		//dbeta := assembler.L1(m.DW)
		/*

			//dbeta = np.sum(dy, axis=0)
			//dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy, axis=0)
			dh = (1. / N) *  sqrt(var + eps) * (N * dy - np.sum(dy)
			- (h - mu) / (var + eps) * np.sum(dy * (h - mu)))*/

		g.backprop = append(g.backprop, func() {
			N := float32(len(m.W))
			/*
				scale := (1.0 - 1.0/float32(len(m.W))) / (stdDev + epsilon)
				for i := range m.W {
					m.DW[i] += scale * out.DW[i]
				}
			*/
			sum := assembler.Sum(out.DW)
			var sum2 float32
			for i := range m.W {
				sum2 += out.DW[i] * (m.W[i] - mean)
			}
			for i := range m.W {
				m.DW[i] += assembler.Sqrt(variance+epsilon) / N * (N*out.DW[i] - sum - (m.W[i]-mean)/(variance+epsilon)*sum2)
			}
		})
	}
	return out
}

func (g *Graph) Tanh(m *Matrix, messages ...string) *Matrix {
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
			if len(messages) > 0 && g.Print {
				fmt.Printf("%s Tanh In(%p N:%f GN:%f) Out(%p N:%f GN:%f)\n",
					messages[0], m, m.Norm(), m.NormGradient(), out, out.Norm(), out.NormGradient())
			}
		})
	}
	return out
}

func (g *Graph) Lookup(lt *Matrix, i int) *Matrix {
	// pickup rows as embeddings for speed so lt Matrix is treated as column major
	out := Mat(lt.Rows, 1)
	offset := i * lt.Rows
	// we can point to region in slice instead of copy
	out.W = lt.W[offset : offset+lt.Rows]

	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			// gradient landing
			assembler.Sxpy(out.DW, lt.DW[offset:offset+lt.Rows])
		})
	}
	return out
}

//Softmax probability distribution interpretation of any vector/matrix
func (g *Graph) Softmax(m *Matrix) *Matrix {
	out := Mat(m.Rows, m.Columns) // probability volume
	maxval := m.W[assembler.Ismax(m.W)]
	for i := range m.W {
		out.W[i] = float32(math.Exp(float64(m.W[i] - maxval)))
	}
	sum := assembler.Sum(out.W)
	assembler.Sscale(1/sum, out.W)

	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			assembler.Sxmuleyplusz(out.DW, out.DW, m.DW)
			//for i := range m.W {
			//	m.DW[i] += out.W[i] * out.DW[i]
			//}
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
			// grad for z = sigmoid(x) is sigmoid(x)(1 - sigmoid(x))
			assembler.Sigmoidbackprop(1, out.W, out.DW, m.DW)
			//for i := range m.W {
			//	m.DW[i] += out.W[i] * (1.0 - out.W[i]) * out.DW[i]
			//}
		})
	}
	return out
}

func (g *Graph) Add(m1, m2 *Matrix, messages ...string) *Matrix {
	l1 := len(m1.W)
	l2 := len(m2.W)
	if l1 != l2 {
		panic(fmt.Errorf("matadd number of elements must be equal numel(m1)=%d must be equal numel(m2)=%d", l1, l2))
	}

	out := m1.CopyAs() // copy only weights not gradients
	assembler.Sxpy(m2.W, out.W)
	/*
		out := m1.SameAs()
		for i := 0; i < l1; i++ {
			out.W[i] = m1.W[i] + m2.W[i]
		}*/
	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			assembler.Sxpy(out.DW, m1.DW)
			assembler.Sxpy(out.DW, m2.DW)
			if len(messages) > 0 && g.Print {
				fmt.Printf("%s Add In1(%p N:%f GN:%f) In2(%p N:%f GN:%f) Out(%p N:%f GN:%f)\n",
					messages[0], m1, m1.Norm(), m1.NormGradient(), m2, m2.Norm(), m2.NormGradient(), out, out.Norm(), out.NormGradient())
			}
		})
	}
	return out
}

func (g *Graph) Sub(m1, m2 *Matrix) *Matrix {
	l1 := len(m1.W)
	l2 := len(m2.W)
	if l1 != l2 {
		panic(fmt.Errorf("matsub number of elements must be equal numel(m1)=%d must be equal numel(m2)=%d", l1, l2))
	}

	out := m1.SameAs()
	for i := 0; i < l1; i++ {
		out.W[i] = m1.W[i] - m2.W[i]
	}

	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			for i := 0; i < l1; i++ {
				m1.DW[i] += out.DW[i]
				m2.DW[i] += -out.DW[i]
			}
		})
	}
	return out
}

func (g *Graph) mulv(m1, m2 *Matrix, messages ...string) *Matrix {
	// multiply matrix and vector m1 * m2
	if m1.Columns != m2.Rows {
		panic(fmt.Errorf("matmul dimensions misaligned m1.columns=%d must be equal m2.rows=%d", m1.Columns, m2.Rows))
	}

	out := Mat(m1.Rows, 1)
	for i := 0; i < m1.Rows; i++ { // loop over rows of m1
		out.W[i] = assembler.Sdot(m1.W[m1.Columns*i:m1.Columns*i+m1.Columns], m2.W)
	}

	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			for i := 0; i < m1.Rows; i++ { // loop over rows of m1
				assembler.Saxpy(out.DW[i], m2.W, m1.DW[m1.Columns*i:m1.Columns*i+m1.Columns])
				assembler.Saxpy(out.DW[i], m1.W[m1.Columns*i:m1.Columns*i+m1.Columns], m2.DW)
			}
			if len(messages) > 0 && g.Print {
				fmt.Printf("%s Mul In1(%p N:%f GN:%f) In2(%p N:%f GN:%f) Out(%p N:%f GN:%f)\n",
					messages[0], m1, m1.Norm(), m1.NormGradient(), m2, m2.Norm(), m2.NormGradient(), out, out.Norm(), out.NormGradient())
			}
		})
	}
	return out
}

func (g *Graph) Mul(m1, m2 *Matrix, messages ...string) *Matrix {
	// multiply matrices m1 * m2
	if m1.Columns != m2.Rows {
		panic(fmt.Errorf("matmul dimensions misaligned m1.columns=%d must be equal m2.rows=%d", m1.Columns, m2.Rows))
	}
	if m2.Columns == 1 { // use highly optimized special case when m2 is vector
		return g.mulv(m1, m2)
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
			//if len(messages) > 0 {
			//	fmt.Printf("%s Norm:%f GradientNorm:%f\n", messages[0], out.Norm(), out.NormGradient())
			//}
			for i := 0; i < m1.Rows; i++ { // loop over rows of m1
				for j := 0; j < m2.Columns; j++ { // loop over cols of m2
					b := out.DW[m2.Columns*i+j]
					for k := 0; k < m1.Columns; k++ { // dot product loop
						m1.DW[m1.Columns*i+k] += m2.W[m2.Columns*k+j] * b
						m2.DW[m2.Columns*k+j] += m1.W[m1.Columns*i+k] * b
					}
				}
			}
			if len(messages) > 0 && g.Print {
				fmt.Printf("%s Mul In1(%p N:%f GN:%f) In2(%p N:%f GN:%f) Out(%p N:%f GN:%f)\n",
					messages[0], m1, m1.Norm(), m1.NormGradient(), m2, m2.Norm(), m2.NormGradient(), out, out.Norm(), out.NormGradient())
			}
		})
	}
	return out
}

// Sum of weights of x
func (g *Graph) Sum(x *Matrix) *Matrix {
	out := Mat(x.Rows, x.Columns)
	out.W[0] = assembler.Sum(x.W)
	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			for i := range x.DW {
				x.DW[i] += out.DW[0]
			}
		})
	}
	return out
}

// Add non learnable constant to x
func (g *Graph) AddConstant(c float32, x *Matrix) *Matrix {
	out := x.ConstantAs(c)
	assembler.Sxpy(x.W, out.W)
	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			assembler.Sxpy(out.DW, x.DW)
		})
	}
	return out
}

// Multiply x by a non learnable constant
func (g *Graph) MulConstant(c float32, x *Matrix) *Matrix {
	out := x.CopyAs()
	assembler.Sscale(c, out.W)
	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			assembler.Saxpy(c, out.DW, x.DW)
		})
	}
	return out
}

// Take elementwise exponent of x
func (g *Graph) Exp(x *Matrix) *Matrix {
	out := Mat(x.Rows, x.Columns)
	for i := range x.W {
		out.W[i] = float32(math.Exp(float64(x.W[i])))
	}
	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			assembler.Sxmuleyplusz(out.DW, out.W, x.DW)
		})
	}
	return out
}

// EMul elementwise matrix matrix multiplication
func (g *Graph) EMul(m1, m2 *Matrix, messages ...string) *Matrix {
	l1 := len(m1.W)
	l2 := len(m2.W)
	if l1 != l2 {
		panic(fmt.Errorf("emul number of elements must be equal numel(m1)=%d must be equal numel(m2)=%d", l1, l2))
	}

	/*out := m1.SameAs()
	for i := range m1.W {
		out.W[i] = m1.W[i] * m2.W[i]
	}
	*/
	out := m1.CopyAs()
	assembler.Sxmuley(m2.W, out.W)

	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			assembler.Sxmuleyplusz(m2.W, out.DW, m1.DW)
			assembler.Sxmuleyplusz(m1.W, out.DW, m2.DW)
			//for i := range m1.W {
			//	m1.DW[i] += m2.W[i] * out.DW[i]
			//	m2.DW[i] += m1.W[i] * out.DW[i]
			//}
			if len(messages) > 0 && g.Print {
				fmt.Printf("%s EMul In1(%p N:%f GN:%f) In2(%p N:%f GN:%f) Out(%p N:%f GN:%f)\n",
					messages[0], m1, m1.Norm(), m1.NormGradient(), m2, m2.Norm(), m2.NormGradient(), out, out.Norm(), out.NormGradient())
			}
		})
	}
	return out
}

// Concatenate two or more vectors
func (g *Graph) Concat(m ...*Matrix) *Matrix {
	L := len(m)
	if L < 2 {
		panic(fmt.Errorf("concat is for two or more vectors but %d vectors given", L))
	}
	L = 0
	for _, v := range m {
		L += v.Rows
	}
	out := Mat(L, 1)
	// copy in natural order
	L = 0
	for _, v := range m {
		for _, f := range v.W {
			out.W[L] = f
			L++
		}
	}
	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			// copy gradients
			L = 0
			for _, v := range m {
				for i := range v.DW {
					v.DW[i] = out.DW[L]
					L++
				}
			}
		})
	}
	return out
}
//MSE mean square error loss function
func (g *Graph) MSE(m1, t *Matrix) float32 {
	l1 := len(m1.W)
	l2 := len(t.W)
	if l1 != l2 {
		panic(fmt.Errorf("mse number of elements must be equal numel(m1)=%d must be equal numel(m2)=%d", l1, l2))
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

//Crossentropy loss function takes logits vector and list of label ids
func (g *Graph) Crossentropy(m1 *Matrix, label uint) (cost, probability float32) {

	if label >= uint(len(m1.W)) {
		panic(fmt.Errorf("label value must be within range [0;numel(m1)]=[0;%d] but %d given", len(m1.W)-1, label))
	}
	// compute probabilities
	probabilities := Softmax(m1)
	probability = probabilities.W[label]
	cost = float32(-math.Log(float64(probability)))
	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			assembler.Sxpy(probabilities.W, m1.DW)
			//for i := range m1.DW {
			//	m1.DW[i] += probabilities.W[i]
			//}
			m1.DW[label] -= 1
		})
	}

	return
}
