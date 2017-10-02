package gortex

import (
	"fmt"
	"math"
	"sync"

	"github.com/vseledkin/gortex/assembler"
)

const epsilon = 1e-9

var mutex sync.Mutex

type Graph struct {
	NeedsBackprop bool
	NeedsParallel bool
	gradients     map[*Matrix][]float32
	Print         bool

	// this will store a list of functions that perform backprop,
	// in their forward pass order. So in backprop we will go
	// backwards and evoke each one
	backprop []func()
}

func (g *Graph) grad(m *Matrix) []float32 {
	if g.NeedsParallel {
		if g.gradients == nil {
			g.gradients = make(map[*Matrix][]float32)
		}
		if grad, ok := g.gradients[m]; ok {
			return grad
		} else {
			grad = make([]float32, m.Numel())
			g.gradients[m] = grad
			return grad
		}
	} else {
		return m.DW
	}
}

func (g *Graph) Backward() {
	for i := len(g.backprop) - 1; i >= 0; i-- {
		g.backprop[i]() // tick!
	}
	if g.NeedsParallel {
		// accumulate gradients
		for m, grad := range g.gradients {
			// it is assumed that here is the only place where
			// multiple threads can mix everything up
			mutex.Lock()
			assembler.Sxpy(grad, m.DW)
			mutex.Unlock()
		}
	}
}

func (g *Graph) InstanceNormalization(m *Matrix) *Matrix {
	if g.NeedsParallel {
		panic("Does not support parallel execution")
	}
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

func (g *Graph) Tanh(m *Matrix) *Matrix {
	// tanh non linearity
	out := m.SameAs()
	for i := range m.W {
		out.W[i] = float32(math.Tanh(float64(m.W[i])))
	}
	if g.NeedsBackprop {
		outDW := g.grad(out)
		mDW := g.grad(m)
		g.backprop = append(g.backprop, func() {
			// todo: speedup
			for i := range m.W {
				// grad for z = tanh(x) is (1 - z^2)
				mDW[i] += (1.0 - out.W[i]*out.W[i]) * outDW[i]
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
	// we can point to region in slice instead of copy
	out.DW = lt.DW[offset : offset+lt.Rows]
	// backprop is transparent and not needed
	//if g.NeedsBackprop {}
	return out
}

func (g *Graph) Lookup2(lt *Matrix, i int, count map[int]int) *Matrix {
	// register lookup
	count[i]++
	// pickup rows as embeddings for speed so lt Matrix is treated as column major
	out := Mat(lt.Rows, 1)
	offset := i * lt.Rows
	// we can point to region in slice instead of copy
	out.W = lt.W[offset : offset+lt.Rows]
	// we can point to region in slice instead of copy
	out.DW = lt.DW[offset : offset+lt.Rows]
	// backprop is transparent and not needed
	//if g.NeedsBackprop {}
	return out
}

//Softmax probability distribution interpretation of any vector/matrix
func (g *Graph) Softmax(m *Matrix) *Matrix {

	if g.NeedsParallel {
		panic("Does not support parallel execution")
	}
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
		outDW := g.grad(out)
		mDW := g.grad(m)
		g.backprop = append(g.backprop, func() {
			// grad for z = sigmoid(x) is sigmoid(x)(1 - sigmoid(x))
			assembler.Sigmoidbackprop(1, out.W, outDW, mDW)
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

	out := m1.CopyAs() // copy only weights not gradients
	assembler.Sxpy(m2.W, out.W)

	if g.NeedsBackprop {
		outDW := g.grad(out)
		m1DW := g.grad(m1)
		m2DW := g.grad(m2)
		g.backprop = append(g.backprop, func() {
			assembler.Sxpy(outDW, m1DW)
			assembler.Sxpy(outDW, m2DW)
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

	out := m1.CopyAs()
	assembler.Saxpy(-1, m2.W, out.W)

	if g.NeedsBackprop {
		outDW := g.grad(out)
		m1DW := g.grad(m1)
		m2DW := g.grad(m2)
		g.backprop = append(g.backprop, func() {
			assembler.Sxpy(outDW, m1DW)
			assembler.Saxpy(-1, outDW, m2DW)
		})
	}
	return out
}

func (g *Graph) mulv(m1, m2 *Matrix) *Matrix {
	// multiply matrix and vector m1 * m2
	if m1.Columns != m2.Rows {
		panic(fmt.Errorf("matmul dimensions misaligned m1.columns=%d must be equal m2.rows=%d", m1.Columns, m2.Rows))
	}

	out := Mat(m1.Rows, 1)
	for i := 0; i < m1.Rows; i++ { // loop over rows of m1
		out.W[i] = assembler.Sdot(m1.W[m1.Columns*i:m1.Columns*i+m1.Columns], m2.W)
	}

	if g.NeedsBackprop {
		outDW := g.grad(out)
		m1DW := g.grad(m1)
		m2DW := g.grad(m2)
		g.backprop = append(g.backprop, func() {
			for i := 0; i < m1.Rows; i++ { // loop over rows of m1
				assembler.Saxpy(outDW[i], m2.W, m1DW[m1.Columns*i:m1.Columns*i+m1.Columns])
				assembler.Saxpy(outDW[i], m1.W[m1.Columns*i:m1.Columns*i+m1.Columns], m2DW)
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
		outDW := g.grad(out)
		m1DW := g.grad(m1)
		m2DW := g.grad(m2)
		g.backprop = append(g.backprop, func() {
			for i := 0; i < m1.Rows; i++ { // loop over rows of m1
				for j := 0; j < m2.Columns; j++ { // loop over cols of m2
					b := outDW[m2.Columns*i+j]
					for k := 0; k < m1.Columns; k++ { // dot product loop
						m1DW[m1.Columns*i+k] += m2.W[m2.Columns*k+j] * b
						m2DW[m2.Columns*k+j] += m1.W[m1.Columns*i+k] * b
					}
				}
			}
		})
	}
	return out
}

// Sum of weights of x
func (g *Graph) Sum(m *Matrix) *Matrix {
	out := Mat(1, 1)
	out.W[0] = assembler.Sum(m.W)
	if g.NeedsBackprop {
		outDW := g.grad(out)
		mDW := g.grad(m)
		g.backprop = append(g.backprop, func() {
			for i := range mDW {
				mDW[i] += outDW[0]
			}
		})
	}
	return out
}

// Add non learnable constant to x
func (g *Graph) AddConstant(c float32, m *Matrix) *Matrix {
	out := m.ConstantAs(c)
	assembler.Sxpy(m.W, out.W)
	if g.NeedsBackprop {
		outDW := g.grad(out)
		mDW := g.grad(m)
		g.backprop = append(g.backprop, func() {
			assembler.Sxpy(outDW, mDW)
		})
	}
	return out
}

// Multiply x by a non learnable constant
func (g *Graph) MulConstant(c float32, m *Matrix) *Matrix {
	out := m.CopyAs()
	assembler.Sscale(c, out.W)
	if g.NeedsBackprop {
		outDW := g.grad(out)
		mDW := g.grad(m)
		g.backprop = append(g.backprop, func() {
			assembler.Saxpy(c, outDW, mDW)
		})
	}
	return out
}

// Take elementwise exponent of x
func (g *Graph) Exp(m *Matrix) *Matrix {
	out := Mat(m.Rows, m.Columns)
	for i := range m.W {
		out.W[i] = float32(math.Exp(float64(m.W[i])))
	}
	if g.NeedsBackprop {
		outDW := g.grad(out)
		mDW := g.grad(m)
		g.backprop = append(g.backprop, func() {
			assembler.Sxmuleyplusz(outDW, out.W, mDW)
		})
	}
	return out
}

// Relu
func (g *Graph) Relu(x *Matrix) *Matrix {
	out := Mat(x.Rows, x.Columns)
	for i := range x.W {
		if x.W[i] < 0 {
			out.W[i] = 0
		} else {
			out.W[i] = x.W[i]
		}
	}
	if g.NeedsBackprop {
		outDW := g.grad(out)
		xDW := g.grad(x)
		g.backprop = append(g.backprop, func() {
			for i := range x.W {
				if x.W[i] < 0 {
					xDW[i] = 0
				} else {
					xDW[i] += outDW[i]
				}
			}
		})
	}
	return out
}

// Self normalizing Elu-Selu implementation
func (g *Graph) Selu(x *Matrix) *Matrix {

	if g.NeedsParallel {
		panic("Does not support parallel execution")
	}
	bias := float32(1.6732632423543772848170429916717)
	scale := float32(1.0507009873554804934193349852946)
	out := Mat(x.Rows, x.Columns)
	for i := range x.W {
		if x.W[i] > 0 {
			out.W[i] = scale * x.W[i]
		} else {
			out.W[i] = scale * (bias*float32(math.Exp(float64(x.W[i]))) - bias)
		}
	}
	if g.NeedsBackprop {
		g.backprop = append(g.backprop, func() {
			for i := range x.W {
				if x.W[i] > 0 {
					x.DW[i] += scale * out.DW[i]
				} else {
					x.DW[i] += (out.W[i] + scale*bias) * out.DW[i]
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

	out := m1.CopyAs()
	assembler.Sxmuley(m2.W, out.W)

	if g.NeedsBackprop {
		outDW := g.grad(out)
		m1DW := g.grad(m1)
		m2DW := g.grad(m2)
		g.backprop = append(g.backprop, func() {
			assembler.Sxmuleyplusz(m2.W, outDW, m1DW)
			assembler.Sxmuleyplusz(m1.W, outDW, m2DW)
		})
	}
	return out
}

// Concatenate two or more vectors
func (g *Graph) Concat(m ...*Matrix) *Matrix {

	if g.NeedsParallel {
		panic("Does not support parallel execution")
	}
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

	if g.NeedsParallel {
		panic("Does not support parallel execution")
	}
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
func (g *Graph) Crossentropy(m *Matrix, label uint) (cost, probability float32) {
	if label >= uint(len(m.W)) {
		panic(fmt.Errorf("label value must be within range [0;numel(m1)]=[0;%d] but %d given", len(m.W)-1, label))
	}
	// compute probabilities
	probabilities := Softmax(m)
	probability = probabilities.W[label]
	cost = float32(-math.Log(float64(probability)))
	if g.NeedsBackprop {
		mDW := g.grad(m)
		g.backprop = append(g.backprop, func() {
			assembler.Sxpy(probabilities.W, mDW)
			mDW[label] -= 1
		})
	}
	return
}
