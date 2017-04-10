package gortex

import "fmt"

// DeltaRNN cell https://arxiv.org/pdf/1703.08864.pdf
type DeltaRNN struct {
	Wx    *Matrix
	Wh    *Matrix
	Wo    *Matrix
	BiasR *Matrix
	Bias  *Matrix
	A     *Matrix
	B     *Matrix
	C     *Matrix
}

// MakeDeltaRNN create new cell
func MakeDeltaRNN(x_size, h_size, out_size int) *DeltaRNN {
	net := new(DeltaRNN)
	net.Wx = RandXavierMat(h_size, x_size)
	net.Wh = RandXavierMat(h_size, h_size)
	net.Wo = RandXavierMat(out_size, h_size)
	net.BiasR = RandXavierMat(h_size, 1)
	net.Bias = RandXavierMat(h_size, 1)
	net.A = RandXavierMat(h_size, 1)
	net.B = RandXavierMat(h_size, 1)
	net.C = RandXavierMat(h_size, 1)
	return net
}

func (rnn *DeltaRNN) GetParameters(namespace string) map[string]*Matrix {
	return map[string]*Matrix{
		namespace + "_Wx":    rnn.Wx,
		namespace + "_Wh":    rnn.Wh,
		namespace + "_Wo":    rnn.Wo,
		namespace + "_A":     rnn.A,
		namespace + "_B":     rnn.B,
		namespace + "_C":     rnn.C,
		namespace + "_BiasR": rnn.BiasR,
		namespace + "_Bias":  rnn.Bias}
}

func (rnn *DeltaRNN) SetParameters(namespace string, parameters map[string]*Matrix) error {
	for k, v := range rnn.GetParameters(namespace) {
		fmt.Printf("Look for %s parameters\n", k)
		if m, ok := parameters[k]; ok {
			fmt.Printf("Got %s parameters\n", k)
			v.W = m.W
		} else {
			return fmt.Errorf("Model geometry is not compatible, parameter %s is unknown", k)
		}
	}
	return nil
}

func (rnn *DeltaRNN) Step(g *Graph, x, h_prev *Matrix) (h, y *Matrix) {
	// make DeltaRNN computation graph at one time-step
	xx := g.Mul(rnn.Wx, x)
	hh := g.Mul(rnn.Wh, h_prev)
	r := g.Sigmoid(g.Add(xx, rnn.BiasR))

	// Hadamard product
	z1 := g.EMul(rnn.A, xx)
	z2 := g.EMul(rnn.B, hh)
	z3 := g.EMul(rnn.C, g.EMul(xx, hh))
	z := g.Add(g.Add(g.Add(z1, z2), z3), rnn.Bias)

	h = g.Add(g.EMul(r, h_prev), g.EMul(g.Sub(r.OnesAs(), r), z))

	y = g.Mul(rnn.Wo, g.Tanh(h))
	return
}
