package gortex

import "fmt"
import "github.com/vseledkin/gortex/assembler"

// Gated recurrent unit

type GRU struct {
	Wz *Matrix
	Uz *Matrix
	Bz *Matrix

	Wr *Matrix
	Ur *Matrix
	Br *Matrix

	Wh *Matrix
	Uh *Matrix
	Bh *Matrix

	Who *Matrix
}

func (gru *GRU) ForgetGateTrick(v float32) {
	if gru.Bz != nil {
		assembler.Sset(v, gru.Bz.W)
	}
}

func MakeGRU(x_size, h_size, out_size int) *GRU {
	rnn := new(GRU)
	rnn.Wz = RandXavierMat(h_size, x_size)
	rnn.Uz = RandXavierMat(h_size, h_size)
	rnn.Bz = RandXavierMat(h_size, 1)

	rnn.Wr = RandXavierMat(h_size, x_size)
	rnn.Ur = RandXavierMat(h_size, h_size)
	rnn.Br = RandXavierMat(h_size, 1)

	rnn.Wh = RandXavierMat(h_size, x_size)
	rnn.Uh = RandXavierMat(h_size, h_size)
	rnn.Bh = RandXavierMat(h_size, 1)

	rnn.Who = RandXavierMat(out_size, h_size)
	return rnn
}

func (rnn *GRU) GetParameters(namespace string) map[string]*Matrix {
	return map[string]*Matrix{
		namespace + "_Wz": rnn.Wz,
		namespace + "_Uz": rnn.Uz,
		namespace + "_Bz": rnn.Bz,

		namespace + "_Wr": rnn.Wr,
		namespace + "_Ur": rnn.Ur,
		namespace + "_Br": rnn.Br,

		namespace + "_Wh": rnn.Wh,
		namespace + "_Uh": rnn.Uh,
		namespace + "_Bh": rnn.Bh,

		namespace + "_Who": rnn.Who,
	}
}

func (rnn *GRU) SetParameters(namespace string, parameters map[string]*Matrix) error {
	for k, v := range rnn.GetParameters(namespace) {
		fmt.Printf("Look for %s parameters\n", k)
		if m, ok := parameters[k]; ok {
			fmt.Printf("Got %s parameters\n", k)
			copy(v.W, m.W)
		} else {
			return fmt.Errorf("Model geometry is not compatible, parameter %s is unknown", k)
		}
	}
	return nil
}

func (rnn *GRU) Step(g *Graph, x, h_prev *Matrix) (h, y *Matrix) {
	// make GRU computation graph at one time-step
	zt := g.Sigmoid(g.Add(g.Add(g.Mul(rnn.Wz, x), g.Mul(rnn.Uz, h_prev)), rnn.Bz))
	rt := g.Sigmoid(g.Add(g.Add(g.Mul(rnn.Wr, x), g.Mul(rnn.Ur, h_prev)), rnn.Br))

	ht := g.Tanh(g.Add(g.Add(g.Mul(rnn.Wh, x), g.Mul(rnn.Uh, g.EMul(rt, h_prev))), rnn.Bh))
	//h = g.InstanceNormalization(g.Add(g.EMul(zt, h_prev), g.EMul(g.Sub(zt.OnesAs(), zt), ht)))
	h = g.Add(g.EMul(zt, h_prev), g.EMul(g.Sub(zt.OnesAs(), zt), ht))

	y = g.Mul(rnn.Who, g.Tanh(h))
	return
}
