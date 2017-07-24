package gortex

import "fmt"
import "github.com/vseledkin/gortex/assembler"

// Gated recurrent unit

type OutputlessGRU struct {
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

func (gru *OutputlessGRU) ForgetGateTrick(v float32) {
	if gru.Bz != nil {
		assembler.Sset(v, gru.Bz.W)
	}
}

func MakeOutputlessGRU(x_size, h_size int) *OutputlessGRU {
	rnn := new(OutputlessGRU)
	rnn.Wz = RandXavierMat(h_size, x_size)
	rnn.Uz = RandXavierMat(h_size, h_size)
	rnn.Bz = RandXavierMat(h_size, 1)

	rnn.Wr = RandXavierMat(h_size, x_size)
	rnn.Ur = RandXavierMat(h_size, h_size)
	rnn.Br = RandXavierMat(h_size, 1)

	rnn.Wh = RandXavierMat(h_size, x_size)
	rnn.Uh = RandXavierMat(h_size, h_size)
	rnn.Bh = RandXavierMat(h_size, 1)

	return rnn
}

func (rnn *OutputlessGRU) GetParameters(namespace string) map[string]*Matrix {
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

	}
}

func (rnn *OutputlessGRU) SetParameters(namespace string, parameters map[string]*Matrix) error {
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

func (rnn *OutputlessGRU) Step(g *Graph, x, h_prev *Matrix) (h *Matrix) {
	// make GRU computation graph at one time-step
	zt := g.Sigmoid(g.Add(g.Add(g.Mul(rnn.Wz, x), g.Mul(rnn.Uz, h_prev)), rnn.Bz))
	rt := g.Sigmoid(g.Add(g.Add(g.Mul(rnn.Wr, x), g.Mul(rnn.Ur, h_prev)), rnn.Br))

	ht := g.Tanh(g.Add(g.Add(g.Mul(rnn.Wh, x), g.Mul(rnn.Uh, g.EMul(rt, h_prev))), rnn.Bh))
	//h = g.InstanceNormalization(g.Add(g.EMul(zt, h_prev), g.EMul(g.Sub(zt.OnesAs(), zt), ht)))
	h = g.Add(g.EMul(zt, h_prev), g.EMul(g.Sub(zt.OnesAs(), zt), ht))

	return
}
