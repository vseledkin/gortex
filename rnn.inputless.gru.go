package gortex

import "fmt"
import "github.com/vseledkin/gortex/assembler"

// Gated recurrent unit

type InputlessGRU struct {
	Uz *Matrix
	Bz *Matrix

	Ur *Matrix
	Br *Matrix

	Uh *Matrix
	Bh *Matrix

	Who *Matrix
}

func (gru *InputlessGRU) ForgetGateTrick(v float32) {
	if gru.Bz != nil {
		assembler.Sset(v, gru.Bz.W)
	}
}

func MakeInputlessGRU(h_size, out_size int) *InputlessGRU {
	rnn := new(InputlessGRU)

	rnn.Uz = RandXavierMat(h_size, h_size)
	rnn.Bz = RandXavierMat(h_size, 1)

	rnn.Ur = RandXavierMat(h_size, h_size)
	rnn.Br = RandXavierMat(h_size, 1)

	rnn.Uh = RandXavierMat(h_size, h_size)
	rnn.Bh = RandXavierMat(h_size, 1)

	rnn.Who = RandXavierMat(out_size, h_size)
	return rnn
}

func (rnn *InputlessGRU) GetParameters(namespace string) map[string]*Matrix {
	return map[string]*Matrix{

		namespace + "_Uz": rnn.Uz,
		namespace + "_Bz": rnn.Bz,

		namespace + "_Ur": rnn.Ur,
		namespace + "_Br": rnn.Br,

		namespace + "_Uh": rnn.Uh,
		namespace + "_Bh": rnn.Bh,

		namespace + "_Who": rnn.Who,
	}
}

func (rnn *InputlessGRU) SetParameters(namespace string, parameters map[string]*Matrix) error {
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

func (rnn *InputlessGRU) Step(g *Graph, h_prev *Matrix) (h, y *Matrix) {
	// make GRU computation graph at one time-step
	zt := g.Sigmoid(g.Add(g.Mul(rnn.Uz, h_prev), rnn.Bz))
	rt := g.Sigmoid(g.Add(g.Mul(rnn.Ur, h_prev), rnn.Br))

	ht := g.Tanh(g.Add(g.Mul(rnn.Uh, g.EMul(rt, h_prev)), rnn.Bh))
	//h = g.InstanceNormalization(g.Add(g.EMul(zt, h_prev), g.EMul(g.Sub(zt.OnesAs(), zt), ht)))
	h = g.Add(g.EMul(zt, h_prev), g.EMul(g.Sub(zt.OnesAs(), zt), ht))

	y = g.Mul(rnn.Who, g.Tanh(h))
	return
}
