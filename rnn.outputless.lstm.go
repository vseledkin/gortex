package gortex

import (
	"fmt"

	"github.com/vseledkin/gortex/assembler"
)

// Long Short Term Memory cell

type OutputlessLSTM struct {
	Wf *Matrix
	Uf *Matrix
	Bf *Matrix

	Wi *Matrix
	Ui *Matrix
	Bi *Matrix

	Wo *Matrix
	Uo *Matrix
	Bo *Matrix

	Wc *Matrix
	Uc *Matrix
	Bc *Matrix
}

func (lstm *OutputlessLSTM) ForgetGateTrick(v float32) {
	if lstm.Bf != nil {
		assembler.Sset(v, lstm.Bf.W)
	}
}

func MakeOutputlessLSTM(x_size, h_size int) *OutputlessLSTM {
	rnn := new(OutputlessLSTM)
	rnn.Wf = RandXavierMat(h_size, x_size)
	rnn.Uf = RandXavierMat(h_size, h_size)
	rnn.Bf = RandXavierMat(h_size, 1) // forget gate bias initialization trick will be applied here

	rnn.Wi = RandXavierMat(h_size, x_size)
	rnn.Ui = RandXavierMat(h_size, h_size)
	rnn.Bi = RandXavierMat(h_size, 1)

	rnn.Wo = RandXavierMat(h_size, x_size)
	rnn.Uo = RandXavierMat(h_size, h_size)
	rnn.Bo = RandXavierMat(h_size, 1)

	rnn.Wc = RandXavierMat(h_size, x_size)
	rnn.Uc = RandXavierMat(h_size, h_size)
	rnn.Bc = RandXavierMat(h_size, 1)

	return rnn
}

func (rnn *OutputlessLSTM) GetParameters(namespace string) map[string]*Matrix {
	return map[string]*Matrix{
		namespace + "_Wf": rnn.Wf,
		namespace + "_Uf": rnn.Uf,
		namespace + "_Bf": rnn.Bf,

		namespace + "_Wi": rnn.Wi,
		namespace + "_Ui": rnn.Ui,
		namespace + "_Bi": rnn.Bi,

		namespace + "_Wo": rnn.Wo,
		namespace + "_Uo": rnn.Uo,
		namespace + "_Bo": rnn.Bo,

		namespace + "_Wc": rnn.Wc,
		namespace + "_Uc": rnn.Uc,
		namespace + "_Bc": rnn.Bc,
	}
}

func (rnn *OutputlessLSTM) SetParameters(namespace string, parameters map[string]*Matrix) error {
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

func (rnn *OutputlessLSTM) Step(g *Graph, x, h_prev, c_prev *Matrix) (h, c *Matrix) {
	// make LSTM computation graph at one time-step
	f := g.Sigmoid(g.Add(g.Add(g.Mul(rnn.Wf, x), g.Mul(rnn.Uf, h_prev)), rnn.Bf))
	i := g.Sigmoid(g.Add(g.Add(g.Mul(rnn.Wi, x), g.Mul(rnn.Ui, h_prev)), rnn.Bi))
	o := g.Sigmoid(g.Add(g.Add(g.Mul(rnn.Wo, x), g.Mul(rnn.Uo, h_prev)), rnn.Bo))
	c = g.Tanh(g.Add(g.Add(g.Mul(rnn.Wc, x), g.Mul(rnn.Uc, h_prev)), rnn.Bc))
	c = g.Add(g.EMul(f, c_prev), g.EMul(i, c))
	h = g.EMul(o, g.Tanh(c))

	return
}
