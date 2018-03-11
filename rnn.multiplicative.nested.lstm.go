package gortex

import (
	"fmt"

	"github.com/vseledkin/gortex/assembler"
)

// Long Short Term Memory cell

type MultiplicativeNestedLSTM struct {
	Wmx *Matrix
	Umh *Matrix

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

	Who *Matrix

	innerMemory *MultiplicativeLSTM
}

func (lstm *MultiplicativeNestedLSTM) ForgetGateTrick(v float32) {
	if lstm.Bf != nil {
		assembler.Sset(v, lstm.Bf.W)
	}
}

func MakeMultiplicativeNestedLSTM(x_size, h_size, out_size int) *MultiplicativeNestedLSTM {
	rnn := new(MultiplicativeNestedLSTM)
	rnn.Wmx = RandXavierMat(h_size, x_size)
	rnn.Umh = RandXavierMat(h_size, h_size)

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

	rnn.Who = RandXavierMat(out_size, h_size)

	rnn.innerMemory = MakeMultiplicativeLSTM(h_size, h_size, h_size)
	return rnn
}

func (rnn *MultiplicativeNestedLSTM) GetParameters(namespace string) map[string]*Matrix {
	params := map[string]*Matrix{
		namespace + "_Wmx": rnn.Wmx,
		namespace + "_Umh": rnn.Umh,

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

		namespace + "_Who": rnn.Who,
	}

	for k, v := range rnn.innerMemory.GetParameters(namespace + "_inner") {
		params[k] = v
	}
	return params
}

func (rnn *MultiplicativeNestedLSTM) SetParameters(namespace string, parameters map[string]*Matrix) error {
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

func (rnn *MultiplicativeNestedLSTM) Step(g *Graph, x, h_prev, c_prev, c_previn *Matrix) (h, c, cin, y *Matrix) {
	// make MultiplicativeLSTM computation graph at one time-step

	m := g.EMul(g.Mul(rnn.Wmx, x), g.Mul(rnn.Umh, h_prev))
	f := g.Sigmoid(g.Add(g.Add(g.Mul(rnn.Wf, x), g.Mul(rnn.Uf, m)), rnn.Bf))
	i := g.Sigmoid(g.Add(g.Add(g.Mul(rnn.Wi, x), g.Mul(rnn.Ui, m)), rnn.Bi))
	o := g.Sigmoid(g.Add(g.Add(g.Mul(rnn.Wo, x), g.Mul(rnn.Uo, m)), rnn.Bo))
	c = g.Tanh(g.Add(g.Add(g.Mul(rnn.Wc, x), g.Mul(rnn.Uc, m)), rnn.Bc))

	//c = g.Add(g.EMul(f, c_prev), g.EMul(i, c))
	c_new, cin, _ := rnn.innerMemory.Step(g, g.EMul(i, c), g.EMul(f, c_prev), c_previn)

	h = g.EMul(o, g.Tanh(c_new))

	y = g.Mul(rnn.Who, h)
	return
}
