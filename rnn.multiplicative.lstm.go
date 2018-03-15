package gortex

import (
	"fmt"

	"github.com/vseledkin/gortex/assembler"
)

// Long Short Term Memory cell

type MultiplicativeLSTM struct {
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
}

func (lstm *MultiplicativeLSTM) ForgetGateTrick(v float32) {
	if lstm.Bf != nil {
		assembler.Sset(v, lstm.Bf.W)
	}
}

func MakeMultiplicativeLSTM(x_size, h_size, out_size int) *MultiplicativeLSTM {
	rnn := new(MultiplicativeLSTM)
	rnn.Wmx = LSUVMat(h_size, x_size)
	rnn.Umh = LSUVMat(h_size, h_size)

	rnn.Wf = LSUVMat(h_size, x_size)
	rnn.Uf = LSUVMat(h_size, h_size)
	rnn.Bf = Mat(h_size, 1) // forget gate bias initialization trick will be applied here

	rnn.Wi = LSUVMat(h_size, x_size)
	rnn.Ui = LSUVMat(h_size, h_size)
	rnn.Bi = Mat(h_size, 1)

	rnn.Wo = LSUVMat(h_size, x_size)
	rnn.Uo = LSUVMat(h_size, h_size)
	rnn.Bo = Mat(h_size, 1)

	rnn.Wc = LSUVMat(h_size, x_size)
	rnn.Uc = LSUVMat(h_size, h_size)
	rnn.Bc = Mat(h_size, 1)

	rnn.Who = LSUVMat(out_size, h_size)
	return rnn
}

func (rnn *MultiplicativeLSTM) GetParameters(namespace string) map[string]*Matrix {
	return map[string]*Matrix{
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
}

func (rnn *MultiplicativeLSTM) SetParameters(namespace string, parameters map[string]*Matrix) error {
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

func (rnn *MultiplicativeLSTM) Step(g *Graph, x, h_prev, c_prev *Matrix) (h, c, y *Matrix) {
	// make MultiplicativeLSTM computation graph at one time-step

	m := g.EMul(g.Mul(rnn.Wmx, x), g.Mul(rnn.Umh, h_prev))
	f := g.Sigmoid(g.Add(g.Add(g.Mul(rnn.Wf, x), g.Mul(rnn.Uf, m)), rnn.Bf))
	i := g.Sigmoid(g.Add(g.Add(g.Mul(rnn.Wi, x), g.Mul(rnn.Ui, m)), rnn.Bi))
	o := g.Sigmoid(g.Add(g.Add(g.Mul(rnn.Wo, x), g.Mul(rnn.Uo, m)), rnn.Bo))
	c = g.Tanh(g.Add(g.Add(g.Mul(rnn.Wc, x), g.Mul(rnn.Uc, m)), rnn.Bc))
	c = g.Add(g.EMul(f, c_prev), g.EMul(i, c))
	h = g.EMul(o, g.Tanh(c))

	y = g.Mul(rnn.Who, h)
	return
}
