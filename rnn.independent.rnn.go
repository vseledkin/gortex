package gortex

import (
	"fmt"
)

/*
	The implementation is based on: https://arxiv.org/abs/1803.04831
	Shuai Li, Wanqing Li, Chris Cook, Ce Zhu, Yanbo Gao
	"Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN"
*/

type IndRNN struct {
	W []*Matrix
	U [][]*Matrix

	B []*Matrix

	Who *Matrix
}

func MakeIndRNN(layers, x_size, length, h_size, out_size int) *IndRNN {
	rnn := new(IndRNN)
	rnn.U = make([][]*Matrix, layers) // layers/length
	rnn.W = make([]*Matrix, layers)   // layers/length
	rnn.B = make([]*Matrix, layers)   // layers/length
	for l := range rnn.U {
		if l == 0 {
			rnn.W[l] = LSUVMat(h_size, x_size)
		} else {
			rnn.W[l] = LSUVMat(h_size, h_size)
		}
		rnn.U[l] = make([]*Matrix, length)
		for i := range rnn.U[l] {
			rnn.U[l][i] = RandXavierMat(h_size, 1)
		}
		rnn.B[l] = Mat(h_size, 1)
	}

	rnn.Who = LSUVMat(out_size, h_size)
	return rnn
}

func (rnn *IndRNN) GetParameters(namespace string) map[string]*Matrix {
	p := make(map[string]*Matrix)

	p[namespace+"_Who"] = rnn.Who
	for l := range rnn.W {
		p[fmt.Sprintf("%s_W_%d", namespace, l)] = rnn.W[l]
		p[fmt.Sprintf("%s_B_%d", namespace, l)] = rnn.B[l]
		for i := range rnn.U[l] {
			p[fmt.Sprintf("%s_U_%d_%d", namespace, l, i)] = rnn.U[l][i]
		}
	}

	return p
}

func (rnn *IndRNN) SetParameters(namespace string, parameters map[string]*Matrix) error {
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

func (rnn *IndRNN) Step(g *Graph, i int, x *Matrix, h_prev []*Matrix) (h []*Matrix, y *Matrix) {
	// make IndRNN computation graph at one time-step
	h = make([]*Matrix, len(rnn.W))
	for l := range rnn.W {
		var input *Matrix
		if l == 0 {
			input = x
		} else {
			input = h[l-1]
		}
		//fmt.Printf("l: %d %d lw%d lu%d lul%d lb%d hprev%d \n", l, i, len(rnn.W), len(rnn.U), len(rnn.U), len(rnn.B), len(h_prev))
		h[l] = g.BipolarElu(g.Add(g.Add(g.Mul(rnn.W[l], input), g.EMul(rnn.U[l][i], h_prev[l])), rnn.B[l]))
	}
	y = g.Mul(rnn.Who, h[len(h)-1])
	return
}
