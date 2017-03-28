package gortex

type RNN struct {
	Wxh  *Matrix
	Whh  *Matrix
	Who  *Matrix
	Bias *Matrix
}

func MakeRNN(x_size, h_size, out_size int) *RNN {
	rnn := new(RNN)
	rnn.Wxh = RandXavierMat(x_size, h_size)
	rnn.Whh = RandXavierMat(h_size, h_size)
	rnn.Who = RandXavierMat(out_size, h_size)
	rnn.Bias = RandXavierMat(h_size, 1)
	return rnn
}

func (rnn *RNN) Model(namespace string) map[string]*Matrix {
	return map[string]*Matrix{namespace + "_Wxh": rnn.Wxh, namespace + "_Whh": rnn.Whh, namespace + "_Who": rnn.Who, namespace + "_Bias": rnn.Bias}
}

func (rnn *RNN) Step(g *Graph, x, h_prev *Matrix) (h, y *Matrix) {
	// make RNN computation graph at one time-step
	// h = tanh( Wxh * x+Whh * h + bias )
	// y = Who * h
	h = g.Tanh(g.Add(g.Add(g.Mul(rnn.Wxh, x), g.Mul(rnn.Whh, h_prev)), rnn.Bias))
	y = g.Mul(rnn.Who, h)
	return
}
