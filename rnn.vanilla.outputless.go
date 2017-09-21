package gortex

import "fmt"

type OutputlessRNN struct {
	Wxh  *Matrix
	Whh  *Matrix
	Bias *Matrix
}

func MakeOutputlessRNN(x_size, h_size int) *OutputlessRNN {
	net := new(OutputlessRNN)
	net.Wxh = RandXavierMat(h_size, x_size)
	net.Whh = RandXavierMat(h_size, h_size)
	net.Bias = RandXavierMat(h_size, 1)
	return net
}

func (rnn *OutputlessRNN) GetParameters(namespace string) map[string]*Matrix {
	return map[string]*Matrix{
		namespace + "_Wxh": rnn.Wxh,
		namespace + "_Whh": rnn.Whh, namespace + "_Bias": rnn.Bias}
}

func (rnn *OutputlessRNN) SetParameters(namespace string, parameters map[string]*Matrix) error {
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

func (rnn *OutputlessRNN) Step(g *Graph, x, h_prev *Matrix) (h *Matrix) {
	// make RNN computation graph at one time-step
	// h = tanh( Wxh * x+Whh * h + bias )

	h = g.Tanh(g.Add(g.Add(g.Mul(rnn.Wxh, x), g.Mul(rnn.Whh, h_prev)), rnn.Bias))

	return
}
