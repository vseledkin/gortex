package gortex

import (
	"fmt"
	"log"
	"github.com/vseledkin/gortex/assembler"
)

type DilatedTemporalConvolution struct {
	Kernels [][]*Matrix // any number of layers each having any number of kernels of size input X 3
	Biases  []*Matrix
	Wo      *Matrix
	inputs  [][]*Matrix `json:"-"`
	zeros   []*Matrix
}

func MakeDilatedTemporalConvolution(inputSize, outputSize int, kernelSizes []int) *DilatedTemporalConvolution {
	dtc := new(DilatedTemporalConvolution)
	dtc.Wo = RandXavierMat(outputSize, kernelSizes[len(kernelSizes)-1])

	dtc.Kernels = make([][]*Matrix, len(kernelSizes))
	dtc.Biases = make([]*Matrix, len(kernelSizes))
	inputSizes := []int{inputSize}
	inputSizes = append(inputSizes, kernelSizes...)
	dtc.zeros = make([]*Matrix, len(inputSizes))
	for i, n := range kernelSizes {
		dtc.Kernels[i] = make([]*Matrix, n)
		for j := range dtc.Kernels[i] {
			dtc.Kernels[i][j] = RandXavierMat(inputSizes[i], 3)
		}
		dtc.Biases[i] = RandXavierMat(kernelSizes[i], 1)
	}
	for i, n := range inputSizes {
		dtc.zeros[i] = Mat(n, 1)
	}
	return dtc
}

func (dtc *DilatedTemporalConvolution) ReceptiveField(g *Graph, i, layer int, x []*Matrix) (field []int, ret *Matrix) {
	if i < len(x) && i >= 0 {

		var input []*Matrix
		pow := 1 << uint(layer)
		if i > pow-1 {
			input = append(input, x[i-pow])
			field = append(field, i-pow)
		} else {
			input = append(input, dtc.zeros[layer])
		}
		input = append(input, x[i])
		field = append(field, i)
		if i < len(x)-pow {
			input = append(input, x[i+pow])
			field = append(field, i+pow)
		} else {
			input = append(input, dtc.zeros[layer])
		}
		//log.Printf("layer %d step %d input %v", layer, i, x)
		ret = g.PackColumnVectors(input)
		return
	}
	panic("wrong attempt to receive valid perceptive field")
}

func (dtc *DilatedTemporalConvolution) SetParameters(namespace string, parameters map[string]*Matrix) error {
	for k, v := range dtc.GetParameters(namespace) {
		//fmt.Printf("Look for %s parameters\n", k)
		if m, ok := parameters[k]; ok {
			//fmt.Printf("Got %s parameters\n", k)
			copy(v.W, m.W)
		} else {
			return fmt.Errorf("Model geometry is not compatible, parameter %s is unknown", k)
		}
	}
	return nil
}

func (dtc *DilatedTemporalConvolution) GetParameters(namespace string) map[string]*Matrix {
	p := make(map[string]*Matrix)
	for layer, kernels := range dtc.Kernels {
		for i, kernel := range kernels {
			p[fmt.Sprintf("%s_layer%d_kernel%d", namespace, layer, i)] = kernel
		}
	}

	for layer, bias := range dtc.Biases {
		p[fmt.Sprintf("%s_layer%d_bias", namespace, layer)] = bias
	}

	p[fmt.Sprintf("%s_Wo", namespace)] = dtc.Wo
	return p
}

func (dtc *DilatedTemporalConvolution) LookFullStep(g *Graph, layer, t int) {
	// check if we have already computed nessesary outputs
	if dtc.inputs[layer+1][t] == nil { // if not
		// run recursion to calculate all nessesry prerequisites from lower layers
		if layer > 0 {
			dtc.LookFullStep(g, layer-1, t) // reqcursion!!!!!
		}

		// get receptive field for conv neuron
		field, input := dtc.ReceptiveField(g, t, layer, dtc.inputs[layer])

		log.Printf("layer %d step %d input %+v", layer, t, field)
		layerKernels := dtc.Kernels[layer]
		output := make([]*Matrix, len(layerKernels))
		for i := range layerKernels {
			output[i] = g.Conv(input, layerKernels[i])
		}
		dtc.inputs[layer+1][t] = g.Relu(g.Add(g.Concat(output...), dtc.Biases[layer]))

	}
	return
}

func (dtc *DilatedTemporalConvolution) lookPastStep(g *Graph, layer, t int) {
	// check if we have already computed nessesary outputs
	if dtc.inputs[layer+1][t] == nil { // if not
		// run recursion to calculate all nessesry prerequisites from lower layers
		if layer > 0 {
			dtc.lookPastStep(g, layer-1, t) // reqcursion!!!!!
		}

		// get receptive field for conv neuron
		_, input := dtc.ReceptiveField(g, t, layer, dtc.inputs[layer][:t+1])

		//log.Printf("layer %d step %d input %+v", layer, t, field)
		layerKernels := dtc.Kernels[layer]
		output := make([]*Matrix, len(layerKernels))
		for i := range layerKernels {
			output[i] = g.Conv(input, layerKernels[i])
		}
		dtc.inputs[layer+1][t] = g.Relu(g.Add(g.Concat(output...), dtc.Biases[layer]))

	}
	return
}

func (dtc *DilatedTemporalConvolution) SetInput(input []*Matrix) {
	dtc.inputs = nil
	dtc.inputs = make([][]*Matrix, len(dtc.Kernels)+1)
	dtc.inputs[0] = input
	for i := range dtc.Kernels {
		dtc.inputs[i+1] = make([]*Matrix, len(input))
	}
	for i := range dtc.zeros {
		assembler.Sclean(dtc.zeros[i].DW)
	}

}

func (dtc *DilatedTemporalConvolution) AddInput(inp *Matrix) {
	dtc.inputs[0] = append(dtc.inputs[0], inp)
	for i := range dtc.Kernels {
		dtc.inputs[i+1] = append(dtc.inputs[i+1], nil)
	}
}

func (dtc *DilatedTemporalConvolution) LookPastStep(g *Graph, t int) (y *Matrix) {
	L := len(dtc.Kernels) - 1
	if dtc.inputs[L][t] == nil { // if inputs are not ready
		dtc.lookPastStep(g, L, t)
	}
	return g.Mul(dtc.Wo, dtc.inputs[L+1][t])
}

func (dtc *DilatedTemporalConvolution) FullStep(g *Graph, t int) (y *Matrix) {
	L := len(dtc.Kernels) - 1
	if dtc.inputs[L][t] == nil { // if inputs are not ready
		dtc.LookFullStep(g, L, t)
	}
	return g.Mul(dtc.Wo, dtc.inputs[L+1][t])
}
