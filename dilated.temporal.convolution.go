package gortex

import (
	"fmt"
	"github.com/vseledkin/gortex/assembler"
)

type DilatedTemporalConvolution struct {
	Kernels     [][]*Matrix // any number of layers each having any number of kernels of size input X 3
	Biases      []*Matrix
	ConvBiases  []*Matrix
	Gates       []*Matrix
	KernelSizes []int
	UseGates    bool
	inputs      [][]*Matrix `json:"-"`
	Zeros       []*Matrix
}

func MakeDilatedTemporalConvolution(inputSize int, kernelSizes []int, useGates bool) *DilatedTemporalConvolution {
	dtc := new(DilatedTemporalConvolution)
	dtc.UseGates = useGates
	dtc.KernelSizes = kernelSizes

	dtc.Kernels = make([][]*Matrix, len(kernelSizes))
	if dtc.UseGates {
		dtc.Biases = make([]*Matrix, len(kernelSizes))
		dtc.Gates = make([]*Matrix, len(kernelSizes))
	}
	dtc.ConvBiases = make([]*Matrix, len(kernelSizes))
	inputSizes := []int{inputSize}
	inputSizes = append(inputSizes, kernelSizes...)
	dtc.Zeros = make([]*Matrix, len(inputSizes))
	for i, n := range kernelSizes {
		dtc.Kernels[i] = make([]*Matrix, n)
		for j := range dtc.Kernels[i] {
			dtc.Kernels[i][j] = LSUVMat(inputSizes[i], 3)
		}
		dtc.ConvBiases[i] = Mat(kernelSizes[i], 1)
		if dtc.UseGates {
			dtc.Gates[i] = LSUVMat(kernelSizes[i], kernelSizes[i])
			dtc.Biases[i] = Mat(kernelSizes[i], 1)
		}
	}

	for i, n := range inputSizes {
		dtc.Zeros[i] = Mat(n, 1)
	}
	return dtc
}

func (dtc *DilatedTemporalConvolution) OutputSize() int {
	return dtc.KernelSizes[len(dtc.KernelSizes)-1]
}

func (dtc *DilatedTemporalConvolution) Pack(g *Graph, i, layer int, field []int) (ret *Matrix) {
	input := make([]*Matrix, len(field))
	for i, pos := range field {
		if pos > 0 {
			input[i] = dtc.inputs[layer][pos]
		} else {
			input[i] = dtc.Zeros[layer]
		}
	}
	ret = g.PackColumnVectors(input)
	return
}

func (dtc *DilatedTemporalConvolution) ReceptiveField(i, layer int) (field []int) {
	L := len(dtc.inputs[layer])
	if i < L && i >= 0 {
		pow := 1 << uint(layer)
		if i > pow-1 {
			field = append(field, i-pow)
		} else {
			field = append(field, -1)
		}
		field = append(field, i)
		if i < L-pow {
			field = append(field, i+pow)
		} else {
			field = append(field, -1)
		}
		//fmt.Printf("ReceptiveField at Layer: %d at Pos: %d on Len: %d Field: %+v\n", layer, i, L, field)
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

	for layer, bias := range dtc.ConvBiases {
		p[fmt.Sprintf("%s_layer%d_conv_bias", namespace, layer)] = bias
	}

	if dtc.UseGates {
		for layer, bias := range dtc.Biases {
			p[fmt.Sprintf("%s_layer%d_bias", namespace, layer)] = bias
		}
		for layer, wo := range dtc.Gates {
			p[fmt.Sprintf("%s_layer%d_wo", namespace, layer)] = wo
		}
	}

	//p[fmt.Sprintf("%s_Wo", namespace)] = dtc.Gates
	return p
}

func (dtc *DilatedTemporalConvolution) Check() {
	// check that second layer looks at the start of sequence
	L := len(dtc.inputs[1])
	for pos := range dtc.inputs[1] {
		if dtc.inputs[1][pos] != nil {
			field := dtc.ReceptiveField(pos, 0)
			hasZero := false
			for _, f := range field {
				if f == 0 {
					hasZero = true
					break
				}
			}
			if !hasZero {
				panic(fmt.Errorf("insufficient attention range %+v for sequence of %d elements pos %d at start", field, L, pos))
			}
			return
		}
	}
	// check that second layer looks at the end of sequence
	for pos := L - 1; pos >= 0; pos-- {
		if dtc.inputs[1][pos] != nil {
			field := dtc.ReceptiveField(pos, 0)
			hasZero := false
			for _, f := range field {
				if f == L-1 {
					hasZero = true
					break
				}
			}
			if !hasZero {
				panic(fmt.Errorf("insufficient attention range %+v for sequence of %d elements pos %d at end", field, L, pos))
			}
			return
		}
	}

}

func (dtc *DilatedTemporalConvolution) LayerAttentionWidth(layer int) int {
	return 2<<uint(layer+1) - 1
}

func (dtc *DilatedTemporalConvolution) MaxAttentionWidth() int {
	return dtc.LayerAttentionWidth(len(dtc.Kernels) - 1)
}

func (dtc *DilatedTemporalConvolution) LookFullStep(g *Graph, layer, t int) {
	//fmt.Printf("LookFullStep Layer: %d Pos: %d\n", layer, t)
	// check if we have already computed nessesary outputs
	if dtc.inputs[layer+1][t] == nil { // if not
		field := dtc.ReceptiveField(t, layer)
		if layer > 0 {
			// run recursion for every position which value is not calculated yet
			for _, pos := range field {
				if pos >= 0 && dtc.inputs[layer][pos] == nil {
					dtc.LookFullStep(g, layer-1, pos) // reqcursion!!!!!
				}
			}
		}
		// now we have all inputs ready
		// pack inputs from lover layers
		input := dtc.Pack(g, t, layer, field)

		//log.Printf("layer %d step %d input %+v", layer, t, field)
		layerKernels := dtc.Kernels[layer]
		output := make([]*Matrix, len(layerKernels))
		for i := range layerKernels {
			output[i] = g.Conv(input, layerKernels[i])
		}
		conv_output := g.Tanh(g.Add(g.Concat(output...), dtc.ConvBiases[layer]))
		if dtc.UseGates {
			dtc.inputs[layer+1][t] = g.Tanh(g.Add(g.Mul(dtc.Gates[layer], conv_output), dtc.Biases[layer]))
		} else {
			dtc.inputs[layer+1][t] = conv_output
		}
		//fmt.Printf("Put LookFullStep Layer: %d Pos: %d for layer %d\n", layer, t, layer+1)
	} //else {
		//fmt.Printf("Ready LookFullStep Layer: %d Pos: %d\n", layer, t)
	//}
	return
}

func (dtc *DilatedTemporalConvolution) lookPastStep(g *Graph, layer, t int) {
	// check if we have already computed nessesary outputs
	if dtc.inputs[layer+1][t] == nil { // if not
		// run recursion to calculate all nessesry prerequisites from lower layers
		field := dtc.ReceptiveField(t, layer)
		if layer > 0 {
			// run recursion for every position which value is not calculated yet
			for _, pos := range field {
				if dtc.inputs[layer][pos] == nil {
					dtc.LookFullStep(g, layer-1, pos) // reqcursion!!!!!
				}
			}
		}
		// now we have all inputs ready
		// pack inputs from lover layers
		input := dtc.Pack(g, t, layer, field)

		// get receptive field for conv neuron
		//_, input := dtc.ReceptiveField(g, t, layer, dtc.inputs[layer][:t+1])

		//log.Printf("layer %d step %d input %+v", layer, t, field)
		layerKernels := dtc.Kernels[layer]
		output := make([]*Matrix, len(layerKernels))
		for i := range layerKernels {
			output[i] = g.Conv(input, layerKernels[i])
		}
		dtc.inputs[layer+1][t] = g.Tanh(g.Add(g.Concat(output...), dtc.Biases[layer]))

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
	for i := range dtc.Zeros {
		assembler.Sclean(dtc.Zeros[i].DW)
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
	return dtc.inputs[L+1][t]
}

func (dtc *DilatedTemporalConvolution) FullStep(g *Graph, t int) (y *Matrix) {
	//fmt.Printf("FullStep Pos: %d\n", t)
	L := len(dtc.Kernels) - 1
	if dtc.inputs[L+1][t] == nil { // if inputs are not ready
		dtc.LookFullStep(g, L, t)
	} //else {
	//	fmt.Printf("FullStep Ready Pos: %d\n", t)
	//}
	return dtc.inputs[L+1][t]
}
