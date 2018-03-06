package gortex

import (
	"fmt"
	//"log"
)

// Long Short Term Memory cell

type TemporalConvolution struct {
	Kernels     []*Matrix
	Biases      *Matrix
	Pads        []*Matrix // learnable pads !!
	KernelSize  int
	KernelShift int
}

func MakeTemporalConvolution(kernels, input_size, kernel_size, kernel_shift int) *TemporalConvolution {
	tcv := new(TemporalConvolution)
	tcv.KernelSize = kernel_size
	tcv.KernelShift = kernel_shift
	tcv.Kernels = make([]*Matrix, kernels)
	tcv.Biases = RandXavierMat(kernels, 1)
	tcv.Pads = make([]*Matrix, kernel_size-1)

	for i := range tcv.Kernels {
		tcv.Kernels[i] = RandXavierMat(input_size, kernel_size)
	}

	for i := range tcv.Pads {
		tcv.Pads[i] = RandXavierMat(input_size, 1)
	}
	return tcv
}

func (tcv *TemporalConvolution) GetParameters(namespace string) map[string]*Matrix {
	p := make(map[string]*Matrix)
	for i, kernel := range tcv.Kernels {
		p[fmt.Sprintf("%s_kernel_%d", namespace, i)] = kernel
	}
	p[fmt.Sprintf("%s_bias", namespace)] = tcv.Biases

	for i, pad := range tcv.Pads {
		p[fmt.Sprintf("%s_pad_%d", namespace, i)] = pad
	}
	return p
}

func (tcv *TemporalConvolution) SetParameters(namespace string, parameters map[string]*Matrix) error {
	for k, v := range tcv.GetParameters(namespace) {
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

func (tcv *TemporalConvolution) Step(g *Graph, x []*Matrix) (y []*Matrix) {
	if len(x) < tcv.KernelSize {
		//log.Printf("CONV Pad: l: %d", len(x))
		// pad sequence
		tail := tcv.KernelSize - len(x)
		//log.Printf("CONV tail: %d tails: %d", tail, len(tcv.Pads))
		for i := 0; i < tail; i++ {
			x = append(x, tcv.Pads[tail-i-1])
		}
		//log.Printf("CONV Paded: l: %d", len(x))
	}

	//log.Printf("CONV l: %d o: %d", len(x), (len(x)-tcv.KernelSize)/tcv.KernelShift+1)
	// compute convolutions
	y = make([]*Matrix, (len(x)-tcv.KernelSize)/tcv.KernelShift+1)
	//log.Printf("CONV: l: %d o: %d", len(x), len(y))
	for i := range y {
		packed_input := g.PackColumnVectors(x[i*tcv.KernelShift:i*tcv.KernelShift+tcv.KernelSize])
		conv := make([]*Matrix, len(tcv.Kernels))
		for k := range conv {
			conv[k] = g.Conv(packed_input, tcv.Kernels[k])
			//log.Printf("conv[%d]: %dx%d", k, conv[k].Rows, conv[k].Columns)

		}
		y[i] = g.Add(g.Concat(conv...), tcv.Biases)
	}

	return
}
