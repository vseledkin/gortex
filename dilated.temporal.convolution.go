package gortex


type DilatedTemporalConvolution struct {
	NumberOfKernels []int
	KernelSizes     []int
}

/*
func (dtc *DilatedTemporalConvolution) PadStart(x []int) {
	if len(x) < dtc.KernelSizes[0] {
		// pad sequence
		tail := dtc.KernelSizes[0] - len(x)
		for i := 0; i < tail; i++ {
			x = append(x, tcv.Pads[tail-i-1])
		}
		//log.Printf("CONV Paded: l: %d", len(x))
	}
}
*/
func (dtc *DilatedTemporalConvolution) ReceptiveField(i, layer int, x []int) []int {
	delta := (dtc.KernelSizes[layer] / 2)
	switch layer {
	case 0:
		start := i - delta
		end := i + delta
		if start < 0 {
			start = 0
		}
		if end > len(x)-1 {
			end = len(x) - 1
		}
		return x[start:end+1]
	case 1: // dilation one
		d := 1
		past := []int{}
		future := []int{}
		for s := 1; s < delta; s++ {
			pos := i-d-s
			if pos > 0 {
				past = append(past, pos)
			}
			pos = i+d+s
			if pos > len(x)-1 {
				future = append(future, pos)
			}
		}
		// add past
		for i, j := 0, len(past)-1; i < j; i, j = i+1, j-1 {
			past[i], past[j] = past[j], past[i]
		}

		past = append(past,i)

		return append(past,future...)
	default:
		panic("not implemented")
	}

}
