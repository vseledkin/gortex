package gortex

import "github.com/vseledkin/gortex/assembler"

type MovingAverage struct {
	Window      int
	values      []float32
	valPos      int
	slotsFilled bool
}

func (ma *MovingAverage) Avg() float32 {
	var c = ma.Window - 1

	// Are all slots filled? If not, ignore unused
	if !ma.slotsFilled {
		c = ma.valPos - 1
		if c < 0 {
			// Empty register
			return 0
		}
	}

	// Sum values

	data := ma.values[:c]
	return assembler.Sum(data) / float32(len(data))
}

func (ma *MovingAverage) Add(val float32) {
	// Put into values array
	ma.values[ma.valPos] = val

	// Increment value position
	ma.valPos = (ma.valPos + 1) % ma.Window

	// Did we just go back to 0, effectively meaning we filled all registers?
	if !ma.slotsFilled && ma.valPos == 0 {
		ma.slotsFilled = true
	}
}

func NewMovingAverage(window int) *MovingAverage {
	return &MovingAverage{
		Window:      window,
		values:      make([]float32, window),
		valPos:      0,
		slotsFilled: false,
	}
}
