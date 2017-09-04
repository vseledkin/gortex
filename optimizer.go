package gortex

import (
	"math"

	"log"

	"github.com/vseledkin/gortex/assembler"
)

type OpMethod int

const (
	SGD OpMethod = iota
	ADAM
	RMSPROP
	ADAGRAD
	ADADELTA
	WINDOWGRAD
	NETSTEROV
	POWERBALL
)

const (
	DefaultMomentum = 0.9
)

type OpOp struct {
	LearningRate float32
	L1Decay      float32
	L2Decay      float32
	RmsDecayRate float32
	Momentum     float32
	Ro           float32 // used by adadelta
	Eps          float32 // used by adadelta and adam
	Beta1        float32 // used by adam
	Beta2        float32 // used by adam
	Clip         float32 // used by all
	Method       OpMethod
	Powerball    float32
}

type OpRet struct {
	L1Loss     float32
	L2Loss     float32
	NumClipped int
}

type Optimizer struct {
	OpOp
	PreviousGradient map[string][]float32 // previous iteration gradients (used for momentum calculations)
	PreviousWeight   map[string][]float32 // previous iteration weight (used by adadelta)
	Iteration        float32
}

func NewOptimizer(ops OpOp) *Optimizer {
	op := new(Optimizer)
	op.PreviousGradient = make(map[string][]float32)
	op.PreviousWeight = make(map[string][]float32)
	op.OpOp = ops
	if op.LearningRate == 0 {
		op.LearningRate = 0.01
	}
	if op.Ro == 0 {
		op.Ro = 0.95
	}
	if op.Eps == 0 {
		op.Eps = 1e-8
	}
	if op.Beta1 == 0 {
		op.Beta1 = 0.9
	}
	if op.Beta2 == 0 {
		op.Beta2 = 0.999
	}
	if op.RmsDecayRate == 0 {
		op.RmsDecayRate = 0.999
	}
	if op.Momentum == 0 && op.Method == NETSTEROV {
		panic("Nesterov assumes momentum is positive!")
	}
	if op.Powerball == 0 {
		op.Powerball = 0.1
	}
	return op
}

func (o *Optimizer) getPreviousGradient(name string, m *Matrix) []float32 {
	previousGradient, ok := o.PreviousGradient[name]
	if !ok {
		previousGradient = make([]float32, m.Numel())
		o.PreviousGradient[name] = previousGradient
	}
	return previousGradient
}

func (o *Optimizer) setPreviousGradient(name string, g []float32) {
	o.PreviousGradient[name] = g
}

func (o *Optimizer) getPreviousWeight(name string, m *Matrix) []float32 {
	previousWeight, ok := o.PreviousWeight[name]
	if !ok {
		previousWeight = make([]float32, m.Numel())
		o.PreviousWeight[name] = previousWeight
	}
	return previousWeight
}

func (o *Optimizer) setPreviousWeight(name string, w []float32) {
	o.PreviousWeight[name] = w
}

func (o *Optimizer) clip(w []float32) int {
	var num_clipped int
	// gradient clip
	for i := range w {
		if w[i] > o.Clip {
			w[i] = o.Clip
			num_clipped++
			continue
		}
		if w[i] < -o.Clip {
			w[i] = -o.Clip
			num_clipped++
		}
	}
	return num_clipped
}
func sign(w float32) float32 {
	if w >= 0 {
		return 1
	}
	return -1
}
func (o *Optimizer) Step(model map[string]*Matrix) OpRet {
	ret := OpRet{}
	// make method specific weight optimization
	o.Iteration++
	for name, m := range model {
		if o.Clip > 0 {
			ret.NumClipped += o.clip(m.DW)
		}
		if o.L1Decay > 0 {
			for i := range m.W {
				m.DW[i] += o.L1Decay * sign(m.W[i])
			}
		}
		if o.L2Decay > 0 {
			assembler.Saxpy(o.L2Decay, m.W, m.DW)
		}

		//assembler.Sscale(1/(assembler.L2(m.DW)+o.Eps), m.DW)
		if assembler.L2(m.DW) == 0 {
			log.Printf("WARNING: %s W:%f DW:%f\n", name, assembler.L2(m.W), assembler.L2(m.DW))
		}
		switch o.Method {
		case RMSPROP:
			xsumi := o.getPreviousWeight(name, m)
			assembler.Saxplusbyvsetz(o.RmsDecayRate, xsumi, 1-o.RmsDecayRate, m.DW, m.DW, xsumi)
			if o.Iteration > 10 {
				assembler.Saxdivsqrteyplusz(-o.LearningRate, m.DW, o.Eps, xsumi, m.W)
			}
		case ADAM:
			gsumi := o.getPreviousGradient(name, m)
			xsumi := o.getPreviousWeight(name, m)
			assembler.Saxplusbysetz(o.Beta1, gsumi, 1-o.Beta1, m.DW, gsumi)        // update biased first moment estimate
			assembler.Saxplusbyvsetz(o.Beta2, xsumi, 1-o.Beta2, m.DW, m.DW, xsumi) // update biased second moment estimate
			beta1iteration := (1 - Pow(o.Beta1, o.Iteration))
			beta2iteration := (1 - Pow(o.Beta2, o.Iteration))
			biasCorr1 := make([]float32, m.Numel())
			biasCorr2 := make([]float32, m.Numel())
			assembler.Saxpy(beta1iteration, gsumi, biasCorr1) // correct bias first moment estimate
			assembler.Saxpy(beta2iteration, xsumi, biasCorr2) // correct bias second moment estimate
			if o.Iteration > 10 {
				assembler.Saxdivsqrteyplusz(-o.LearningRate, biasCorr1, o.Eps, biasCorr2, m.W)
			}
		case ADAGRAD:
			gsumi := o.getPreviousGradient(name, m)
			assembler.Sxmuleyplusz(m.DW, m.DW, gsumi)
			assembler.Saxdivsqrteyplusz(-o.LearningRate, m.DW, o.Eps, gsumi, m.W)
		case WINDOWGRAD:
			// this is adagrad but with a moving window weighted average
			// so the gradient is not accumulated over the entire history of the run.
			gsumi := o.getPreviousGradient(name, m)
			assembler.Saxplusbyvsetz(o.Ro, gsumi, 1-o.Ro, m.DW, m.DW, gsumi)
			assembler.Saxdivsqrteyplusz(-o.LearningRate, m.DW, o.Eps, gsumi, m.W)
		case ADADELTA:
			gsumi := o.getPreviousGradient(name, m)
			xsumi := o.getPreviousWeight(name, m)
			assembler.Saxplusbyvsetz(o.Ro, gsumi, 1-o.Ro, m.DW, m.DW, gsumi)
			for i := range m.W {
				dx := -assembler.Sqrt((xsumi[i]+o.Eps)/(gsumi[i]+o.Eps)) * m.DW[i]
				xsumi[i] = o.Ro*xsumi[i] + (1-o.Ro)*dx*dx
				m.W[i] += dx
			}
		case NETSTEROV:
			dx := o.getPreviousGradient(name, m)
			gsumi := make([]float32, m.Numel())
			assembler.Saxplusbysetz(o.Momentum, dx, o.LearningRate, m.DW, gsumi)
			assembler.Saxplusbyplusz(o.Momentum, dx, -(1 + o.Momentum), gsumi, m.W)
			o.setPreviousGradient(name, gsumi)
		case POWERBALL:
			if o.Momentum > 0 {
				dx := o.getPreviousGradient(name, m)
				for i := range m.DW {
					m.DW[i] = sign(m.DW[i]) * float32(math.Pow(float64(Abs(m.DW[i])), float64(o.Powerball)))
				}
				assembler.Saxplusbysetz(o.Momentum, dx, -o.LearningRate, m.DW, dx)
				// apply corrected gradient
				assembler.Sxpy(dx, m.W)
			} else {
				for i := range m.DW {
					m.DW[i] = sign(m.DW[i]) * float32(math.Pow(float64(Abs(m.DW[i])), float64(o.Powerball)))
				}
				assembler.Saxpy(-o.LearningRate, m.DW, m.W)
			}
		case SGD:
			if o.Momentum > 0 {
				dx := o.getPreviousGradient(name, m)
				assembler.Saxplusbysetz(o.Momentum, dx, -o.LearningRate, m.DW, dx)
				// apply corrected gradient
				assembler.Sxpy(dx, m.W)
			} else {
				// vanilla sgd no momentum
				assembler.Saxpy(-o.LearningRate, m.DW, m.W)
			}
		default:
			panic("Not implemented")
		}
	}

	// reset gradients
	for _, m := range model {
		assembler.Sclean(m.DW)
	}

	return ret
}
