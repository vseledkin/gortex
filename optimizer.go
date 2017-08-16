package gortex

import (
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
	Method       OpMethod
}

type OpRet struct {
	L1Loss float32
	L2Loss float32
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

func (o *Optimizer) Step(model map[string]*Matrix) OpRet {
	ret := OpRet{}
	// make method specific weight optimization
	o.Iteration++
	for name, m := range model {
		switch o.Method {
		case RMSPROP:
			xsumi := o.getPreviousWeight(name, m)
			for i := range m.W {
				xsumi[i] = xsumi[i]*o.RmsDecayRate + (1.0-o.RmsDecayRate)*m.DW[i]*m.DW[i]
				m.W[i] += -o.LearningRate * m.DW[i] / Sqrt(xsumi[i]+o.Eps)
			}
		case ADAM:
			gsumi := o.getPreviousGradient(name, m)
			xsumi := o.getPreviousWeight(name, m)
			for i := range m.W {
				gsumi[i] = gsumi[i]*o.Beta1 + (1-o.Beta1)*m.DW[i]         // update biased first moment estimate
				xsumi[i] = xsumi[i]*o.Beta2 + (1-o.Beta2)*m.DW[i]*m.DW[i] // update biased second moment estimate
				biasCorr1 := gsumi[i] * (1 - Pow(o.Beta1, o.Iteration))   // correct bias first moment estimate
				biasCorr2 := xsumi[i] * (1 - Pow(o.Beta2, o.Iteration))   // correct bias second moment estimate
				m.W[i] += -o.LearningRate * biasCorr1 / (Sqrt(biasCorr2) + o.Eps)
			}
		case ADAGRAD:
			gsumi := o.getPreviousGradient(name, m)
			assembler.Sxmuleyplusz(m.DW, m.DW, gsumi)
			for i := range m.W {
				m.W[i] += -o.LearningRate * m.DW[i] / Sqrt(gsumi[i]+o.Eps)
			}
		case WINDOWGRAD:
			// this is adagrad but with a moving window weighted average
			// so the gradient is not accumulated over the entire history of the run.
			gsumi := o.getPreviousGradient(name, m)
			for i := range m.W {
				gsumi[i] = o.Ro*gsumi[i] + (1-o.Ro)*m.DW[i]*m.DW[i]
				m.W[i] += -o.LearningRate * m.DW[i] / Sqrt(gsumi[i]+o.Eps)
			}
		case ADADELTA:
			gsumi := o.getPreviousGradient(name, m)
			xsumi := o.getPreviousWeight(name, m)
			for i := range m.W {
				gsumi[i] = o.Ro*gsumi[i] + (1-o.Ro)*m.DW[i]*m.DW[i]
				dx := -Sqrt((xsumi[i]+o.Eps)/(gsumi[i]+o.Eps)) * m.DW[i]
				xsumi[i] = o.Ro*xsumi[i] + (1-o.Ro)*dx*dx
				m.W[i] += dx
			}
		case NETSTEROV:
			dx := o.getPreviousGradient(name, m)
			gsumi := make([]float32, m.Numel())
			assembler.Saxplusbysetz(o.Momentum, dx, o.LearningRate, m.DW, gsumi)
			assembler.Saxplusbyplusz(o.Momentum, dx, -(1.0 + o.Momentum), gsumi, m.W)
			o.setPreviousGradient(name, gsumi)
		case SGD:
			if o.Momentum > 0.0 {
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
