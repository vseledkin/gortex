package gortex

type OpMethod int

const (
	SGD OpMethod = iota
	ADAM
	ADAGRAD
	ADADELTA
	WINDOWGRAD
	NETSTEROV
)

type OpOp struct {
	LearningRate float32
	L1Decay      float32
	L2Decay      float32
	Momentum     float32
	Ro           float32 // used by adadelta
	Eps          float32 // used by adadelta and adam
	Beta1        float32 // used by adam
	Beta2        float32 // used by adam
	Method       OpMethod
}

type Optimizer struct {
	OpOp
}

func NewOptimizer(ops OpOp) *Optimizer {
	op := new(Optimizer)
	op.OpOp = ops
	if op.LearningRate == 0 {
		op.LearningRate = 0.01
	}
	if op.Momentum == 0 {
		op.Momentum = 0.9
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

	return op
}
