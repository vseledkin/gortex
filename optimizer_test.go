package gortex

import "testing"

func TestOptimizer(t *testing.T) {
	op := NewOptimizer(OpOp{Method: SGD, LearningRate: 0.0001, L1Decay: 0.000001})
	if op.LearningRate != 0.0001 {
		t.Fail()
	}
	println(op)
}
