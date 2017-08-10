package gortex

import "github.com/vseledkin/gortex/assembler"

type SGDSolver struct{}

func NewSGDSolver() *SGDSolver {
	s := new(SGDSolver)
	return s
}

func (this *SGDSolver) Step(model map[string]*Matrix, step_size float32) {
	for _, m := range model {
		assembler.Saxpy(-step_size, m.DW, m.W)
		assembler.Sclean(m.DW)
	}
	return
}
