package gortex

type SGDSolver struct{}

func NewSGDSolver() *SGDSolver {
	s := new(SGDSolver)
	return s
}

func (this *SGDSolver) Step(model map[string]*Matrix, step_size float32) {
	for _, m := range model {
		for i := range m.W {
			m.W[i] += -step_size * m.DW[i]
			m.DW[i] = 0 // reset gradients for next iteration
		}
	}
	return
}
