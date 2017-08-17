package gortex

import "github.com/vseledkin/gortex/assembler"

type Solver struct {
	decay_rate float32
	smooth_eps float32
	step_cache map[string]*Matrix
}

func NewSolver() *Solver {
	s := new(Solver)
	s.decay_rate = 0.999
	s.smooth_eps = 1e-8
	s.step_cache = make(map[string]*Matrix)
	return s
}

func (this *Solver) Step(model map[string]*Matrix, step_size, regc, clipval float32) map[string]float32 {
	// perform parameter update
	solver_stats := make(map[string]float32)
	var num_clipped, num_tot float32

	for k, m := range model {
		s, ok := this.step_cache[k]
		if !ok {
			s = Mat(m.Rows, m.Columns)
			this.step_cache[k] = s
		}
		l := len(m.W)
		for i := 0; i < l; i++ {
			// rmsprop adaptive learning rate
			var mdwi = m.DW[i]
			s.W[i] = s.W[i]*this.decay_rate + (1.0-this.decay_rate)*mdwi*mdwi
			if clipval > 0 {
				// gradient clip
				if mdwi > clipval {
					mdwi = clipval
					num_clipped++
				}
				if mdwi < -clipval {
					mdwi = -clipval
					num_clipped++
				}
			}
			num_tot++

			// update (and regularize)
			m.W[i] += -step_size*mdwi/assembler.Sqrt(s.W[i]+this.smooth_eps) - regc*m.W[i]
			m.DW[i] = 0 // reset gradients for next iteration
		}
	}
	solver_stats["ratio_clipped"] = num_clipped * 1.0 / num_tot
	return solver_stats
}
