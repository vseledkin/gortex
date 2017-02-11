package gortex

import "math"

type Solver struct {
	decay_rate float64
	smooth_eps float64
	step_cache map[string]*Matrix
}

func NewSolver() *Solver {
	s := new(Solver)
	s.decay_rate = 0.999
	s.smooth_eps = 1e-8
	s.step_cache = make(map[string]*Matrix)
	return s
}

func (this *Solver) Step(model map[string]*Matrix, step_size, regc, clipval float64) map[string]float64 {
	// perform parameter update
	solver_stats := make(map[string]float64)
	num_clipped := 0.0
	num_tot := 0.0
	for k, m := range model {
		s, ok := this.step_cache[k]
		if !ok {
			this.step_cache[k] = Mat(m.Rows, m.Columns)
			s = this.step_cache[k]
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
			m.W[i] += -step_size*mdwi/math.Sqrt(s.W[i]+this.smooth_eps) - regc*m.W[i]
			m.DW[i] = 0 // reset gradients for next iteration
		}

	}
	solver_stats["ratio_clipped"] = num_clipped * 1.0 / num_tot
	return solver_stats
}
