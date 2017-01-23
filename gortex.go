package gortex

func Zeros(n int) []float64 {
	if n <= 0 {
		return []float64{}
	}
	return make([]float64, n)
}

