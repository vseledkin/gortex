package assembler

// Index of largest (absoulute) element of the vector X
func isamax_asm(X []float32) int

func isamax(X []float32) int {
	var (
		max_x float32
		max_i int
	)
	for i, x := range X {
		if x < 0 {
			x = -x
		}
		if x > max_x {
			max_x = x
			max_i = i
		}
	}
	return max_i
}
