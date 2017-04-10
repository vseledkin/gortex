package assembler

// Index of largest element of the vector X
func ismax_asm(X []float32) int

func ismax(X []float32) int {
	var (
		max_x float32
		max_i int
	)
	for i, x := range X {
		if x > max_x {
			max_x = x
			max_i = i
		}
	}
	return max_i
}
