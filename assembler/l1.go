package assembler

func l1asm(x []float32) float32

func l1(x []float32) (sum float32) {
	for _, v := range x {
		if v > 0 {
			sum += v
			continue
		}
		sum -= v
	}
	return
}
