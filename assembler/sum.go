package assembler

func sumasm(x []float32) float32

func sum(x []float32) (sum float32) {
	for _, v := range x {
		sum += v
	}
	return
}
