package assembler

func Ssum(x []float32) float32

func ssum(x []float32) (sum float32) {
	for _, v := range x {
		sum += v
	}
	return
}
