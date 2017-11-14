//+build !amd64 noasm

package assembler

func L1(x []float32) (sum float32) {
	for _, v := range x {
		if v > 0 {
			sum += v
			continue
		}
		sum -= v
	}
	return
}
