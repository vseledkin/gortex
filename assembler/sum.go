//+build !amd64 noasm

package assembler

func Sum(x []float32) (sum float32) {
	for i := range x {
		sum += x[i]
	}
	return
}



