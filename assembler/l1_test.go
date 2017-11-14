package assembler

import (
	"testing"
)

func l1(x []float32) (sum float32) {
	for i := range x {
		if x[i] > 0 {
			sum += x[i]
		} else {
			sum -= x[i]
		}
	}
	return
}

func TestL1(t *testing.T) {
	vector2ScalarTest(L1, l1, t)
}

func BenchmarkL1(b *testing.B) {
	vector2ScalarBench(l1, b)
}

func BenchmarkOptimizedL1(b *testing.B) {
	vector2ScalarBench(L1, b)
}
