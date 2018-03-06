package assembler

import "testing"

//Sset  set all components of a vector to a
func sclean(x []float32) {
	for i := range x {
		x[i] = 0.0
	}
}

func TestSclean(t *testing.T) {
	vector2SelfTest(Sclean, sclean, t)
}

func BenchmarkSclean(b *testing.B) {
	vector2selfBench(sclean, b)
}

func BenchmarkOptimizedSclean(b *testing.B) {
	vector2selfBench(Sclean, b)
}
