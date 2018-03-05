package assembler

import "testing"

//Sset  set all components of a vector to a
func sset(a float32, x []float32) {
	for i := range x {
		x[i] = a
	}
}

func TestSset(t *testing.T) {
	scalarVector2VectorTest(Sset, sset, t)
}
func TestSsetAVX(t *testing.T) {
	if useAVX {
		scalarVector2VectorTest(SsetAVX, sset, t)
	}
}

func BenchmarkSset(b *testing.B) {
	scalarVector2VectorBench(sset, b)
}

func BenchmarkOptimizedSset(b *testing.B) {
	scalarVector2VectorBench(Sset, b)
}

func BenchmarkOptimizedSsetAVX(b *testing.B) {
	if useAVX {
		scalarVector2VectorBench(SsetAVX, b)
	}
}
