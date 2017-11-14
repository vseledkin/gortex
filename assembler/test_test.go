package assembler

import (
	"math"
	"testing"
	"math/rand"
)


func scalarTest(f, ft func(v float32) float32, t *testing.T) {
	// for test purpocces
	x := make([]float32, 100)
	for i := range x {
		x[i] = float32(rand.NormFloat64())
	}
	for i := 0; i < 100; i++ {
		sumfast := f(x[i])
		s := ft(x[i])
		// numeric issues
		if float32(math.Abs(float64(sumfast-s))) > 1e-5 {
			t.Fatalf("sums do not match want %0.6f got %0.6f in vector of length %d\n", s, sumfast, len(x[:i]))
		}
	}
}

func scalarBench(f func(v float32) float32, b *testing.B) {
	b.StopTimer()
	x := make([]float32, 100)
	for i := range x {
		x[i] = float32(rand.NormFloat64())
	}
	b.StartTimer()
	for i := 0; i < 100; i++ {
		for j := 0; j < b.N; j++ {
			f(x[i])
		}
	}
}

func vector2ScalarTest(f, ft func(v []float32) float32, t *testing.T) {
	// for test purpocces
	x := make([]float32, 100)
	for i := range x {
		x[i] = float32(rand.NormFloat64())
	}
	for i := 0; i < 100; i++ {
		sumfast := f(x[:i])
		s := ft(x[:i])
		// numeric issues
		if float32(math.Abs(float64(sumfast-s))) > 5e-5 {
			t.Fatalf("sums do not match want %0.6f got %0.6f in vector of length %d\n", s, sumfast, len(x[:i]))
		}
	}
}

func vector2ScalarBench(f func(v []float32) float32, b *testing.B) {
	b.StopTimer()
	x := make([]float32, 100)
	for i := range x {
		x[i] = float32(rand.NormFloat64())
	}
	b.StartTimer()
	for i := 0; i < 100; i++ {
		for j := 0; j < b.N; j++ {
			f(x[:i])
		}
	}
}
