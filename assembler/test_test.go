package assembler

import (
	"testing"
	"math/rand"
)

// scalar functions

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
		if Abs(sumfast-s) > 1e-5 {
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

// vector to scalar functions

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
		if Abs(sumfast-s) > 5e-5 {
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

// vector , scalar to vector functions

func scalarVector2VectorTest(f, ft func(a float32, v []float32), t *testing.T) {
	// for test purpocces
	x := make([]float32, 100)
	a := make([]float32, 100)
	for i := range x {
		x[i] = float32(rand.NormFloat64())
		a[i] = float32(rand.NormFloat64())
	}
	for i := 0; i < 100; i++ {
		var fast, purego []float32
		copy(fast, x[:i])
		copy(purego, x[:i])
		f(a[i], fast)
		ft(a[i], purego)

		for j := range fast {
			if Abs(fast[j]-purego[j]) > 5e-5 {
				t.Fatalf("sums do not match want %0.6f got %0.6f in vector of length %d\n", purego[j], fast[j], len(fast))
			}
		}
	}
}

func vectorScalar2VectorBench(f func(a float32, v []float32), b *testing.B) {
	b.StopTimer()
	x := make([]float32, 100)
	a := make([]float32, 100)
	for i := range x {
		x[i] = float32(rand.NormFloat64())
		a[i] = float32(rand.NormFloat64())
	}
	b.StartTimer()
	for i := 0; i < 100; i++ {
		for j := 0; j < b.N; j++ {
			f(a[i], x[:i])
			f(1/a[i], x[:i])
		}
	}
}
