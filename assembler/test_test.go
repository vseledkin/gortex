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
		x[i] = rand.Float32()
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
		x[i] = rand.Float32()
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
		x[i] = rand.Float32()
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
		x[i] = rand.Float32()
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
		x[i] = rand.Float32()
		a[i] = rand.Float32()
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

func scalarVector2VectorBench(f func(a float32, v []float32), b *testing.B) {
	b.StopTimer()
	x := make([]float32, 100)
	a := make([]float32, 100)
	for i := range x {
		x[i] = rand.Float32()
		a[i] = rand.Float32()
	}
	b.StartTimer()
	for i := 0; i < 100; i++ {
		for j := 0; j < b.N; j++ {
			f(a[i], x[:i])
			f(1/a[i], x[:i])
		}
	}
}

func make2PairsOfVectors(l int) ([]float32, []float32, []float32, []float32) {
	x1 := make([]float32, l)
	y1 := make([]float32, l)
	x2 := make([]float32, l)
	y2 := make([]float32, l)
	for i := range x1 {
		y1[i] = rand.Float32()
		x1[i] = rand.Float32()
	}
	copy(x2, x1)
	copy(y2, y1)
	return x1, y1, x2, y2
}

func makePairOfVectors(l int) ([]float32, []float32) {
	x := make([]float32, l)
	y := make([]float32, l)
	for i := range x {
		y[i] = rand.Float32()
		x[i] = rand.Float32()
	}
	return x, y
}

func EQ(x, y []float32, t *testing.T, epsilon float32) {
	if len(x) != len(y) {
		t.Fatalf("Lengths are not equal %d!=%d", len(x), len(y))
	}
	for j := range x {
		if Abs(x[j]-y[j]) > epsilon {
			t.Fatalf(`do not match want %0.6f got %0.6f in vector of length %d
at position %d
%v
%v`, x[j], y[j], len(x), j, x, y)
		}
	}
}

// functions of two vectors
func funcVectorVectorTest(f, ft func(x, y []float32), t *testing.T) {
	for i := 0; i < testVectorLength; i++ {
		x1, y1, x2, y2 := make2PairsOfVectors(i)
		f(x1, y1)
		ft(x2, y2)
		// check
		EQ(x1, x2, t, 0)
		EQ(y1, y2, t, 0)
	}
}

// functions of two vectors giving scalar
func funcVectorVector2scalarTest(f, ft func(x, y []float32) float32, t *testing.T) {
	for i := 0; i < testVectorLength; i++ {
		x1, y1, x2, y2 := make2PairsOfVectors(i)
		z1 := f(x1, y1)
		z2 := ft(x2, y2)
		//t.Logf("vl %d",i)
		EQ([]float32{z1}, []float32{z2}, t, 1e-3)
		// check that arguments are unaffected
		EQ(x1, x2, t, 0)
		EQ(y1, y2, t, 0)
	}
}

// functions of scalar and pair of vectors
func funcScalarVectorVectorTest(f, ft func(a float32, x, y []float32), t *testing.T) {
	for i := 0; i < testVectorLength; i++ {
		a := rand.Float32()
		aa := a
		aaa := a
		x1, y1, x2, y2 := make2PairsOfVectors(i)
		f(aa, x1, y1)
		ft(aaa, x2, y2)
		//t.Logf("vl %d",i)
		EQ([]float32{a}, []float32{aa}, t, 0)

		// check that arguments are equal
		EQ(x1, x2, t, 0)
		EQ(y1, y2, t, 0)
	}
}

var testVectorLength = 8*128 + 4 + 3
// functions of scalar and pair of vectors
func funcScalarVectorVectorBench(f func(a float32, x, y []float32), b *testing.B) {
	b.StopTimer()
	a := rand.Float32()
	x, y := makePairOfVectors(testVectorLength)
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		f(a, x, y)
	}
}
func vector2SelfTest(f, ft func(x []float32), t *testing.T) {
	// for test purpocces
	y := make([]float32, 100)
	x := make([]float32, 100)
	for i := range x {
		y[i] = rand.Float32()
		x[i] = rand.Float32()
	}
	for i := 0; i < 100; i++ {
		var fast, purego []float32
		copy(fast, y[:i])
		copy(purego, y[:i])
		f(fast)
		ft(purego)

		for j := range purego {
			if Abs(purego[j]-fast[j]) > 5e-5 {
				t.Fatalf("sums do not match want %0.6f got %0.6f in vector of length %d\n", purego[j], fast[j], len(purego))
			}
		}
	}
}
func vector2selfBench(f func(x []float32), b *testing.B) {
	b.StopTimer()
	x := make([]float32, 100)
	y := make([]float32, 100)
	for i := range x {
		x[i] = rand.Float32()
	}
	b.StartTimer()
	for i := 0; i < 100; i++ {
		b.StopTimer()
		copy(y[:i], x[:i])
		b.StartTimer()
		for j := 0; j < b.N; j++ {
			f(y[:i])
		}
	}
}

func funcVectorVectorBench(f func(x, y []float32), b *testing.B) {
	b.StopTimer()
	x, y := makePairOfVectors(testVectorLength)
	b.StartTimer()
	b.N = 1000
	for i := 0; i < b.N; i++ {
		f(x, y)
	}
}

func funcVectorVector2scalarBench(f func(x, y []float32) float32, b *testing.B) {
	b.StopTimer()
	var z float32
	x, y := makePairOfVectors(testVectorLength)
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		z = f(x, y)
	}
	b.StopTimer()
	z = z + 1
	b.StartTimer()
}
