package assembler

import (
	"math/rand"
	"testing"
)

func BenchmarkSset(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1000000)
	for i := range x {
		x[i] = .1
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		sset(2.1, x)
	}
}

func BenchmarkSsetOptimized(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1000000)
	for i := range x {
		x[i] = .1
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		Sset(2.1, x)
	}
}

func BenchmarkSxpy(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1000000)
	y := make([]float32, 1000000)
	for i := range x {
		x[i] = .1
		y[i] = .2
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		sxpy(x, y)
	}
}

func BenchmarkSxpyOptimized(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1000000)
	y := make([]float32, 1000000)
	for i := range x {
		x[i] = .1
		y[i] = .2
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		Sxpy(x, y)
	}
}

func BenchmarkSdot(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1000000)
	y := make([]float32, 1000000)
	for i := range x {
		x[i] = .1
		y[i] = .2
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		sdot(x, y)
	}
}

func BenchmarkOptimizedSdot(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1000000)
	y := make([]float32, 1000000)
	for i := range x {
		x[i] = .1
		y[i] = .2
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		Sdot(x, y)
	}
}

func BenchmarkSclean(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1000000)
	for i := range x {
		x[i] = .1
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		sclean(x)
	}
}

func BenchmarkOptimizedSclean(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1000000)
	for i := range x {
		x[i] = .1
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		Sclean(x)
	}
}

func BenchmarkSaxplusbysetz(tb *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	tb.StopTimer()

	x := make([]float32, 10000)
	y := make([]float32, 10000)
	z := make([]float32, 10000)
	a := rand.Float32()
	b := rand.Float32()
	for i := range x {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
	}
	tb.StartTimer() //restart timer
	for i := 0; i < tb.N; i++ {
		saxplusbysetz(a, x, b, y, z)
	}
}

func BenchmarkOptimizedSaxplusbysetz(tb *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	tb.StopTimer()

	x := make([]float32, 10000)
	y := make([]float32, 10000)
	z := make([]float32, 10000)
	a := rand.Float32()
	b := rand.Float32()
	for i := range x {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
	}
	tb.StartTimer() //restart timer
	for i := 0; i < tb.N; i++ {
		Saxplusbysetz(a, x, b, y, z)
	}
}

func BenchmarkSaxplusbyplusz(tb *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	tb.StopTimer()

	x := make([]float32, 10000)
	y := make([]float32, 10000)
	z := make([]float32, 10000)
	a := rand.Float32()
	b := rand.Float32()
	for i := range x {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
	}
	tb.StartTimer() //restart timer
	for i := 0; i < tb.N; i++ {
		saxplusbyplusz(a, x, b, y, z)
	}
}

func BenchmarkOptimizedSaxplusbyplusz(tb *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	tb.StopTimer()

	x := make([]float32, 10000)
	y := make([]float32, 10000)
	z := make([]float32, 10000)
	a := rand.Float32()
	b := rand.Float32()
	for i := range x {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
	}
	tb.StartTimer() //restart timer
	for i := 0; i < tb.N; i++ {
		Saxplusbyplusz(a, x, b, y, z)
	}
}

func BenchmarkSaxplusbyvsetz(tb *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	tb.StopTimer()

	x := make([]float32, 10000)
	y := make([]float32, 10000)
	v := make([]float32, 10000)
	z := make([]float32, 10000)
	a := rand.Float32()
	b := rand.Float32()
	for i := range x {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
		v[i] = rand.Float32()
	}
	tb.StartTimer() //restart timer
	for i := 0; i < tb.N; i++ {
		saxplusbyvsetz(a, x, b, y, v, z)
	}
}

func BenchmarkOptimizedSaxplusbyvsetz(tb *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	tb.StopTimer()

	x := make([]float32, 10000)
	y := make([]float32, 10000)
	v := make([]float32, 10000)
	z := make([]float32, 10000)
	a := rand.Float32()
	b := rand.Float32()
	for i := range x {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
		v[i] = rand.Float32()
	}
	tb.StartTimer() //restart timer
	for i := 0; i < tb.N; i++ {
		Saxplusbyvsetz(a, x, b, y, v, z)
	}
}

func BenchmarkSaxdivsqrteyplusz(tb *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	tb.StopTimer()

	x := make([]float32, 10000)
	y := make([]float32, 10000)
	z := make([]float32, 10000)
	a := rand.Float32()
	b := rand.Float32()
	for i := range x {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
	}
	tb.StartTimer() //restart timer
	for i := 0; i < tb.N; i++ {
		saxdivsqrteyplusz(a, x, b, y, z)
	}
}

func BenchmarkOptimizedSaxdivsqrteyplusz(tb *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	tb.StopTimer()

	x := make([]float32, 10000)
	y := make([]float32, 10000)
	z := make([]float32, 10000)
	a := rand.Float32()
	b := rand.Float32()
	for i := range x {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
	}
	tb.StartTimer() //restart timer
	for i := 0; i < tb.N; i++ {
		Saxdivsqrteyplusz(a, x, b, y, z)
	}
}

func BenchmarkSigmoidbackprop(tb *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	tb.StopTimer()

	x := make([]float32, 10000)
	y := make([]float32, 10000)
	z := make([]float32, 10000)
	for i := range x {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
		z[i] = rand.Float32()
	}
	tb.StartTimer() //restart timer
	for i := 0; i < tb.N; i++ {
		sigmoidbackprop(1, x, y, z)
	}
}

func BenchmarkOptimizedSigmoidbackprop(tb *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	tb.StopTimer()

	x := make([]float32, 10000)
	y := make([]float32, 10000)
	z := make([]float32, 10000)
	for i := range x {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
		z[i] = rand.Float32()
	}
	tb.StartTimer() //restart timer
	for i := 0; i < tb.N; i++ {
		Sigmoidbackprop(1, x, y, z)
	}
}
