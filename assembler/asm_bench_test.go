package assembler

import (
	"math/rand"
	"testing"
)

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
