package assembler

import (
	"math/rand"
	"testing"
)

func BenchmarkSscale(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1024)
	for i := range x {
		x[i] = float32(rand.NormFloat64())
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		sscale(float32(rand.NormFloat64()), x)
	}
}

func BenchmarkOptimizedSscale(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1024)
	for i := range x {
		x[i] = float32(rand.NormFloat64())
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		Sscale(float32(rand.NormFloat64()), x)
	}
}


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

func BenchmarkL2(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1000000)
	for i := range x {
		x[i] = .1
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		l2(x)
	}
}

func BenchmarkOptimizedL2(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1000000)
	for i := range x {
		x[i] = .1
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		L2(x)
	}
}

func BenchmarkL2squared(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1000000)
	for i := range x {
		x[i] = .1
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		l2squared(x)
	}
}

func BenchmarkOptimizedL2squared(b *testing.B) { //benchmark function starts with "Benchmark" and takes a pointer to type testing.B
	b.StopTimer()

	x := make([]float32, 1000000)
	for i := range x {
		x[i] = .1
	}
	b.StartTimer() //restart timer
	for i := 0; i < b.N; i++ {
		L2squared(x)
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
