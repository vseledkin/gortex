package assembler

import (
	"testing"
	"math"
	"math/rand"
)

func l1test(t *testing.T,b *testing.B) {
	for j := 0; j < 100; j++ {
		if b!=nil {
			b.StopTimer()
		}
		x := make([]float32, j)
		for i := range x {
			x[i] = float32(rand.NormFloat64())
		}
		if b!=nil {
			b.StartTimer()
			for i := 0; i < b.N; i++ {
				L1(x)
			}
		}
		if t!=nil{
			sumfast := l1asm(x)
			sum := l1(x)
			// numeric issues
			if float32(math.Abs(float64(sumfast-sum))) > 1e-4 {
				t.Fatalf("sums do not match want %0.6f got %0.6f in vector of length %d\n", sum, sumfast, len(x))
			}
		}
	}
}

func TestL1(t *testing.T) {
	l1test(t,nil)
}

func BenchmarkL1(b *testing.B) {
	Init(false)
	l1test(nil,b)
}

func BenchmarkOptimizedL1(b *testing.B) {
	Init(true)
	l1test(nil,b)
}