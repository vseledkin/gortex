package assembler

import (
	"testing"
	"math"
	"math/rand"
)

func sumtest(t *testing.T,b *testing.B) {
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
				Sum(x)
			}
		}
		if t!=nil{
			sumfast := sumasm(x)
			s := sum(x)
			// numeric issues
			if float32(math.Abs(float64(sumfast-s))) > 1e-5 {
				t.Fatalf("sums do not match want %0.6f got %0.6f in vector of length %d\n", s, sumfast, len(x))
			}
		}
	}
}

func TestSum(t *testing.T) {
	sumtest(t,nil)
}

func BenchmarkSum(b *testing.B) {
	Init(false)
	sumtest(nil,b)
}

func BenchmarkOptimizedSum(b *testing.B) {
	Init(true)
	sumtest(nil,b)
}