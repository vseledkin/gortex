package assembler

import (
	"math/rand"
	"testing"
)

func isamaxtest(t *testing.T, b *testing.B) {
	for j := 0; j < 100; j++ {
		if b != nil {
			b.StopTimer()
		}
		x := make([]float32, j)
		for i := range x {
			x[i] = float32(rand.NormFloat64())
		}
		if b != nil {
			b.StartTimer()
			for i := 0; i < b.N; i++ {
				Isamax(x)
			}
		}
		if t != nil {
			ifast := isamax_asm(x)
			isimple := isamax(x)
			// numeric issues
			if ifast != isimple {

				t.Logf("%v\n", x)
				t.Fatalf("maxindex do not match want %d=%f got %d=%f in vector of length %d\n", isimple, x[isimple], ifast, x[ifast], len(x))
			}
		}
	}
}

func TestIsamax(t *testing.T) {
	isamaxtest(t, nil)
}

func BenchmarkIsamax(b *testing.B) {
	Init(false)
	isamaxtest(nil, b)
}

func BenchmarkOptimizedIsamax(b *testing.B) {
	Init(true)
	isamaxtest(nil, b)
}
