package assembler

import (
	"math/rand"
	"testing"
)

func ismaxtest(t *testing.T, b *testing.B) {
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
				Ismax(x)
			}
		}
		if t != nil {
			ifast := ismax_asm(x)
			isimple := ismax(x)
			// numeric issues
			if ifast != isimple {

				t.Logf("%v\n", x)
				t.Fatalf("maxindex do not match want %d=%f got %d=%f in vector of length %d\n", isimple, x[isimple], ifast, x[ifast], len(x))
			}
		}
	}
}

func TestIsmax(t *testing.T) {
	ismaxtest(t, nil)
}

func BenchmarkIsmax(b *testing.B) {
	Init(false)
	ismaxtest(nil, b)
}

func BenchmarkOptimizedIsmax(b *testing.B) {
	Init(true)
	ismaxtest(nil, b)
}
