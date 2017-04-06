package assembler

import (
	"math"
	"math/rand"
	"testing"
)

func TestSscale(t *testing.T) {
	for j := 0; j < 100; j++ {
		X := make([]float32, j)
		X1 := make([]float32, j)
		Y := make([]float32, j)
		for i := range X {
			X[i] = float32(rand.NormFloat64())
			X1[i] = X[i]
		}
		factor := float32(rand.NormFloat64())
		sscale(factor, X)
		Sscale(factor, X1)
		for i := range Y {
			if X[i] != X1[i] {
				t.Fatalf("sums do not match want %f got %f in vector of length %d\n", X[i], X1[i], len(X))
			}
		}
	}
}

func TestL1(t *testing.T) {
	for j := 0; j < 100; j++ {
		vector := make([]float32, j)
		for i := range vector {
			vector[i] = float32(rand.NormFloat64())
		}
		sumfast := L1(vector)
		sum := l1(vector)
		// numeric issues
		if float32(math.Abs(float64(sumfast-sum))) > 1e-5 {
			t.Fatalf("sums do not match want %0.6f got %0.6f in vector of length %d\n", sum, sumfast, len(vector))
		}
	}
}

func TestL2(t *testing.T) {
	for j := 0; j < 100; j++ {
		X := make([]float32, j)
		for i := range X {
			X[i] = .1
		}
		dot := l2(X)
		dot1 := L2(X)
		if dot-dot1 > 0.000001 || dot-dot1 < -0.000001 {
			t.Fatalf("norm do not match want %f got %f in vector of length %d\n", dot, dot1, len(X))
		}
	}
}

func TestSdot(t *testing.T) {
	x := []float32{0.40724772, 0.0712502, 0.041903675, 0.15231317, 0.21472728, 0.4622725, -0.0903995, 0.24077353, 0.006599188, -0.47139943, 0.3086093, 0.1786874, 0.42446965, 0.22735131, 0.46515256}
	//y := []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	y := make([]float32, len(x))
	dot := sdot(x, y)
	dot1 := Sdot(x, y)

	if dot1 != 0 || dot != 0 || dot-dot1 > 0.00004 || dot-dot1 < -0.00004 {
		t.Fatalf("dot do not match want %f got %f in vector of length %d\n", dot, dot1, len(x))
	}

	for j := 0; j < 100; j++ {
		X := make([]float32, j)
		Y := make([]float32, j)
		for i := range X {
			X[i] = rand.Float32()*2. - 1.
			Y[i] = rand.Float32()*2. - 1.
		}

		for i := 0; i < 10; i++ {
			dot := sdot(X, Y)
			dot1 := Sdot(X, Y)
			if dot-dot1 > 0.00004 || dot-dot1 < -0.00004 {
				t.Fatalf("dot do not match want %f got %f in vector of length %d in test %d\n", dot, dot1, len(X), i)
			}
		}
	}
}

func TestL2squared(t *testing.T) {
	for j := 0; j < 100; j++ {
		X := make([]float32, j)
		for i := range X {
			X[i] = .1
		}
		dot := l2squared(X)
		dot1 := L2squared(X)
		if dot-dot1 > 0.000001 || dot-dot1 < -0.000001 {
			t.Fatalf("L2squared do not match want %f got %f in vector of length %d\n", dot, dot1, len(X))
		}
	}
}


func TestSaxpy(t *testing.T) {
	for j := 0; j < 100; j++ {
		X := make([]float32, j)
		Y := make([]float32, j)
		Y1 := make([]float32, j)
		for i := range X {
			X[i] = rand.Float32()*0.2 - 0.1
			Y[i] = rand.Float32()*0.2 - 0.1
		}
		for i, y := range Y {
			Y1[i] = y
		}
		r := rand.Float32()*0.2 - 0.1
		saxpy(r, X, Y)
		//fmt.Println(Y1)
		Saxpy(r, X, Y1)
		//fmt.Println(Y1)
		for i := range Y {
			if Y[i] != Y1[i] {
				t.Fatalf("sums do not match want %f got %f in vector of length %d\n", Y[i], Y1[i], len(Y))
			}
		}
	}
}

func TestSset(t *testing.T) {
	for j := 0; j < 100; j++ {
		X := make([]float32, j)
		X1 := make([]float32, j)
		for i := range X {
			X[i] = .1
			X1[i] = .1
		}
		sset(2.1, X)
		Sset(2.1, X1)
		for i := range X {
			if X[i] != X1[i] {
				t.Fatalf("values do not match want %f got %f in vector of length %d\n", X[i], X1[i], len(X))
			}
			if X[i] != 2.1 {
				t.Fatalf("values do not match want %f got %f in vector of length %d\n", 2.1, X1[i], len(X))
			}
		}
	}
}