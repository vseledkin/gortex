package assembler

import (
	"math/rand"
	"testing"
)


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

func TestSaxplusbysetz(t *testing.T) {

	x := make([]float32, 10000)
	y := make([]float32, 10000)
	z1 := make([]float32, 10000)
	z2 := make([]float32, 10000)
	a := rand.Float32()
	b := rand.Float32()
	for i := range x {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
	}
	for i := 1; i < 10000; i++ {
		saxplusbysetz(a, x[:i], b, y[:i], z1[:i])
		Saxplusbysetz(a, x[:i], b, y[:i], z2[:i])
		for j := 0; j < i; j++ {
			if z1[j] != z2[j] {
				t.Fatalf("Experiment %d values do not match want %f got %f in vector of length %d\n", i, z1[j], z2[j], len(z2[:i]))
			}
		}
	}
}

func TestSaxplusbyplusz(t *testing.T) {

	x := make([]float32, 10000)
	y := make([]float32, 10000)
	z1 := make([]float32, 10000)
	z2 := make([]float32, 10000)
	a := rand.Float32()
	b := rand.Float32()
	for i := range x {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
	}
	for i := 1; i < 10000; i++ {
		saxplusbyplusz(a, x[:i], b, y[:i], z1[:i])
		Saxplusbyplusz(a, x[:i], b, y[:i], z2[:i])
		for j := 0; j < i; j++ {
			if z1[j] != z2[j] {
				t.Fatalf("Experiment %d values do not match want %f got %f in vector of length %d\n", i, z1[j], z2[j], len(z2[:i]))
			}
		}
	}
}

func TestSaxplusbyvsetz(t *testing.T) {

	x := make([]float32, 10000)
	y := make([]float32, 10000)
	v := make([]float32, 10000)
	z1 := make([]float32, 10000)
	z2 := make([]float32, 10000)
	a := rand.Float32()
	b := rand.Float32()
	for i := range x {
		x[i] = rand.Float32()
		y[i] = rand.Float32()
		v[i] = rand.Float32()
	}
	for i := 1; i < 10000; i++ {
		saxplusbyvsetz(a, x[:i], b, y[:i], v[:i], z1[:i])
		Saxplusbyvsetz(a, x[:i], b, y[:i], v[:i], z2[:i])
		for j := 0; j < i; j++ {
			if z1[j] != z2[j] {
				t.Fatalf("Experiment %d values do not match want %f got %f in vector of length %d\n",
					i, z1[j], z2[j], len(z2[:i]))
			}
		}
	}
}

func TestSaxdivsqrteyplusz(t *testing.T) {

	x := make([]float32, 10000)
	y := make([]float32, 10000)

	a := rand.Float32()
	b := rand.Float32() + 1e-1
	for i := range x {
		x[i] = rand.Float32()
		y[i] = Abs(rand.Float32())
	}
	for i := 1; i < 10000; i++ {
		z1 := make([]float32, i)
		z2 := make([]float32, i)
		for i := range z1 {
			z1[i] = rand.Float32()
			z2[i] = z1[i]
		}
		saxdivsqrteyplusz(a, x[:i], b, y[:i], z1[:i])
		Saxdivsqrteyplusz(a, x[:i], b, y[:i], z2[:i])
		for j := 0; j < i; j++ {
			if z1[j] != z2[j] {
				t.Fatalf("Experiment %d values do not match want %f got %f in vector of length %d\n",
					i, z1[j], z2[j], len(z2[:i]))
			}
		}
	}
}

func TestSigmoidbackprop(t *testing.T) {

	x := make([]float32, 10000)
	y := make([]float32, 10000)

	for i := range x {
		x[i] = rand.Float32()
		y[i] = Abs(rand.Float32())
	}
	for i := 1; i < 10000; i++ {
		z1 := make([]float32, i)
		z2 := make([]float32, i)
		for i := range z1 {
			z1[i] = rand.Float32()
			z2[i] = z1[i]
		}
		sigmoidbackprop(1, x[:i], y[:i], z1[:i])
		Sigmoidbackprop(1, x[:i], y[:i], z2[:i])
		for j := 0; j < i; j++ {
			if z1[j] != z2[j] {
				t.Fatalf("Experiment %d values do not match want %f got %f in vector of length %d\n",
					i, z1[j], z2[j], len(z2[:i]))
			}
		}
	}
}
