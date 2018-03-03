package assembler

import (
	"math/rand"
	"testing"
	"time"
)

func init(){
	rand.Seed(time.Now().Unix())
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
