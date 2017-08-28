package gortex

import "fmt"

// Gated recurrent unit

type VAE struct {
	z_size int

	W  *Matrix
	WB *Matrix

	W1  *Matrix
	W1B *Matrix

	WM *Matrix
	WD *Matrix

	WW1  *Matrix
	WW1B *Matrix

	WW  *Matrix
	WWB *Matrix
}

func MakeVae(x_size, z_size int) *VAE {
	vae := new(VAE)
	vae.z_size = z_size

	vae.W = RandXavierMat(z_size, x_size)
	vae.WB = RandXavierMat(z_size, 1)

	vae.W1 = RandXavierMat(z_size, z_size)
	vae.W1B = RandXavierMat(z_size, 1)

	vae.WW = RandXavierMat(x_size, z_size)
	vae.WWB = RandXavierMat(x_size, 1)

	vae.WW1 = RandXavierMat(x_size, x_size)
	vae.WW1B = RandXavierMat(x_size, 1)

	vae.WM = RandXavierMat(z_size, z_size)
	vae.WD = RandXavierMat(z_size, z_size)

	return vae
}

func (vae *VAE) GetParameters(namespace string) map[string]*Matrix {
	return map[string]*Matrix{
		namespace + "_W":  vae.W,
		namespace + "_WB": vae.WB,

		namespace + "_W1":  vae.W1,
		namespace + "_W1B": vae.W1B,

		namespace + "_WW":  vae.WW,
		namespace + "_WWB": vae.WWB,

		namespace + "_WW1":  vae.WW1,
		namespace + "_WW1B": vae.WW1B,

		namespace + "_WM": vae.WM,
		namespace + "_WD": vae.WD,
	}
}

func (vae *VAE) SetParameters(namespace string, parameters map[string]*Matrix) error {
	for k, v := range vae.GetParameters(namespace) {
		fmt.Printf("Look for %s parameters\n", k)
		if m, ok := parameters[k]; ok {
			fmt.Printf("Got %s parameters\n", k)
			copy(v.W, m.W)
		} else {
			return fmt.Errorf("Model geometry is not compatible, parameter %s is unknown", k)
		}
	}
	return nil
}

func (vae *VAE) Step(g *Graph, x *Matrix) (sample, mean, logvar *Matrix) {
	// make VAE computation graph
	xz := g.MulConstant(1.3, g.Tanh(g.Add(g.Mul(vae.W, x), vae.WB)))
	xz = g.MulConstant(1.3, g.Tanh(g.Add(g.Mul(vae.W1, xz), vae.W1B)))
	mean = g.Mul(vae.WM, xz)
	logvar = g.Mul(vae.WD, xz)
	// sample random vector from normal 0 1 distribution
	eps := RandMatMD(vae.z_size, 1, 0, 0.01)

	// sample exemplar from generated distribution
	sample = g.Add(mean, g.EMul(g.Exp(g.MulConstant(0.5, logvar)), eps))
	return
}

func (vae *VAE) StepAmplitude(g *Graph, x *Matrix, amplitude float64) (sample, mean, logvar *Matrix) {
	// make VAE computation graph
	xz := g.MulConstant(1.6, g.Tanh(g.Add(g.Mul(vae.W, x), vae.WB)))
	xz = g.MulConstant(1.6, g.Tanh(g.Add(g.Mul(vae.W1, xz), vae.W1B)))
	//xz := g.Tanh(g.Add(g.Mul(vae.W, x), vae.WB))
	//xz = g.Tanh(g.Add(g.Mul(vae.W1, xz), vae.W1B))
	mean = g.Mul(vae.WM, xz)
	logvar = g.Mul(vae.WD, xz)
	// sample random vector from normal 0 1 distribution
	eps := RandMatMD(vae.z_size, 1, 0, amplitude)

	// sample exemplar from generated distribution
	sample = g.Add(mean, g.EMul(g.Exp(g.MulConstant(0.5, logvar)), eps))
	return
}

func (vae *VAE) Step1(g *Graph, x *Matrix) (y *Matrix) {
	y = g.MulConstant(1.6, g.Tanh(g.Add(g.Mul(vae.WW, x), vae.WWB)))
	y = g.MulConstant(1.6, g.Tanh(g.Add(g.Mul(vae.WW1, y), vae.WW1B)))
	return
}

func (vae *VAE) KLD(g *Graph, scale float32, mean, logvar *Matrix) float32 {
	// make VAE computation graph
	if scale == 0 {
		g.NeedsBackprop = false
	}
	kld := g.MulConstant(-0.5, g.Sum(g.Sub(g.Sub(g.AddConstant(1.0, logvar), g.EMul(mean, mean)), g.Exp(logvar))))
	if scale == 0 {
		g.NeedsBackprop = true
	}
	if g.NeedsBackprop && scale > 0 {
		g.backprop = append(g.backprop, func() {
			kld.DW[0] += scale * kld.W[0]
		})
	}
	return kld.W[0]
}
