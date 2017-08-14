package gortex

import "fmt"

// Gated recurrent unit

type VAE struct {
	z_size int
	W      *Matrix
	B      *Matrix

	WM *Matrix
	BM *Matrix

	WD *Matrix
	BD *Matrix
}

func MakeVae(x_size, z_size int) *VAE {
	vae := new(VAE)
	vae.z_size = z_size
	vae.W = RandXavierMat(z_size, x_size)
	vae.B = RandXavierMat(z_size, 1)

	vae.WM = RandXavierMat(z_size, z_size)
	vae.BM = RandXavierMat(z_size, 1)

	vae.WD = RandXavierMat(z_size, z_size)
	vae.BD = RandXavierMat(z_size, 1)

	return vae
}

func (vae *VAE) GetParameters(namespace string) map[string]*Matrix {
	return map[string]*Matrix{
		namespace + "_W": vae.W,
		namespace + "_B": vae.B,

		namespace + "_WM": vae.WM,
		namespace + "_BM": vae.BM,

		namespace + "_WD": vae.WD,
		namespace + "_BD": vae.BD,
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
	xz := g.Tanh(g.Add(g.Mul(vae.W, x), vae.B))
	mean = g.Add(g.Mul(vae.WM, xz), vae.BM)
	logvar = g.Add(g.Mul(vae.WD, xz), vae.BD)
	// sample random vector from normal 0 1 distribution
	eps := RandMat(vae.z_size, 1)
	// sample exemplar from generated distribution
	sample = g.Add(mean, g.EMul(g.Exp(g.MulConstant(0.5, logvar)), eps))
	return
}

func (vae *VAE) KLD(g *Graph, scale float32, mean, logvar *Matrix) float32 {
	// make VAE computation graph
	kld := g.MulConstant(-0.5, g.Sum(g.Sub(g.Sub(g.AddConstant(1.0, logvar), g.EMul(mean, mean)), g.Exp(logvar))))
	if g.NeedsBackprop && scale > 0 {
		g.backprop = append(g.backprop, func() {
			kld.DW[0] += scale * kld.W[0]
		})
	}
	return kld.W[0]
}
