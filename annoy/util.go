package annoy

import (
	"github.com/vseledkin/gortex/assembler"
)

func twoMeans(distance Distance, nodes []Node, random Random, cosine bool) ([]float32, []float32) {
	iteration_steps := 200
	count := len(nodes)

	i := random.index(count)
	j := random.index(count - 1)
	if j >= i {
		j++
	}
	iv := make([]float32, len(nodes[i].v))
	copy(iv, nodes[i].v)

	jv := make([]float32, len(nodes[j].v))
	copy(jv, nodes[j].v)

	if cosine {
		assembler.Sscale(1/assembler.L2(iv),iv)
		assembler.Sscale(1/assembler.L2(jv),jv)
	}

	ic := 1
	jc := 1

	for l := 0; l < iteration_steps; l++ {
		k := random.index(count)

		di := float32(ic) * distance.distance(iv, nodes[k].v)
		dj := float32(jc) * distance.distance(jv, nodes[k].v)

		norm := float32(1.0)
		if cosine {
			norm = assembler.L2(nodes[k].v)
		}

		if di < dj {
			for z, _ := range iv {
				iv[z] = (iv[z]*float32(ic) + nodes[k].v[z]/norm) / float32(ic+1)
			}
			ic++
		} else if dj < di {
			for z, _ := range jv {
				jv[z] = (jv[z]*float32(jc) + nodes[k].v[z]/norm) / float32(jc+1)
			}
			jc++
		}
	}
	return iv, jv
}

func Max(x, y float32) float32 {
	if x < y {
		return y
	}
	return x
}

func Min(x, y float32) float32 {
	if x < y {
		return x
	}
	return y
}
