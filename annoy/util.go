package annoy

import (
	"math"
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
		normalize(iv)
		normalize(jv)
	}

	ic := 1
	jc := 1

	for l := 0; l < iteration_steps; l++ {
		k := random.index(count)

		di := float32(ic) * distance.distance(iv, nodes[k].v)
		dj := float32(jc) * distance.distance(jv, nodes[k].v)

		norm := float32(1.0)
		if cosine {
			norm = getNorm(nodes[k].v)
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

func normalize(v []float32) []float32 {
	norm := getNorm(v)
	for z, _ := range v {
		v[z] /= norm
	}
	return v
}

func getNorm(v []float32) float32 {
	var sq_norm float32
	for _, vz := range v {
		sq_norm += vz * vz
	}
	return Sqrt(sq_norm)
}

func Sqrt(v float32) float32 {
	return float32(math.Sqrt(float64(v)))
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
