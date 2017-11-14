package annoy

import "github.com/vseledkin/gortex/assembler"

type Distance interface {
	createSplit([]Node, Random, Node) Node
	distance([]float32, []float32) float32
	side(Node, []float32, Random) int
	margin(Node, []float32) float32
}

type Angular struct {
}

func (a Angular) createSplit(nodes []Node, random Random, n Node) Node {
	bestIv, bestJv := twoMeans(a, nodes, random, true)
	v := make([]float32, len(nodes[0].v))
	for z, _ := range v {
		v[z] = bestIv[z] - bestJv[z]
	}
	assembler.Sscale(1/assembler.L2(v), v)
	n.v = v
	return n
}

func (a Angular) distance(x, y []float32) float32 {
	var pp, qq, pq float32
	for z, xz := range x {
		pp += xz * xz
		qq += y[z] * y[z]
		pq += xz * y[z]
	}
	ppqq := pp * qq
	if ppqq > 0 {
		return 2.0 - 2.0*pq/assembler.Sqrt(ppqq)
	}
	return 2.0
}

func (a Angular) side(n Node, y []float32, random Random) int {
	dot := a.margin(n, y)
	if dot != 0.0 {
		if dot > 0 {
			return 1
		} else {
			return 0
		}
	}
	return random.flip()
}

func (a Angular) margin(n Node, y []float32) float32 {
	var dot float32
	for z, v := range n.v {
		dot += v * y[z]
	}
	return dot
}
