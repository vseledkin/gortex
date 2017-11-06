package gortex

import (
	"github.com/vseledkin/gortex/assembler"
	"github.com/vseledkin/gortex/vptree"
)

type KnnLoss struct {
	index    *vptree.VPTree
	centroid map[int][]float32
	ids      []int
	k        int
}

func CreateKnnLoss(k int, keys []int, vectors [][]float32) *KnnLoss {
	kl := &KnnLoss{k: k}
	items := make([]*vptree.Item, len(keys))
	dim := len(vectors[0])
	for i := range keys {
		item := &vptree.Item{ID: keys[i]}
		copy(item.Vector, vectors[i])
		items[i]=item
	}
	kl.index = vptree.NewVPTree(vptree.Euclidean, items)
	// compute centroid for group of k nearest neighbours
	kl.centroid = map[int][]float32{}

	for i, id := range keys {
		nearest, _ := kl.index.Search(&vptree.Item{id, vectors[i]}, kl.k+1, -1.)

		centroid := make([]float32, dim)
		nearest = nearest[1:]
		for i := range nearest { // except himself
			assembler.Sxpy(nearest[i].Vector, centroid)
		}

		assembler.Sscale(1.0/float32(len(nearest)), centroid)

		kl.centroid[id] = centroid
	}
	return kl
}

func (kl *KnnLoss) Loss(G *Graph, id int, vector *Matrix) *Matrix {
	centroid := kl.centroid[id]
	target := Mat(len(centroid), 1)
	target.W = centroid
	return G.MSE_t(vector, target)
}
