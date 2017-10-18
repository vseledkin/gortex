package gortex

import (
	"os"

	"github.com/vseledkin/gortex/annoy"
	"github.com/vseledkin/gortex/assembler"
)

type KnnLoss struct {
	index    annoy.AnnoyIndex
	centroid map[int][]float32
	ids      []int
	dim      int
	k        int
	name     string
}

func filexists(name string) bool {
	if _, err := os.Stat(name); err != nil {
		if os.IsNotExist(err) {
			return false
		}
	}
	return true
}

func CreateKnnLoss(k, tree, dim int, name string) (knn_loss *KnnLoss, e error) {
	if filexists(name + ".meta") {
		os.Remove(name + ".meta")
	}
	if filexists(name + ".tree") {
		os.Remove(name + ".tree")
	}

	knn_loss = &KnnLoss{dim: dim, k: k, name: name}
	if e = annoy.CreateMeta(".", name, tree, dim, 2*dim); e != nil {
		return nil, e
	}

	knn_loss.index, e = annoy.NewAnnoyIndex(name+".meta", annoy.Angular{}, annoy.RandRandom{})
	if e != nil {
		return nil, e
	}
	return knn_loss, nil
}

/*
func (kl *KnnLoss) Add(id int, v []float32) {
	copy := append([]float32{}, v...) // use copy to detach from orogonal data because it can change later
	kl.index.AddItem(id, copy)
	kl.ids = append(kl.ids, id)
}
*/

func (kl *KnnLoss) AddItems(keys []int, ws [][]float32) error {
	kl.ids = append([]int{}, keys...)
	vv := make([][]float32, len(ws))
	for i := range ws {
		vv[i] = append([]float32{}, ws[i]...) // use copy to detach from orogonal data because it can change later
	}
	return kl.index.AddItems(kl.ids, vv)
}

func (kl *KnnLoss) Build() (e error) {
	kl.index, e = annoy.NewAnnoyIndex(kl.name+".meta", annoy.Angular{}, annoy.RandRandom{})
	if e != nil {
		return e
	}
	// compute centroid for group of k nearest neighbours
	kl.centroid = map[int][]float32{}
	for _, id := range kl.ids {
		nearest, err := kl.index.GetNnsByKey(id, kl.k, -1)
		if err != nil {
			return err
		}

		centroid := make([]float32, kl.dim)

		for i := range nearest {
			if v, err := kl.index.GetVectorByKey(nearest[i]); err != nil {
				return err
			} else {
				assembler.Sxpy(v, centroid)
			}
		}

		assembler.Sscale(1.0/float32(len(nearest)), centroid)

		kl.centroid[id] = centroid
	}
	return nil
}

func (kl *KnnLoss) Loss(G *Graph, id int, vector *Matrix) *Matrix {
	centroid := kl.centroid[id]
	target := Mat(len(centroid), 1)
	target.W = centroid
	return G.MSE_t(vector, target)
}
