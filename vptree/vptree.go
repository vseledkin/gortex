package vptree

import (
	"math"
	"math/rand"
)

func Euclidean(x, y []float32) float32 {
	var s, tmp float32
	for i := range x {
		tmp = x[i] - y[i]
		s += tmp * tmp
	}
	return float32(math.Sqrt(float64(s)))
}

type Item struct {
	ID     int
	Vector []float32
}

//Node VPTree node
type Node struct {
	Item      *Item
	Threshold float32
	Left      *Node
	Right     *Node
	Size      int
}

// A VPTree struct represents a Vantage-point tree. Vantage-point trees are
// useful for nearest-neighbour searches in high-dimensional metric spaces.
type VPTree struct {
	ID2node        map[int]*Node
	root           *Node
	distanceMetric func(x, y []float32) float32
}

//MetricCalls increases every time metric of two vectors evaluated
var MetricCalls int

// NewVPTree creates a new VP-tree using the metric and items provided. The metric
// measures the distance between two items, so that the VP-tree can find the
// nearest neighbour(s) of a target item.
func NewVPTree(metric func(x, y []float32) float32, items []*Item) (t *VPTree) {
	// make copy of items to not damage original data
	t = &VPTree{
		ID2node:        make(map[int]*Node, len(items)),
		distanceMetric: metric,
	}
	t.root = t.build(items)
	return
}

func (vp *VPTree) build(items []*Item) (n *Node) {
	if len(items) == 0 {
		return nil
	}

	n = &Node{}
	// Take a random item out of the items slice and make it this node's item
	idx := rand.Intn(len(items))
	n.Item = items[idx]
	n.Size = len(items)
	vp.ID2node[n.Item.ID] = n
	// put last element instead of item
	// remove slice length by 1
	items[idx], items = items[len(items)-1], items[:len(items)-1]

	if len(items) > 0 {
		// Now partition the items into two equal-sized sets, one
		// closer to the node's item than the median, and one farther
		// away.
		median := len(items) / 2
		MetricCalls++
		// distance to random median item
		pivotDist := vp.distanceMetric(items[median].Vector, n.Item.Vector)

		// put median item to the end of slice and
		// end item replaces previous median
		items[median], items[len(items)-1] = items[len(items)-1], items[median]

		storeIndex := 0
		// go thought all items excluding median and now excluding item itself
		for i := 0; i < len(items)-1; i++ {
			MetricCalls++
			if vp.distanceMetric(items[i].Vector, n.Item.Vector) <= pivotDist {
				// if some item closer than median to the item itself
				// then put this item to the starting part of a clice
				// and item at storeindex (farer than median) instead of item
				items[storeIndex], items[i] = items[i], items[storeIndex]
				storeIndex++
			}
		}
		// swap median item (which is at the end of slice) and item at the end of closer items list
		items[len(items)-1], items[storeIndex] = items[storeIndex], items[len(items)-1]
		// so now median is at storeIndex position of a slice
		median = storeIndex
		MetricCalls++
		// we can reuse threshold
		n.Threshold = pivotDist

		n.Left = vp.build(items[:median])
		n.Right = vp.build(items[median:])
	}
	return
}

// Search searches the VP-tree for the k nearest neighbours of target. It
// returns the up to k narest neighbours and the corresponding distances in
// order of least distance to largest distance.
func (vp *VPTree) Search(target *Item, k int, cutoff float32) (results []*Item, distances []float32) {
	if k < 1 {
		return
	}

	h := make(priorityQueue, 0, k)
	var tau float32 = math.MaxFloat32
	if cutoff > 0 {
		tau = cutoff
	}
	// we search k + 1 because we will exclude item itself from search result
	vp.search(vp.root, target, k+1, &h, &tau)

	for len(h) > 0 {
		hi := h.Pop()
		results = append(results, hi.Item)
		distances = append(distances, hi.Dist)
	}

	// Reverse results and distances, because we popped them from the heap
	// in large-to-small order
	for i, j := 0, len(results)-1; i < j; i, j = i+1, j-1 {
		results[i], results[j] = results[j], results[i]
		distances[i], distances[j] = distances[j], distances[i]
	}

	return
}

func (vp *VPTree) search(n *Node, target *Item, k int, h *priorityQueue, tau *float32) {
	var d float32
	if n.Item.ID != target.ID {
		MetricCalls++
		d = vp.distanceMetric(n.Item.Vector, target.Vector)
		if d < *tau {
			if len(*h) == k {
				h.Pop()
			}
			h.Push(&heapItem{n.Item, d})

			if len(*h) == k {
				*tau = h.Top().Dist
			}
		}
	} else {
		d = 0
	}

	if d < n.Threshold {
		if d - *tau <= n.Threshold && n.Left != nil {
			vp.search(n.Left, target, k, h, tau)
		}

		if d + *tau >= n.Threshold && n.Right != nil {
			vp.search(n.Right, target, k, h, tau)
		}
	} else {
		if d + *tau >= n.Threshold && n.Right != nil {
			vp.search(n.Right, target, k, h, tau)
		}

		if d - *tau <= n.Threshold && n.Left != nil {
			vp.search(n.Left, target, k, h, tau)
		}
	}
}
