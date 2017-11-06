package vptree

type heapItem struct {
	Item *Item
	Dist float32
}

// A heap must be initialized before any of the heap operations
// can be used. Init is idempotent with respect to the heap invariants
// and may be called whenever the heap invariants may have been invalidated.
// Its complexity is O(n) where n = h.Len().
//
//func Init(h Interface) {
//	// heapify
//	n := h.Len()
//	for i := n/2 - 1; i >= 0; i-- {
//		down(h, i, n)
//	}
//}

// Push pushes the element x onto the heap. The complexity is
// O(log(n)) where n = h.Len().
//
func (pq *priorityQueue) Push(x *heapItem) {
	*pq = append(*pq, x)
	pq.up(len(*pq) - 1)
}

// Pop removes the minimum element (according to Less) from the heap
// and returns it. The complexity is O(log(n)) where n = h.Len().
// It is equivalent to Remove(h, 0).
//
func (pq *priorityQueue) Pop() *heapItem {
	n := len(*pq) - 1
	pq.Swap(0, n)
	pq.down(0, n)
	old := *pq
	n = len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

// Remove removes the element at index i from the heap.
// The complexity is O(log(n)) where n = h.Len().
//
func (pq *priorityQueue) Remove(i int) *heapItem {
	n := len(*pq) - 1
	if n != i {
		pq.Swap(i, n)
		pq.down(i, n)
		pq.up(i)
	}
	return pq.Pop()
}

// Fix re-establishes the heap ordering after the element at index i has changed its value.
// Changing the value of the element at index i and then calling Fix is equivalent to,
// but less expensive than, calling Remove(h, i) followed by a Push of the new value.
// The complexity is O(log(n)) where n = h.Len().
func (pq *priorityQueue) Fix(i int) {
	pq.down(i, len(*pq))
	pq.up(i)
}

func (pq *priorityQueue) up(j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || !pq.Less(j, i) {
			break
		}
		pq.Swap(i, j)
		j = i
	}
}

func (pq *priorityQueue) down(i, n int) {
	for {
		j1 := 2*i + 1
		if j1 >= n || j1 < 0 { // j1 < 0 after int overflow
			break
		}
		j := j1 // left child
		if j2 := j1 + 1; j2 < n && !pq.Less(j1, j2) {
			j = j2 // = 2*i + 2  // right child
		}
		if !pq.Less(j, i) {
			break
		}
		pq.Swap(i, j)
		i = j
	}
}

type priorityQueue []*heapItem

func (pq priorityQueue) Less(i, j int) bool {
	// We want a max-heap, so we use greater-than here
	return pq[i].Dist > pq[j].Dist
}

func (pq priorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq priorityQueue) Top() *heapItem {
	return pq[0]
}
