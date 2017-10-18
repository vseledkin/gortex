package annoy

import (
	"container/heap"
)

type Queue struct {
	priority float32
	value    int
}

func (q *Queue) Less(other interface{}) bool {
	return q.priority > other.(*Queue).priority
}

type Interface interface {
	Less(other interface{}) bool
}

type pqsorter []Interface

// Implement heap.Interface: Push, Pop, Len, Less, Swap
func (s *pqsorter) Push(x interface{}) {
	*s = append(*s, x.(Interface))
}

func (s *pqsorter) Pop() interface{} {
	n := len(*s)
	if n > 0 {
		x := (*s)[n-1]
		*s = (*s)[0 : n-1]
		return x
	}
	return nil
}

func (s *pqsorter) Len() int {
	return len(*s)
}

func (s *pqsorter) Less(i, j int) bool {
	return (*s)[i].Less((*s)[j])
}

func (s *pqsorter) Swap(i, j int) {
	(*s)[i], (*s)[j] = (*s)[j], (*s)[i]
}

// Define priority queue struct
type PriorityQueue struct {
	s *pqsorter
}

func NewPQ() *PriorityQueue {
	q := &PriorityQueue{s: new(pqsorter)}
	heap.Init(q.s)
	return q
}

func (q *PriorityQueue) Push(x Interface) {
	heap.Push(q.s, x)
}

func (q *PriorityQueue) Pop() Interface {
	return heap.Pop(q.s).(Interface)
}

func (q *PriorityQueue) Top() Interface {
	if len(*q.s) > 0 {
		return (*q.s)[0].(Interface)
	}
	return nil
}

func (q *PriorityQueue) Fix(x Interface, i int) {
	(*q.s)[i] = x
	heap.Fix(q.s, i)
}

func (q *PriorityQueue) Remove(i int) Interface {
	return heap.Remove(q.s, i).(Interface)
}

func (q *PriorityQueue) Len() int {
	return q.s.Len()
}
