package annoy

func HeapSort(array []sorter, order, last int) {
	var heapifier heapifier
	switch order {
	case ASC:
		heapifier = heapifyFunc(downHeapify)
	default:
		heapifier = heapifyFunc(upHeapify)
	}

	heapSort(heapifier, array, last)
}

type heapifier interface {
	heapify([]sorter, int, int)
}

type heapifyFunc func([]sorter, int, int)

func (f heapifyFunc) heapify(array []sorter, root, length int) {
	f(array, root, length)
}

type sorter struct {
	id    int
	value float32
}

func heapSort(heapifier heapifier, array []sorter, last int) {
	// initialize
	for i := len(array) / 2; i >= 0; i-- {
		heapifier.heapify(array, i, len(array))
	}

	// remove top and do heapify
	bp := len(array) - last
	for length := len(array); length > 1; length-- {
		lastIndex := length - 1
		array[0], array[lastIndex] = array[lastIndex], array[0]
		heapifier.heapify(array, 0, lastIndex)
		if lastIndex == bp {
			break
		}
	}
}

func downHeapify(array []sorter, root, length int) {
	max := root
	l := (root * 2) + 1
	r := l + 1

	if l < length && array[l].value > array[max].value {
		max = l
	}

	if r < length && array[r].value > array[max].value {
		max = r
	}

	if max != root {
		array[root], array[max] = array[max], array[root]
		downHeapify(array, max, length)
	}
}

func upHeapify(array []sorter, root, length int) {
	min := root
	l := (root * 2) + 1
	r := l + 1

	if l < length && array[l].value < array[min].value {
		min = l
	}

	if r < length && array[r].value < array[min].value {
		min = r
	}

	if min != root {
		array[root], array[min] = array[min], array[root]
		upHeapify(array, min, length)
	}
}
