package annoy

import (
	"os"
	"testing"
)

func TestGannoyIndexNotFound(t *testing.T) {
	name := "test_gannoy_index_not_found.tree"
	defer os.Remove(name)
	_, err := NewAnnoyIndex("not_found.meta", Angular{}, RandRandom{})
	if err == nil {
		t.Errorf("NewAnnoyIndex with not exist meta file should return error.")
	}

}

func TestGannoyIndexAttribute(t *testing.T) {
	tree := 2
	dim := 3
	K := 4
	name := "test_gannoy_index_attribute"
	CreateMeta(".", name, tree, dim, K)
	defer os.Remove(name + ".meta")

	treeFile := name + ".tree"
	defer os.Remove(treeFile)
	gannoy, _ := NewAnnoyIndex(name+".meta", Angular{}, RandRandom{})

	if gannoy.tree != tree {
		t.Errorf("NewAnnoyIndex should contain tree %d, but %d", tree, gannoy.tree)
	}
	if gannoy.dim != dim {
		t.Errorf("NewAnnoyIndex should contain dim %d, but %d", dim, gannoy.dim)
	}
	if gannoy.K != K {
		t.Errorf("NewAnnoyIndex should contain K %d, but %d", K, gannoy.K)
	}
}
func TestGannoyIndexAddItemAsRoot(t *testing.T) {
	tree := 2
	name := "test_gannoy_index_add_item_as_root"
	CreateMeta(".", name, tree, 3, 4)
	defer os.Remove(name + ".meta")

	treeFile := name + ".tree"
	defer os.Remove(treeFile)
	gannoy, _ := NewAnnoyIndex(name+".meta", Angular{}, RandRandom{})

	// first item (be root)
	err := gannoy.AddItem(10, []float32{1.1, 1.2, 1.3})
	if err != nil {
		t.Errorf("AnnoyIndex AddItem should not return error.")
	}
	node, _ := gannoy.nodes.getNodeByKey(10)
	for i := 0; i < tree; i++ {
		if !node.isRoot(i) {
			t.Errorf("AnnoyIndex AddItem at first should build root node.")
		}
	}

	// second item (change root and make tree)
	err = gannoy.AddItem(20, []float32{1.1, 1.2, 1.3})
	if err != nil {
		t.Errorf("AnnoyIndex AddItem should not return error.")
	}
	node, _ = gannoy.nodes.getNodeByKey(20)
	for i := 0; i < tree; i++ {
		if node.isRoot(i) {
			t.Errorf("AnnoyIndex AddItem at second should not build root node.")
		}

		if parent, _ := gannoy.nodes.getNode(node.parents[i]); !parent.isRoot(i) {
			t.Errorf("AnnoyIndex AddItem at second should build root child node.")
		}
	}
}

func TestGannoyIndexAddItemToLeafNode(t *testing.T) {
	tree := 2
	name := "test_gannoy_index_add_item_to_leaf_node"
	CreateMeta(".", name, tree, 3, 3)
	defer os.Remove(name + ".meta")

	treeFile := name + ".tree"
	defer os.Remove(treeFile)
	gannoy, _ := NewAnnoyIndex(name+".meta", Angular{}, &TestLoopRandom{max: 1})

	// add item to leaf node
	items := [][]float32{
		{1.1, 1.2, 1.3},
		{-1.1, -1.2, -1.3},
		{-1.1, -1.2, -1.3},
		{-1.1, -1.2, -1.3},
	}
	for i, item := range items {
		gannoy.AddItem(i*10, item)
	}

	// Current tree
	// 7 [-1] (-1) [nDescendants: 4, v: [0.5280168968110516 0.576018432884782 0.6240199689585159]]
	//   8 [-1] (7) [nDescendants: 3, v: []]
	//     1 [10] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     4 [20] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     5 [30] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//   0 [0] (7) [nDescendants: 1, v: [1.1 1.2 1.3]]

	err := gannoy.AddItem(40, []float32{1.1, 1.2, 1.3})
	if err != nil {
		t.Errorf("AnnoyIndex AddItem should not return error.")
	}

	// Expect tree (build new bucket node that contain node 0[0] and new node 3[40].)
	// 6 [-1] (-1) [nDescendants: 5, v: [0.5280168968110516 0.576018432884782 0.6240199689585159]]
	//   9 [-1] (6) [nDescendants: 3, v: []]
	//     1 [10] (9) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     4 [20] (9) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     5 [30] (9) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//   10 [-1] (6) [nDescendants: 2, v: []]
	//     0 [0] (10) [nDescendants: 1, v: [1.1 1.2 1.3]]
	//     3 [40] (10) [nDescendants: 1, v: [1.1 1.2 1.3]]

	node, _ := gannoy.nodes.getNodeByKey(40)
	for i := 0; i < tree; i++ {
		parent, _ := gannoy.nodes.getNode(node.parents[i])
		if len(parent.children) != 2 {
			t.Errorf("AnnoyIndex AddItem to leaf node should return node that contain 2 children.")
		}
		for _, child := range parent.children {
			if child != node.id && child != 0 {
				t.Errorf("AnnoyIndex AddItem to leaf node should return node that contain 0[0] and 3[40].")
			}
		}
	}
}

func TestGannoyIndexAddItemToBucketNode(t *testing.T) {
	tree := 2
	name := "test_gannoy_index_add_item_to_bucket_node"
	CreateMeta(".", name, tree, 3, 3)
	defer os.Remove(name + ".meta")

	treeFile := name + ".tree"
	defer os.Remove(treeFile)
	gannoy, _ := NewAnnoyIndex(name+".meta", Angular{}, &TestLoopRandom{max: 1})

	// add item to bucket node
	items := [][]float32{
		{1.1, 1.2, 1.3},
		{-1.1, -1.2, -1.3},
		{-1.1, -1.2, -1.3},
		{-1.1, -1.2, -1.3},
		{1.1, 1.2, 1.3},
	}
	for i, item := range items {
		gannoy.AddItem(i*10, item)
	}

	// Current tree
	// 7 [-1] (-1) [nDescendants: 5, v: [0.5280168968110516 0.576018432884782 0.6240199689585159]]
	//   8 [-1] (7) [nDescendants: 3, v: []]
	//     1 [10] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     4 [20] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     5 [30] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//   2 [-1] (7) [nDescendants: 2, v: []]
	//     0 [0] (2) [nDescendants: 1, v: [1.1 1.2 1.3]]
	//     3 [40] (2) [nDescendants: 1, v: [1.1 1.2 1.3]]

	err := gannoy.AddItem(50, []float32{1.1, 1.2, 1.3})
	if err != nil {
		t.Errorf("AnnoyIndex AddItem should not return error.")
	}

	// Expect tree
	// 7 [-1] (-1) [nDescendants: 5, v: [0.5280168968110516 0.576018432884782 0.6240199689585159]]
	//   8 [-1] (7) [nDescendants: 3, v: []]
	//     1 [10] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     4 [20] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     5 [30] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//   2 [-1] (7) [nDescendants: 3, v: []]
	//     0 [0] (2) [nDescendants: 1, v: [1.1 1.2 1.3]]
	//     3 [40] (2) [nDescendants: 1, v: [1.1 1.2 1.3]]
	//     11 [50] (2) [nDescendants: 1, v: [1.1 1.2 1.3]]

	node, _ := gannoy.nodes.getNodeByKey(50)
	for i := 0; i < tree; i++ {
		parent, _ := gannoy.nodes.getNode(node.parents[i])
		if len(parent.children) != 3 {
			t.Errorf("AnnoyIndex AddItem to leaf node should return node that contain 3 children.")
		}
	}

	// add item in full bucket node
	err = gannoy.AddItem(60, []float32{1.1, 1.2, 1.3})
	if err != nil {
		t.Errorf("AnnoyIndex AddItem should not return error.")
	}

	// Expect tree (build new branch node that contain two branch nodes)
	// 7 [-1] (-1) [nDescendants: 6, v: [0.5280168968110516 0.576018432884782 0.6240199689585159]]
	//   8 [-1] (7) [nDescendants: 3, v: []]
	//     1 [10] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     4 [20] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     5 [30] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//   13 [-1] (7) [nDescendants: 4, v: [0 0 0]]
	//     16 [-1] (13) [nDescendants: 2, v: []]
	//       2 [40] (16) [nDescendants: 1, v: [1.1 1.2 1.3]]
	//       12 [60] (16) [nDescendants: 1, v: [1.1 1.2 1.3]]
	//     18 [-1] (13) [nDescendants: 2, v: []]
	//       0 [0] (18) [nDescendants: 1, v: [1.1 1.2 1.3]]
	//       11 [50] (18) [nDescendants: 1, v: [1.1 1.2 1.3]]

	node, _ = gannoy.nodes.getNodeByKey(60)
	for i := 0; i < tree; i++ {
		parent, _ := gannoy.nodes.getNode(node.parents[i])
		grandParent, _ := gannoy.nodes.getNode(parent.parents[i])
		if grandParent.nDescendants != 4 || grandParent.isLeaf() || grandParent.isBucket() {
			t.Errorf("AnnoyIndex AddItem to full branch node should return branch node that has 4 nDescendants.")
		}
	}
}

func TestGannoyIndexRemoveItem(t *testing.T) {
	tree := 2
	name := "test_gannoy_index_remove_item"
	CreateMeta(".", name, tree, 3, 3)
	defer os.Remove(name + ".meta")

	treeFile := name + ".tree"
	defer os.Remove(treeFile)
	gannoy, _ := NewAnnoyIndex(name+".meta", Angular{}, &TestLoopRandom{max: 1})

	// remove from bucket node
	items := [][]float32{
		{1.1, 1.2, 1.3},
		{-1.1, -1.2, -1.3},
		{-1.1, -1.2, -1.3},
		{-1.1, -1.2, -1.3},
		{1.1, 1.2, 1.3},
		{1.1, 1.2, 1.3},
	}
	for i, item := range items {
		gannoy.AddItem(i*10, item)
	}

	// Current tree
	// 6 [-1] (-1) [nDescendants: 5, v: [0.5280168968110516 0.576018432884782 0.6240199689585159]]
	//   8 [-1] (6) [nDescendants: 3, v: []]
	//     1 [10] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     4 [20] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     5 [30] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//   2 [-1] (6) [nDescendants: 3, v: []]
	//     0 [0] (2) [nDescendants: 1, v: [1.1 1.2 1.3]]
	//     3 [40] (2) [nDescendants: 1, v: [1.1 1.2 1.3]]
	//     11 [50] (2) [nDescendants: 1, v: [1.1 1.2 1.3]]

	removed, _ := gannoy.nodes.getNodeByKey(50)
	removedId := removed.id
	parents := removed.parents

	err := gannoy.removeItem(50)
	if err != nil {
		t.Errorf("AnnoyIndex RemoveItem should not return error.")
	}

	// Expect tree (remove specified node from parent bucket node)
	// 6 [-1] (-1) [nDescendants: 5, v: [0.5280168968110516 0.576018432884782 0.6240199689585159]]
	//   8 [-1] (6) [nDescendants: 3, v: []]
	//     1 [10] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     4 [20] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     5 [30] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//   2 [-1] (6) [nDescendants: 2, v: []]
	//     0 [0] (2) [nDescendants: 1, v: [1.1 1.2 1.3]]
	//     3 [40] (2) [nDescendants: 1, v: [1.1 1.2 1.3]]

	for i := 0; i < tree; i++ {
		parent, _ := gannoy.nodes.getNode(parents[i])
		if len(parent.children) != 2 {
			t.Errorf("AnnoyIndex RemoveItem should return parent node that contain 2 children.")
		}
		for _, child := range parent.children {
			if child == removedId {
				t.Errorf("AnnoyIndex RemoveItem should not return removeItem.")
			}
		}
	}

	removed, _ = gannoy.nodes.getNodeByKey(40)
	removedId = removed.id
	parents = removed.parents
	grandParents := make([]int, tree)
	for i, p := range parents {
		parent, _ := gannoy.nodes.getNode(p)
		grandParents[i] = parent.parents[i]
	}

	err = gannoy.removeItem(40)
	if err != nil {
		t.Errorf("AnnoyIndex RemoveItem should not return error.")
	}

	// Expect tree (remove specified node and parent node, and be leaf node that remaining one.)
	// 6 [-1] (-1) [nDescendants: 4, v: [0.5280168968110516 0.576018432884782 0.6240199689585159]]
	//   8 [-1] (6) [nDescendants: 3, v: []]
	//     1 [10] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     4 [20] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     5 [30] (8) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//   0 [0] (6) [nDescendants: 1, v: [1.1 1.2 1.3]]

	for i := 0; i < tree; i++ {
		grandParent, _ := gannoy.nodes.getNode(grandParents[i])
		if grandParent.nDescendants != 4 {
			t.Errorf("AnnoyIndex RemoveItem should return grand parent node that has 4 nDescendants.")
		}
		for _, p := range grandParent.children {
			parent, _ := gannoy.nodes.getNode(p)
			if parent.isLeaf() {
				if parent.id == removedId || parent.id == parents[i] {
					t.Errorf("AnnoyIndex RemoveItem should be leaf node that remainin node.")
				}
			}
		}
	}
}

func TestGannoyIndexUpdateItem(t *testing.T) {
	tree := 2
	name := "test_gannoy_index_update_item"
	CreateMeta(".", name, tree, 3, 4)
	defer os.Remove(name + ".meta")

	treeFile := name + ".tree"
	defer os.Remove(treeFile)
	gannoy, _ := NewAnnoyIndex(name+".meta", Angular{}, &TestLoopRandom{max: 1})

	// remove from bucket node
	items := [][]float32{
		{1.1, 1.2, 1.3},
		{-1.1, -1.2, -1.3},
		{1.1, 1.2, 1.3},
		{-1.1, -1.2, -1.3},
		{-1.1, -1.2, -1.3},
	}
	for i, item := range items {
		gannoy.AddItem(i*10, item)
	}

	updated, _ := gannoy.nodes.getNodeByKey(30)
	updatedId := updated.id
	updatedParents := updated.parents

	// Current tree
	// 7 [-1] (-1) [nDescendants: 5, v: [0.5280168968110516 0.576018432884782 0.6240199689585159]]
	//   11 [-1] (7) [nDescendants: 3, v: []]
	//     1 [10] (11) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     5 [30] (11) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     6 [40] (11) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//   9 [-1] (7) [nDescendants: 2, v: []]
	//     0 [0] (9) [nDescendants: 1, v: [1.1 1.2 1.3]]
	//     4 [20] (9) [nDescendants: 1, v: [1.1 1.2 1.3]]

	err := gannoy.UpdateItem(30, []float32{1.1, 1.2, 1.3})
	if err != nil {
		t.Errorf("AnnoyIndex UpdateItem should not return error.")
	}

	// Expect tree (move to new bucket node)
	// 7 [-1] (-1) [nDescendants: 5, v: [0.5280168968110516 0.576018432884782 0.6240199689585159]]
	//   11 [-1] (7) [nDescendants: 2, v: []]
	//     1 [10] (11) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//     6 [40] (11) [nDescendants: 1, v: [-1.1 -1.2 -1.3]]
	//   9 [-1] (7) [nDescendants: 3, v: []]
	//     0 [0] (9) [nDescendants: 1, v: [1.1 1.2 1.3]]
	//     4 [20] (9) [nDescendants: 1, v: [1.1 1.2 1.3]]
	//     5 [30] (9) [nDescendants: 1, v: [1.1 1.2 1.3]]

	for i := 0; i < tree; i++ {
		updatedParent, _ := gannoy.nodes.getNode(updatedParents[i])
		for _, c := range updatedParent.children {
			if c == updatedId {
				t.Errorf("AnnoyIndex UpdateItem should move to new bucket node.")
			}
		}
		grandParent, _ := gannoy.nodes.getNode(updatedParent.parents[i])
		for _, p := range grandParent.children {
			if p != updatedParent.id {
				newParent, _ := gannoy.nodes.getNode(p)
				contain := false
				for _, child := range newParent.children {
					if child == updatedId {
						contain = true
					}
				}
				if !contain {
					t.Errorf("2 AnnoyIndex UpdateItem should move to new bucket node.")
				}
			}
		}
	}
}

func TestGannoyIndexGetNnsByKey(t *testing.T) {
	// search nns (from builded tree file)
	tree := 2
	name := "test_gannoy_index_get_nns_by_key"
	CreateMeta(".", name, tree, 3, 4)
	defer os.Remove(name + ".meta")

	treeFile := name + ".tree"
	defer os.Remove(treeFile)
	gannoy, _ := NewAnnoyIndex(name+".meta", Angular{}, &TestLoopRandom{max: 1})

	// remove from bucket node
	items := [][]float32{
		{1.1, 1.2, 1.3},
		{-1.1, -1.2, -1.3},
		{1.1, 1.2, 1.3},
		{-1.1, -1.2, -1.3},
		{-1.1, -1.2, -1.3},
	}
	for i, item := range items {
		gannoy.AddItem(i*10, item)
	}

	size := 3
	// Not found key
	nns, err := gannoy.GetNnsByKey(100, size, -1)
	if len(nns) != 0 {
		t.Errorf("AnnoyIndex GetNnsByKey should return empty list if key is not found.")
	}
	if err == nil {
		t.Errorf("AnnoyIndex GetNnsByKey should return error if key is not found.")
	}

	// Exist key
	nns, err = gannoy.GetNnsByKey(40, size, -1)
	if len(nns) != size {
		t.Errorf("AnnoyIndex GetNnsByKey should return specified size list if key exist.")

	}
	if err != nil {
		t.Errorf("AnnoyIndex GetNnsByKey should not return error if key exist.")
	}
}
