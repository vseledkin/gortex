package gortex

import (
	"fmt"
	"math/rand"
)

type Matrix struct {
	n  int //number of rows
	d  int // number of columns
	W  []float64
	dw []float64
}

func (m *Matrix) Get(row, col int) float64 {
	// slow but careful accessor function
	// we want row-major order
	ix := (m.d * row) + col
	if ix >= 0 && ix < len(m.W) {
		return m.W[ix]
	} else {
		panic(fmt.Errorf("mat element access error index %d out of range", ix))
	}
}

func (m *Matrix) Set(row, col int, v float64) {
	ix := (m.d * row) + col
	if ix >= 0 && ix < len(m.W) {
		m.W[ix] = v
	} else {
		panic(fmt.Errorf("mat element access error index %d out of range", ix))
	}
}

func MatFromSlice(w [][]float64) *Matrix {
	M := new(Matrix)
	M.n = len(w)
	M.d = len(w[0])
	for _, row := range w {
		M.W = append(M.W, row...)
	}
	return M
}

func Mat(rows, columns int) *Matrix {
	M := new(Matrix)
	M.n = rows
	M.d = columns
	M.W = Zeros(rows * columns)
	M.dw = Zeros(rows * columns)
	return M
}

func RandMat(rows, columns int) *Matrix {
	M := Mat(rows, columns)
	for i := range M.W {
		M.W[i] = rand.NormFloat64() // standard normal distribution (mean = 0, stddev = 1)
	}
	return M
}
