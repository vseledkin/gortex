package gortex

import (
	"fmt"
	"math"
	"math/rand"
)

type Matrix struct {
	Rows    int //number of rows
	Columns int // number of columns
	W       []float32
	DW      []float32 `json:"omit"`
}

func (m *Matrix) SameAs() (mm *Matrix) {
	mm = Mat(m.Rows, m.Columns)
	return
}

func (m *Matrix) ZerosAs() (mm *Matrix) {
	mm = Mat(m.Rows, m.Columns)
	return
}

func (m *Matrix) OnesAs() (mm *Matrix) {
	mm = Mat(m.Rows, m.Columns)
	for i := range mm.W {
		mm.W[i] = 1.0
	}
	return
}

func (m *Matrix) Get(row, col int) float32 {
	// slow but careful accessor function
	// we want row-major order
	ix := (m.Columns * row) + col
	if ix >= 0 && ix < len(m.W) {
		return m.W[ix]
	} else {
		panic(fmt.Errorf("mat element access error index %Columns out of range", ix))
	}
}

func (m *Matrix) Set(row, col int, v float32) {
	ix := (m.Columns * row) + col
	if ix >= 0 && ix < len(m.W) {
		m.W[ix] = v
	} else {
		panic(fmt.Errorf("mat element access error index %Columns out of range", ix))
	}
}

func (m *Matrix) GetGradient(row, col int) float32 {
	// slow but careful accessor function
	// we want row-major order
	ix := (m.Columns * row) + col
	if ix >= 0 && ix < len(m.DW) {
		return m.DW[ix]
	} else {
		panic(fmt.Errorf("mat element access error index %Columns out of range", ix))
	}
}

func (m *Matrix) SetGradient(row, col int, v float32) {
	ix := (m.Columns * row) + col
	if ix >= 0 && ix < len(m.DW) {
		m.DW[ix] = v
	} else {
		panic(fmt.Errorf("mat element access error index %Columns out of range", ix))
	}
}

func (m *Matrix) AddGradient(row, col int, v float32) {
	ix := (m.Columns * row) + col
	if ix >= 0 && ix < len(m.DW) {
		m.DW[ix] += v
	} else {
		panic(fmt.Errorf("mat element access error index %Columns out of range", ix))
	}
}

func (m *Matrix) NormGradient() float32 {
	var val float32
	for i := range m.DW {
		if m.DW[i] < 0 {
			val -= m.DW[i]
		} else {
			val += m.DW[i]
		}
	}
	return val / float32(len(m.DW))
}

func (m *Matrix) Norm() float32 {
	var val float32
	for i := range m.W {
		if m.W[i] < 0 {
			val -= m.W[i]
		} else {
			val += m.W[i]
		}
	}
	return val / float32(len(m.W))
}

func MatFromSlice(w [][]float32) *Matrix {
	M := new(Matrix)
	M.Rows = len(w)
	M.Columns = len(w[0])
	for _, row := range w {
		M.W = append(M.W, row...)
	}
	return M
}

func Mat(rows, columns int) *Matrix {
	M := new(Matrix)
	M.Rows = rows
	M.Columns = columns
	M.W = Zeros(rows * columns)
	M.DW = Zeros(rows * columns)
	return M
}

//RandMat uses standard gaussian initialization scheme
func RandMat(rows, columns int) *Matrix {
	M := Mat(rows, columns)
	for i := range M.W {
		M.W[i] = float32(rand.NormFloat64()) // standard normal distribution (mean = 0, stddev = 1)
	}
	return M
}

//RandXavierMat uses Xavier 2 / ( fanin + fanout ) initialization scheme
func RandXavierMat(rows, columns int) *Matrix {
	variance := 2.0 / float64(columns+rows)
	M := Mat(rows, columns)
	for i := range M.W {
		M.W[i] = float32(rand.NormFloat64() * math.Sqrt(variance)) // standard normal distribution (mean = 0, stddev = 1)
	}
	return M
}
