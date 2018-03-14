package gortex

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/vseledkin/gortex/assembler"
)

type Matrix struct {
	Rows    int //number of rows
	Columns int // number of columns
	W       []float32
	DW      []float32 `json:"-"`
}

func (m *Matrix) SameAs() (mm *Matrix) {
	mm = Mat(m.Rows, m.Columns)
	return
}

func (m *Matrix) CopyAs() (mm *Matrix) {
	mm = Mat(m.Rows, m.Columns)
	copy(mm.W, m.W)
	return
}

func (m *Matrix) ZerosAs() (mm *Matrix) {
	mm = Mat(m.Rows, m.Columns)
	return
}

func (m *Matrix) OnesAs() (mm *Matrix) {
	mm = Mat(m.Rows, m.Columns)
	assembler.Sset(1.0, mm.W)
	return
}

func (m *Matrix) ConstantAs(c float32) (mm *Matrix) {
	mm = Mat(m.Rows, m.Columns)
	assembler.Sset(c, mm.W)
	return
}

func (m *Matrix) Get(row, col int) float32 {
	// slow but careful accessor function
	// we want row-major order
	ix := (m.Columns * row) + col
	if ix >= 0 && ix < len(m.W) {
		return m.W[ix]
	} else {
		panic(fmt.Errorf("mat element access error index %d out of range", ix))
	}
}

func (m *Matrix) Set(row, col int, v float32) {
	ix := (m.Columns * row) + col
	if ix >= 0 && ix < len(m.W) {
		m.W[ix] = v
	} else {
		panic(fmt.Errorf("mat element access error index %d out of range", ix))
	}
}

func (m *Matrix) GetGradient(row, col int) float32 {
	// slow but careful accessor function
	// we want row-major order
	ix := (m.Columns * row) + col
	if ix >= 0 && ix < len(m.DW) {
		return m.DW[ix]
	} else {
		panic(fmt.Errorf("mat element access error index %d out of range", ix))
	}
}

func (m *Matrix) SetGradient(row, col int, v float32) {
	ix := (m.Columns * row) + col
	if ix >= 0 && ix < len(m.DW) {
		m.DW[ix] = v
	} else {
		panic(fmt.Errorf("mat element access error index %d out of range", ix))
	}
}

func (m *Matrix) AddGradient(row, col int, v float32) {
	ix := (m.Columns * row) + col
	if ix >= 0 && ix < len(m.DW) {
		m.DW[ix] += v
	} else {
		panic(fmt.Errorf("mat element access error index %d out of range", ix))
	}
}

func (m *Matrix) NormGradient() float32 {
	return assembler.L1(m.DW) / float32(len(m.DW))
}

func (m *Matrix) Norm() float32 {
	return assembler.L1(m.W) / float32(len(m.W))
}

func (m *Matrix) Numel() int {
	return len(m.W)
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

//RandMatMD uses specified gaussian initialization scheme
func RandMatMD(rows, columns int, mean, dev float64) *Matrix {
	M := Mat(rows, columns)
	for i := range M.W {
		M.W[i] = float32(rand.NormFloat64()*dev + mean) // standard normal distribution (mean = 0, stddev = 1)
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

// The Layer-Sequential Unit Variance (LSUV) algorithm Mishkin & Matas, 2015 https://arxiv.org/abs/1511.06422
func LSUVMat(rows, columns int) *Matrix {
	if rows > 0 && columns > 0 {
		M := RandXavierMat(rows, columns)
		// make 1000 unit variance vectors for statistics test

		var activations []float32
		var variance float32 = 10
		var step float32 = 0.1
		hops := 0
		for {
			activations = activations[:0]
			for range make([]interface{}, 1000) {
				vector := RandMat(columns, 1)
				G := &Graph{NeedsBackprop: false}
				A := G.Mul(M, vector)
				activations = append(activations, A.W...)
			}
			V := Mat(len(activations), 1)
			V.W = activations

			_, variance = Moments(V)
			//log.Printf("%d Mean %f variance %f step %f", hops, mean, variance, step)
			hops++
			if variance > 1 {
				assembler.Sscale(1-step, M.W)
				step = step * 0.9
			} else {
				assembler.Sscale(1+step, M.W)
				step = step * 0.9
			}
			if hops > 1000 {
				break
			}
			if Abs(variance-1) < 1e-3 {
				break
			}
		}
		return M
	} else {
		panic("Layer-Sequential Unit Variance is only for matrixes")
	}
}
