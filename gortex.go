package gortex

import "math"
import (
	"encoding/json"
	"fmt"
	"github.com/vseledkin/gortex/assembler"

	"os"
)

func Zeros(n int) []float32 {
	return make([]float32, n)
}

func ScaleGradient(model map[string]*Matrix, v float32) {
	for _, m := range model {
		assembler.Sscale(v, m.DW)
	}
}

//Softmax probability distribution interpretation of any vector/matrix
func Softmax(m *Matrix) *Matrix {
	out := Mat(m.Rows, m.Columns) // probability volume
	var maxval float32 = -math.MaxFloat32
	l := len(m.W)
	for i := 0; i < l; i++ {
		if m.W[i] > maxval {
			maxval = m.W[i]
		}
	}

	for i := 0; i < l; i++ {
		out.W[i] = float32(math.Exp(float64(m.W[i] - maxval)))
	}
	sum := assembler.L1(out.W)
	assembler.Sscale(1/sum, out.W)

	// no backward pass here needed
	// since we will use the computed probabilities outside
	// to set gradients directly on m
	return out
}

func Moments(m *Matrix) (mean, variance float32) {
	mean = assembler.L1(m.W) / float32(len(m.W))

	var total float32
	var tmp float32
	for i := range m.W {
		tmp = m.W[i] - mean
		total += tmp * tmp
	}
	variance = total / float32(len(m.W))
	return
}

func MaxIV(m *Matrix) (int, float32) {
	var max float32 = -math.MaxFloat32
	maxIndex := -1
	for i, v := range m.W {
		if v > max {
			max = v
			maxIndex = i
		}
	}
	return maxIndex, max
}

func SaveModel(name string, m map[string]*Matrix) {
	fmt.Print("\n---------------------------------------------------\n")
	fmt.Printf("Saving model to: %s\n", name)
	fmt.Print("---------------------------------------------------\n")
	// save MODEL_NAME
	f, err := os.Create(name)
	if err != nil {
		panic(err)
	}
	encoder := json.NewEncoder(f)
	encoder.Encode(m)
	f.Close()
}

func LoadModel(name string) map[string]*Matrix {

	if len(name) == 0 {
		panic(fmt.Errorf("No model file provided! [%s]", name))
	}
	fmt.Printf("Loading learned model %s\n", name)
	f, e := os.Open(name)
	if e != nil {
		panic(e)
	}
	var m map[string]*Matrix
	decoder := json.NewDecoder(f)
	decoder.Decode(&m)

	f.Close()
	return m
}
