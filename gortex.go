package gortex

import "math"
import (
	"encoding/json"
	"fmt"

	"github.com/vseledkin/gortex/assembler"

	"math/rand"
	"os"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func Abs(f float32) float32 {
	if f < 0 {
		return -f
	}
	return f
}
func Max(x, y float32) float32 {
	if x < y {
		return y
	}
	return x
}
func MaxInt(x, y int) int {
	if x < y {
		return y
	}
	return x
}
func MinInt(x, y int) int {
	if x > y {
		return y
	}
	return x
}

func Pow(x, y float32) float32 {
	return float32(math.Pow(float64(x), float64(y)))
}

func Zeros(n int) []float32 {
	return make([]float32, n)
}

func ScaleGradient(model map[string]*Matrix, v float32) {
	for _, m := range model {
		assembler.Sscale(v, m.DW)
	}
}

func PrintGradient(model map[string]*Matrix) {
	for k, m := range model {
		fmt.Printf("Grad: %s %f %f\n", k, m.Norm(), m.NormGradient())
	}
}

func PrintZeroGradient(model map[string]*Matrix) {
	for k, m := range model {
		wnorm := m.Norm()
		gnorm := m.NormGradient()
		if wnorm == 0 {
			fmt.Printf("\033[93mWarning!!! Weights of %s = %f\033[0m\n", k, wnorm)
		}
		if gnorm == 0 {
			fmt.Printf("\033[93mWarning!!! Gradients of %s = %f\033[0m\n", k, gnorm)
		}
	}
}

func InitWeights(model map[string]*Matrix, dev float32) {
	for _, m := range model {
		for i := range m.W {
			m.W[i] = dev * float32(rand.NormFloat64()) // standard normal distribution (mean = 0, stddev = 1)
		}
	}
}

func ResetGradients(model map[string]*Matrix) {
	for _, m := range model {
		assembler.Sclean(m.DW)
	}
}

//Softmax probability distribution interpretation of any vector/matrix
func Softmax(m *Matrix) *Matrix {
	out := Mat(m.Rows, m.Columns) // probability volume
	maxval := m.W[assembler.Ismax(m.W)]
	for i := range m.W {
		out.W[i] = float32(math.Exp(float64(m.W[i] - maxval)))
	}
	sum := assembler.Sum(out.W)
	assembler.Sscale(1/sum, out.W)

	// no backward pass here needed
	// since we will use the computed probabilities outside
	// to set gradients directly on m
	return out
}

// Take elementwise exponent of x
func Exp(x *Matrix) *Matrix {
	out := Mat(x.Rows, x.Columns)
	for i := range x.W {
		out.W[i] = float32(math.Exp(float64(x.W[i])))
	}

	return out
}
func Moments(m *Matrix) (mean, variance float32) {
	mean = assembler.Sum(m.W) / float32(len(m.W))

	var total float32
	var tmp float32
	for i := range m.W {
		tmp = m.W[i] - mean
		total += tmp * tmp
	}
	variance = total / float32(len(m.W))
	return
}

func MaxIV(m *Matrix) (maxIndex uint, max float32) {
	max = -math.MaxFloat32
	for i, v := range m.W {
		if v > max {
			max = v
			maxIndex = uint(i)
		}
	}
	return
}

func Multinomial(probabilities *Matrix) uint {
	if probabilities.Columns != 1 {
		panic(fmt.Errorf("Input must be vector"))
	}

	offset := float32(0)
	sample := rand.Float32()
	for i, p := range probabilities.W {
		offset += p
		//sample uniform from [0,1]
		if sample <= offset {
			return uint(i)
		}
	}
	return uint(len(probabilities.W) - 1)
}

func SaveModel(name string, m map[string]*Matrix) error {
	fmt.Print("\n---------------------------------------------------\n")
	fmt.Printf("Saving model to: %s\n", name)
	fmt.Print("---------------------------------------------------\n")
	// save MODEL_NAME
	f, err := os.Create(name)
	if err != nil {
		return err
	}
	defer f.Close()
	encoder := json.NewEncoder(f)
	err = encoder.Encode(m)
	if err != nil {
		return err
	}
	return nil
}

func LoadModel(name string) (map[string]*Matrix, error) {
	if len(name) == 0 {
		return nil, fmt.Errorf("No model file provided! [%s]", name)
	}
	//fmt.Printf("Loading learned model %s\n", name)
	f, e := os.Open(name)
	if e != nil {
		return nil, e
	}
	defer f.Close()
	var m map[string]*Matrix
	decoder := json.NewDecoder(f)
	err := decoder.Decode(&m)
	if err != nil {
		return nil, err
	}
	return m, nil
}

func F1Score(trueLabels, predictedLabels []uint, str []string, excludes map[uint]bool) (float64, string) {
	if len(trueLabels) != len(predictedLabels) {
		panic(fmt.Errorf("Number of true labels %d and predicted ones %d must match", len(trueLabels), len(predictedLabels)))
	}
	f := make(map[uint]*struct {
		tp, fp, fn, p, r, f, c float64
	})

	for i := range trueLabels {
		m, ok := f[trueLabels[i]]
		if !ok {
			m = new(struct {
				tp, fp, fn, p, r, f, c float64
			})
			f[trueLabels[i]] = m
		}
		m.c++
		if trueLabels[i] > uint(len(str)-1) {
			panic(fmt.Errorf("No name for true label %d given", trueLabels[i]))
		}
	}
	// do the same for predicted labels
	for i := range predictedLabels {
		m, ok := f[predictedLabels[i]]
		if !ok {
			m = new(struct {
				tp, fp, fn, p, r, f, c float64
			})
			f[predictedLabels[i]] = m
		}
		if predictedLabels[i] > uint(len(str)-1) {
			panic(fmt.Errorf("No name for predicted label %d given", predictedLabels[i]))
		}
	}
	var maxlabel uint

	for i := range trueLabels {
		x := trueLabels[i]
		y := predictedLabels[i]
		if x == y {
			f[x].tp++
		} else {
			f[x].fn++
			f[y].fp++
		}

		if maxlabel < x {
			maxlabel = x
		}

		if maxlabel < y {
			maxlabel = y
		}

	}
	var F float64

	ff := make([]*struct {
		tp, fp, fn, p, r, f, c float64
	}, maxlabel+1)

	for l, m := range f {
		ff[l] = m
	}
	var denominator float64
	var message string
	for l, m := range ff {
		if m != nil && !excludes[uint(l)] {
			m.p = m.tp / (m.tp + m.fp)
			if math.IsNaN(m.p) {
				m.p = 0
			}
			m.r = m.tp / (m.tp + m.fn)

			m.f = 2.0 * m.p * m.r / (m.r + m.p)
			if math.IsNaN(m.f) {
				m.f = 0
			}
			message += fmt.Sprintf("label:%d-[%s] Count:%.1f tp:%.1f fn:%.1f fp:%.1f p:%f r:%f f:%f\n", l, str[l], m.c, m.tp, m.fn, m.fp, m.p, m.r, m.f)
			F += m.c * m.f
			denominator += float64(m.c)
		}
	}
	F /= denominator
	return F, message
}

func SetParameters(dest, parameters map[string]*Matrix) error {
	for k, v := range dest {
		if m, ok := parameters[k]; ok {
			copy(v.W, m.W)
		} else {
			return fmt.Errorf("Model geometry is not compatible, parameter %s is unknown", k)
		}
	}
	return nil
}

func Sigmoid(shift, x float32) float32 {
	return float32(1 / (1 + math.Exp(float64(shift-x))))
}
