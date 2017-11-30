package models

import (
	"fmt"
	"github.com/vseledkin/gortex"
	"log"
	"math"
)

type PyramidClassifier struct {
	Height         int // defines the input capacity of the model
	Levels         []*gortex.Matrix
	Biases         []*gortex.Matrix
	EmbeddingSize  int
	HiddenSize     int
	OutputSize     int
	Embed          func(text string) []float32
	Parameters     map[string]*gortex.Matrix
	Representation *gortex.Matrix
	Bos            *gortex.Matrix // begin sequence symbol
	Eos, HEos      *gortex.Matrix // end sequence symbol also used as 2^Height padding
	Whos           []*gortex.Matrix
	Logits         *gortex.Matrix
	Name           string
	Updates        map[string]float32
}

func (p *PyramidClassifier) ComposeName() string {
	return fmt.Sprintf("pyramidWL.ft.h%d.o%d.e%d.json", p.EmbeddingSize, p.HiddenSize, p.OutputSize)
}

func (p PyramidClassifier) Create() (*PyramidClassifier, error) {
	// check height is valid
	if p.Height < 1 {
		return nil, fmt.Errorf("pyramid height must be > 0 but %d was given", p.Height)
	}
	// create pyramid levels
	p.Levels = make([]*gortex.Matrix, p.Height)
	p.Biases = make([]*gortex.Matrix, p.Height)
	p.Whos = make([]*gortex.Matrix, p.Height)
	p.Updates = make(map[string]float32)
	for l := range make([]struct{}, p.Height) {
		if l == 0 { // input layer
			p.Levels[l] = gortex.RandXavierMat(p.HiddenSize, 2*p.EmbeddingSize)
		} else { // hidden layer
			p.Levels[l] = gortex.RandXavierMat(p.HiddenSize, 2*p.HiddenSize)
		}
		p.Biases[l] = gortex.RandXavierMat(p.HiddenSize, 1)
		p.Whos[l] = gortex.RandXavierMat(p.OutputSize, p.HiddenSize)
	}
	p.Bos = gortex.RandXavierMat(p.EmbeddingSize, 1)
	p.Eos = gortex.RandXavierMat(p.EmbeddingSize, 1)

	p.HEos = gortex.RandXavierMat(p.HiddenSize, 1)

	p.Parameters = map[string]*gortex.Matrix{
		"Bos":  p.Bos,
		"Eos":  p.Eos,
		"HEos": p.HEos,
	}
	for l := range p.Levels {
		p.Parameters[fmt.Sprintf("L%dw", l)] = p.Levels[l]
		p.Parameters[fmt.Sprintf("L%db", l)] = p.Biases[l]
		p.Parameters[fmt.Sprintf("L%dWho", l)] = p.Whos[l]
	}

	return &p, nil
}

func (p *PyramidClassifier) pad_input(s []*gortex.Matrix) []*gortex.Matrix {
	r := []*gortex.Matrix{p.Bos}
	p.Updates["Bos"]++
	r = append(r, s...)
	r = append(r, p.Eos)
	p.Updates["Eos"]++
	if len(r)%2 != 0 {
		r = append(r, p.Eos)
		p.Updates["Eos"]++
	}
	return r
}
func (p *PyramidClassifier) pad_inner(s []*gortex.Matrix) []*gortex.Matrix {
	if len(s)%2 != 0 {
		return append(s, p.HEos)
		p.Updates["HEos"]++
	}
	return s
}

func (p *PyramidClassifier) Forward(G *gortex.Graph, tokens []string) (class uint, confidence float32) {
	embedings := make([]*gortex.Matrix, len(tokens))
	// embed tokens
	for t, token := range tokens {
		embedings[t] = gortex.Mat(p.EmbeddingSize, 1)
		embedings[t].W = p.Embed(token)
	}

	// propagate layer by layer
	for l := range p.Levels {
		//log.Printf("level %d inputs %d\n", l, len(embedings))
		if l == 0 {
			embedings = p.pad_input(embedings)
		} else {
			embedings = p.pad_inner(embedings)
		}
		L := len(embedings)

		//log.Printf("level %d padded inputs %d\n", l, L)
		for t := range embedings {
			embedings[t] = G.Relu(G.Add(G.Mul(p.Levels[l], G.Concat(embedings[2*t], embedings[2*t+1])), p.Biases[l]))
			p.Updates[fmt.Sprintf("L%dw", l)]++
			p.Updates[fmt.Sprintf("L%db", l)]++
			if 2*t+1 == len(embedings)-1 {
				embedings = embedings[:L/2]
				break
			}
		}
		if len(embedings) == 1 {
			p.Logits = G.Mul(p.Whos[l], embedings[0])
			p.Updates[fmt.Sprintf("L%dWho", l)]++
			break
		}
	}
	if len(embedings) != 1 {
		log.Fatalf("pyramid with height %d can fit sequence of len %d but got %d at the top on sequence of len %d", p.Height, int(math.Pow(2.0, float64(p.Height))), len(embedings), len(tokens))
	}
	class, confidence = gortex.MaxIV(gortex.Softmax(p.Logits))
	return
}

func (p *PyramidClassifier) Load(modelFile string) error {
	var params map[string]*gortex.Matrix
	var e error
	params, e = gortex.LoadModel(modelFile)
	if e != nil {
		return e
	}
	for k, v := range p.Parameters {
		if m, ok := params[k]; ok {
			copy(v.W, m.W)
		} else {
			return fmt.Errorf("Model geometry is not compatible, parameter %s is unknown", k)
		}
	}
	return nil
}
