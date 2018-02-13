package models

import (
	"encoding/json"
	"fmt"
	g "github.com/vseledkin/gortex"
	"github.com/vseledkin/gortex/assembler"
	"log"
	"math"
	"os"
)

type PyramidClassifier struct {
	Height         int // defines the input capacity of the model
	Levels         []*g.Matrix
	Biases         []*g.Matrix
	EmbeddingSize  int
	HiddenSize     int
	Embed          func(text string) []float32 `json:"-"`
	Parameters     map[string]*g.Matrix        `json:"-"`
	Representation *g.Matrix                   `json:"-"`
	Bos            *g.Matrix // begin sequence symbol
	Eos, HEos      *g.Matrix // end sequence symbol also used as 2^Height padding
	Who            *g.Matrix
	Gate           *g.Matrix
	Logits         *g.Matrix                   `json:"-"`
	Name           string
	Updates        map[string]float32          `json:"-"`
	Classes        map[string]*struct{ Index, Count uint }
	forward_count  uint
}

func (p *PyramidClassifier) GetActiveParameters() map[string]*g.Matrix {
	active_model := make(map[string]*g.Matrix)
	for name, scale := range p.Updates {
		if scale > 0 {
			assembler.Sscale(1/scale, p.Parameters[name].DW)
			p.Updates[name] = 0
			active_model[name] = p.Parameters[name]
		}
	}
	return active_model
}

func (tc *PyramidClassifier) Load(modelFile string) error {
	f, e := os.Open(modelFile)
	if e != nil {
		return e
	}
	defer f.Close()

	decoder := json.NewDecoder(f)
	err := decoder.Decode(tc)
	if err != nil {
		return err
	}
	return nil
}

func (tc *PyramidClassifier) Save(modelFile string) error {
	f, err := os.Create(modelFile)
	if err != nil {
		return err
	}
	defer f.Close()
	encoder := json.NewEncoder(f)
	err = encoder.Encode(tc)
	if err != nil {
		return err
	}
	return nil
}

func (p *PyramidClassifier) GetClasses() map[string]*struct{ Index, Count uint } {
	return p.Classes
}

func (tc *PyramidClassifier) GetClassName(c uint) string {
	for name, class := range tc.Classes {
		if class.Index == c {
			return name
		}
	}
	panic(fmt.Errorf("no class with id %d exists", c))
}

func (p *PyramidClassifier) GetLogits() *g.Matrix {
	return p.Logits
}

func (p *PyramidClassifier) GetRepresentation() *g.Matrix {
	return p.Representation
}

func (p *PyramidClassifier) GetParameters() map[string]*g.Matrix {
	return p.Parameters
}

func (p *PyramidClassifier) GetName() string {
	return fmt.Sprintf("pyramid.ft.l%d.e%d.h%d.c%d.json", p.Height, p.EmbeddingSize, p.HiddenSize, len(p.Classes))
}

func (p PyramidClassifier) Create() (*PyramidClassifier, error) {
	// check height is valid
	if p.Height < 1 {
		return nil, fmt.Errorf("pyramid height must be > 0 but %d was given", p.Height)
	}
	// create pyramid levels
	p.Levels = make([]*g.Matrix, p.Height)
	p.Biases = make([]*g.Matrix, p.Height)

	p.Updates = make(map[string]float32)
	for l := range make([]struct{}, p.Height) {
		if l == 0 { // input layer
			p.Levels[l] = g.RandXavierMat(p.HiddenSize, 2*p.EmbeddingSize)
		} else { // hidden layer
			p.Levels[l] = g.RandXavierMat(p.HiddenSize, 2*p.HiddenSize)
		}
		p.Biases[l] = g.RandXavierMat(p.HiddenSize, 1)
	}
	p.Gate = g.RandXavierMat(1, p.HiddenSize)
	p.Who = g.RandXavierMat(len(p.Classes), p.HiddenSize)
	p.Bos = g.RandXavierMat(p.EmbeddingSize, 1)
	p.Eos = g.RandXavierMat(p.EmbeddingSize, 1)

	p.HEos = g.RandXavierMat(p.HiddenSize, 1)

	p.Parameters = map[string]*g.Matrix{
		"Bos":  p.Bos,
		"Eos":  p.Eos,
		"HEos": p.HEos,
		"Who":  p.Who,
		"Gate": p.Gate,
	}
	for l := range p.Levels {
		p.Parameters[fmt.Sprintf("L%dw", l)] = p.Levels[l]
		p.Parameters[fmt.Sprintf("L%db", l)] = p.Biases[l]
	}

	return &p, nil
}

func (p *PyramidClassifier) pad_input(s []*g.Matrix, tokens []string) ([]*g.Matrix, []string) {
	r := []*g.Matrix{p.Bos}
	t := []string{"bos"}
	t = append(t, tokens...)
	t = append(t, "eos")
	p.Updates["Bos"]++
	r = append(r, s...)
	r = append(r, p.Eos)
	p.Updates["Eos"]++
	if len(r)%2 != 0 {
		r = append(r, p.Eos)
		p.Updates["Eos"]++
		t = append(t, "eos")
	}
	return r, t
}

func (p *PyramidClassifier) pad_inner(s []*g.Matrix) []*g.Matrix {
	if len(s)%2 != 0 {
		return append(s, p.HEos)
		p.Updates["HEos"]++
	}
	return s
}

func (p *PyramidClassifier) Forward(G *g.Graph, tokens []string) (class uint, confidence float32) {
	p.forward_count++
	embedings := make([]*g.Matrix, len(tokens))
	// embed tokens
	for t, token := range tokens {
		embedings[t] = g.Mat(p.EmbeddingSize, 1)
		embedings[t].W = p.Embed(token)
	}
	var paddedTokens []string
	// propagate layer by layer

	var story []string
	for l := range p.Levels {
		//log.Printf("level %d inputs %d\n", l, len(embedings))
		if l == 0 {
			embedings, paddedTokens = p.pad_input(embedings, tokens)
		} else {
			embedings = p.pad_inner(embedings)
		}

		//log.Printf("level %d padded inputs %d\n", l, L)
		var newEmbedings []*g.Matrix
		for t := range embedings {
			conv := G.Tanh(G.Add(G.Mul(p.Levels[l], G.Concat(embedings[2*t], embedings[2*t+1])), p.Biases[l]))
			gate := G.Sigmoid(G.Mul(p.Gate, conv))

			gate = G.ReplicateScalar(gate, conv.Numel())
			p.Updates[fmt.Sprintf("L%dw", l)]++
			p.Updates[fmt.Sprintf("L%db", l)]++
			p.Updates["Gate"]++
			newEmbedings = append(newEmbedings, G.EMul(gate, conv))
			if p.forward_count%1000 == 0 && l == 0 {
				story = append(story, fmt.Sprintf("%d %s -> %f", len(story), paddedTokens[2*t]+" "+paddedTokens[2*t+1], gate.W[0]))
			}

			if 2*t+1 == len(embedings)-1 {
				break
			}
		}
		embedings = newEmbedings
		if len(embedings) == 1 {
			p.Logits = G.Mul(p.Who, embedings[0])
			p.Updates["Who"]++
			break
		}
	}

	if len(embedings) != 1 {
		log.Fatalf("pyramid with height %d can fit sequence of len %d but got %d at the top on sequence of len %d", p.Height, int(math.Pow(2.0, float64(p.Height))), len(embedings), len(tokens))
	}
	class, confidence = g.MaxIV(g.Softmax(p.Logits))
	if len(story) > 0 && p.GetClassName(class) == "__label__Positive" {
		log.Println(class, confidence, p.GetClassName(class))
		for i := range story {
			log.Println(story[i])
		}
	}
	return
}
