package models

import (
	"github.com/vseledkin/gortex"
	"os"
	"encoding/json"
	"github.com/vseledkin/gortex/assembler"
)

type Seq2seq struct {
	FirstLookupTable  *gortex.Matrix
	SecondLookupTable *gortex.Matrix
	EmbeddingSize     int
	HiddenSize        int
	EncoderOutputSize int
	LearningRate      float32
	Dic               *gortex.BiDictionary
	Encoder           *gortex.MultiplicativeLSTM
	Decoder           *gortex.MultiplicativeLSTM
	parameters        map[string]*gortex.Matrix
	Z                 *gortex.Matrix   `json:"-"`
	EncoderOutputs    []*gortex.Matrix `json:"-"`
	EBos, EEos        uint
	DEos, DBos        uint
	EncoderInitialH   *gortex.Matrix
	EncoderInitialC   *gortex.Matrix
	DecoderInitialH   *gortex.Matrix
}

func (tc *Seq2seq) GetParameters() map[string]*gortex.Matrix {
	if tc.parameters == nil {
		// model parameters
		tc.parameters = tc.Encoder.GetParameters("Encoder")
		for k, v := range tc.Decoder.GetParameters("Decoder") {
			tc.parameters[k] = v
		}
		tc.parameters["FirstLookupTable"] = tc.FirstLookupTable
		tc.parameters["SecondLookupTable"] = tc.SecondLookupTable
		tc.parameters["EncoderInitialH"] = tc.EncoderInitialH
		tc.parameters["EncoderInitialC"] = tc.EncoderInitialC
		tc.parameters["DecoderInitialH"] = tc.DecoderInitialH
	}
	return tc.parameters
}

func (tc Seq2seq) Create() *Seq2seq {
	tc.EBos = tc.Dic.First.IDByToken(gortex.BOS)
	tc.EEos = tc.Dic.First.IDByToken(gortex.EOS)

	tc.DBos = tc.Dic.Second.IDByToken(gortex.BOS)
	tc.DEos = tc.Dic.Second.IDByToken(gortex.EOS)

	tc.EncoderInitialH = gortex.RandMat(tc.HiddenSize, 1)
	tc.EncoderInitialC = gortex.RandMat(tc.HiddenSize, 1)
	tc.DecoderInitialH = gortex.RandMat(tc.HiddenSize, 1)
	tc.FirstLookupTable = gortex.RandMat(tc.EmbeddingSize, tc.Dic.First.Len())   // Lookup Table matrix
	tc.SecondLookupTable = gortex.RandMat(tc.EmbeddingSize, tc.Dic.Second.Len()) // Lookup Table matrix
	tc.Encoder = gortex.MakeMultiplicativeLSTM(tc.EmbeddingSize, tc.HiddenSize, tc.EncoderOutputSize)
	tc.Encoder.ForgetGateTrick(2.0)

	tc.Decoder = gortex.MakeMultiplicativeLSTM(tc.EmbeddingSize, tc.HiddenSize, tc.Dic.Second.Len())
	tc.Decoder.ForgetGateTrick(2.0)

	return &tc
}

func (tc *Seq2seq) Forward(G *gortex.Graph, input_sequence, target_sequence []uint) (string, float32) {
	he := tc.EncoderInitialH
	ce := tc.EncoderInitialC

	// reset previous memory
	//tc.EncoderOutputs = make([]*gortex.Matrix, len(input_sequence))
	input_sequence = append([]uint{tc.EBos}, append(input_sequence, tc.EEos)...)
	for i := len(input_sequence) - 1; i >= 0; i-- {
		he, ce, _ = tc.Encoder.Step(G, G.Lookup(tc.FirstLookupTable, int(input_sequence[i])), he, ce)
	}
	tc.Z = he

	prev := G.Lookup(tc.SecondLookupTable, int(tc.DBos))

	decoded := ""
	var cost float32
	// predict EOS after sequence decoding has finished
	target_sequence = append(target_sequence, uint(tc.DEos))
	cd := tc.Z
	hd := tc.DecoderInitialH
	for i := range target_sequence {
		var y *gortex.Matrix
		// make attention vector
		hd, cd, y = tc.Decoder.Step(G, prev, hd, cd)
		// predict at every time step
		predictedTokenId, _ := gortex.MaxIV(gortex.Softmax(y))
		decoded += tc.Dic.Second.TokenByID(predictedTokenId) + ""
		nll, _ := G.Crossentropy(y, target_sequence[i])
		cost += nll
		prev = G.Lookup(tc.SecondLookupTable, int(predictedTokenId))
	}

	return decoded, cost / float32(len(target_sequence))
}

func (tc *Seq2seq) Load(modelFile string) error {
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

func (tc *Seq2seq) Save(modelFile string) error {
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
func (tc *Seq2seq) Clone() *Seq2seq {
	clone := Seq2seq{LearningRate: tc.LearningRate, EmbeddingSize: tc.EmbeddingSize, HiddenSize: tc.HiddenSize, EncoderOutputSize: tc.EncoderOutputSize, Dic: tc.Dic}.Create()
	parameters := tc.GetParameters()
	cloneParameters := clone.GetParameters()
	// share parameters
	for k, v := range parameters {
		cloneParameters[k].W = v.W
	}

	return clone
}

func (tc *Seq2seq) CollectGradient(clone *Seq2seq) {
	parameters := tc.GetParameters()
	cloneParameters := clone.GetParameters()
	// share parameters
	for k, v := range parameters {
		assembler.Sxpy(cloneParameters[k].DW, v.DW)
		assembler.Sclean(cloneParameters[k].DW)
	}
}
