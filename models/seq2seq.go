package models

import (
	"github.com/vseledkin/gortex"
	"fmt"
)

type Seq2seq struct {
	InLookupTable, OutLookupTable *gortex.Matrix
	EmbeddingSize                 int
	HiddenSize                    int
	EncoderOutputSize             int
	Window                        int
	Dic                           *gortex.BiDictionary
	Parameters                    map[string]*gortex.Matrix
	fencoder, bencoder            *gortex.OutputlessGRU
	decoder                       *gortex.GRU
	Aw, Ab                        *gortex.Matrix
	Attention                     *gortex.Matrix
	EncoderOutputs                []*gortex.Matrix
	pad                           []*gortex.Matrix
	bos, eos                      int
}

func (tc Seq2seq) Create() *Seq2seq {
	tc.bos = int(tc.Dic.Second.IDByToken(gortex.BOS))
	tc.eos = int(tc.Dic.Second.IDByToken(gortex.EOS))
	//for i := 0; i < (tc.Window-1)/2; i++ {
	//	tc.pad = append(tc.pad, gortex.Mat(2*tc.EncoderOutputSize, 1))
	//}
	tc.InLookupTable = gortex.RandMat(tc.EmbeddingSize, tc.Dic.First.Len())   // Lookup Table matrix
	tc.OutLookupTable = gortex.RandMat(tc.EmbeddingSize, tc.Dic.Second.Len()) // Lookup Table matrix
	tc.fencoder = gortex.MakeOutputlessGRU(tc.EmbeddingSize, tc.HiddenSize)
	tc.fencoder.ForgetGateTrick(2.0)
	//tc.bencoder = gortex.MakeGRU(tc.EmbeddingSize, tc.HiddenSize, tc.EncoderOutputSize)
	//tc.bencoder.ForgetGateTrick(2.0)
	tc.decoder = gortex.MakeGRU(tc.EmbeddingSize, tc.HiddenSize, tc.Dic.Second.Len())
	tc.decoder.ForgetGateTrick(2.0)

	//tc.Aw = gortex.RandXavierMat(tc.Window, tc.Dic.Second.Len()+tc.Window)
	//tc.Ab = gortex.RandXavierMat(tc.Window, 1)

	// model parameters
	tc.Parameters = tc.fencoder.GetParameters("FEncoder")
	//for k, v := range tc.bencoder.GetParameters("BEncoder") {
	//	tc.Parameters[k] = v
	//}
	for k, v := range tc.decoder.GetParameters("Decoder") {
		tc.Parameters[k] = v
	}
	//tc.Parameters["Aw"] = tc.Aw
	//tc.Parameters["Ab"] = tc.Ab

	return &tc
}

func (tc *Seq2seq) Forward(G *gortex.Graph, input_sequence, target_sequence []uint) (string, float32) {
	ht := gortex.Mat(tc.HiddenSize, 1) // vector of zeros

	var y *gortex.Matrix
	// reset previous memory
	//tc.EncoderOutputs = make([]*gortex.Matrix, len(input_sequence))

	//for i, token := range input_sequence {
	for i := len(input_sequence) - 1; i >= 0; i-- {
		ht = tc.fencoder.Step(G, G.Lookup(tc.InLookupTable, int(input_sequence[i])), ht)
	}

	//ht = gortex.Mat(tc.HiddenSize, 1).OnesAs()
	//for i := len(input_sequence) - 1; i >= 0; i-- {
	//	ht, y = tc.bencoder.Step(G, G.Lookup(tc.LookupTable, int(input_sequence[i])), ht)
	//	tc.EncoderOutputs[i] = G.Tanh(G.Concat(tc.EncoderOutputs[i], y))
	//}
	//var sequence []*gortex.Matrix
	// padleft with zeros
	//sequence = append(sequence, tc.pad...)
	//sequence = append(sequence, tc.EncoderOutputs...)
	// padright with zeros
	//sequence = append(sequence, tc.pad...)
	//tc.EncoderOutputs = sequence

	//ht = gortex.Mat(tc.HiddenSize, 1).OnesAs()
	prev := G.Lookup(tc.OutLookupTable, tc.bos)
	//A := gortex.Mat(tc.Window, 1).OnesAs()
	//context := gortex.Mat(2*tc.EncoderOutputSize, 1)

	decoded := ""
	var cost float32
	// predict EOS after sequence decoding has finished
	target_sequence = append(target_sequence, uint(tc.eos))
	for i := range target_sequence {
		// make attention vector
		//A = G.Add(G.Mul(tc.Aw, G.Concat(prev, A)), tc.Ab)
		//println(i, gortex.MinInt(i, len(tc.EncoderOutputs)-tc.Window), gortex.MinInt(i+tc.Window, len(tc.EncoderOutputs)))
		//receptiveField := tc.EncoderOutputs[gortex.MinInt(i, len(tc.EncoderOutputs)-tc.Window):gortex.MinInt(i+tc.Window, len(tc.EncoderOutputs))]
		//tc.Attention = G.Softmax(A)
		//context := G.Attention(tc.EncoderOutputs, tc.Attention)

		ht, y = tc.decoder.Step(G, prev, ht)
		// predict at every time step
		predictedTokenId, _ := gortex.MaxIV(gortex.Softmax(y))
		decoded += tc.Dic.Second.TokenByID(predictedTokenId)
		prev = G.Lookup(tc.OutLookupTable, int(predictedTokenId))
		nll, _ := G.Crossentropy(y, target_sequence[i])
		cost += nll
	}

	return decoded, cost / float32(len(target_sequence))
}

func (tc *Seq2seq) Load(modelFile string) error {
	var params map[string]*gortex.Matrix
	var e error
	params, e = gortex.LoadModel(modelFile)
	if e != nil {
		return e
	}
	for k, v := range tc.Parameters {
		if m, ok := params[k]; ok {
			copy(v.W, m.W)
		} else {
			return fmt.Errorf("Model geometry is not compatible, parameter %s is unknown", k)
		}
	}
	return nil
}
