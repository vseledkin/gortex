package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/vseledkin/gortex"
	"github.com/vseledkin/gortex/assembler"
	"github.com/vseledkin/gortex/models"
	"log"
	"math"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	train            = "train"
	trainAutoencoder = "train.autoencoder"
	translate        = "translate"
	epochs           = 100
)

var input, experiment string

func Sample(trainFile string, dic *gortex.BiDictionary, tokenizer gortex.Tokenizer) (func(batchSize int) (epoch, step int, source, target [][]uint), error) {
	f, e := os.Open(trainFile)
	if e != nil {
		return nil, e
	}
	// count number of samples
	r := bufio.NewReader(f)
	epoch := 0
	step := -1

	return func(batchSize int) (int, int, [][]uint, [][]uint) {
		var sources, targets [][]uint

		for len(sources) < batchSize {
			line, e := r.ReadString('\n')
			if e != nil {
				f.Seek(0, 0)
				line, e = r.ReadString('\n')
				if e != nil {
					panic(e)
				}
				epoch++
			}
			line = strings.TrimSpace(line)
			if len(line) > 0 {
				var pair []string
				if os.Args[1] == trainAutoencoder {
					pair = []string{line, line}
				} else {
					pair = strings.Split(line, "<->")
				}
				terms := tokenizer.Split(pair[0])

				source := make([]uint, len(terms))
				for i, term := range terms {
					source[i] = dic.First.IDByToken(term)
				}
				terms = tokenizer.Split(pair[1])

				target := make([]uint, len(terms))
				for i, term := range terms {
					target[i] = dic.Second.IDByToken(term)
				}
				sources = append(sources, source)
				targets = append(targets, target)
			} else {
				panic(fmt.Errorf("corpus is broken, empty line"))
			}
		}
		step++
		return epoch, step, sources, targets
	}, nil
}

func Train() {
	tokenizer := gortex.CharSplitter{}
	delimiter := ""

	modelFile := experiment + ".model.json"
	kernels := []int{128, 64, 64, 64}
	log.Printf("Dilated convolution:")
	for i := range kernels {
		log.Printf("Layer %d receptive field %d for %d kernels:", i, 2<<uint(i+1)-1, kernels[i])
	}
	model := &models.DilatedSeq2seq{}
	if s, err := os.Stat(modelFile); err == nil && !s.IsDir() {
		model.Load(modelFile)
		for _, p := range model.GetParameters() { // initialize gradients
			p.DW = make([]float32, len(p.W))
		}
	} else {
		var dic *gortex.BiDictionary
		if os.Args[1] == trainAutoencoder {
			sdic := gortex.NewDictionary()
			if err = sdic.FromFile(input, tokenizer); err != nil {
				panic(err)
			}
			log.Printf("Dictionary has %d tokens", sdic.Len())
			sdic.Print(25)
			dic = &gortex.BiDictionary{First: sdic, Second: sdic}
		} else {
			dic, err = gortex.BiDictionary{}.FromFile(input, gortex.CharSplitter{})
			if err != nil {
				panic(err)
			}
		}

		model = models.DilatedSeq2seq{Kernels: kernels, LearningRate: 0.01, EmbeddingSize: 64, HiddenSize: 128, Dic: dic}.Create()
	}
	log.Printf("First SubDictionary has %d tokens", model.Dic.First.Len())
	model.Dic.First.Print(25)
	log.Printf("Second SubDictionary has %d tokens", model.Dic.Second.Len())
	model.Dic.Second.Print(25)

	optimizer := gortex.NewOptimizer(gortex.OpOp{Method: gortex.POWERSIGN, LearningRate: model.LearningRate})

	train_set, err := Sample(input, model.Dic, tokenizer)
	if err != nil {
		panic(err)
	}
	batchSize := 1
	ma_cost := gortex.NewMovingAverage(128)
	ma_cost.Add(100)
	clones := make([]*models.DilatedSeq2seq, batchSize)
	for i := range clones {
		log.Printf("Clone S2S %d", i)
		clones[i] = model.Clone()
	}
	for true {
		epoch, step, source, target := train_set(batchSize)
		start := time.Now()
		numberOfTokens := 0
		var w sync.WaitGroup
		w.Add(len(source))
		decoded := make([]string, len(source))
		for i := range source {
			numberOfTokens += len(source[i])
			go func(i int, source, target []uint) {
				G := &gortex.Graph{NeedsBackprop: true}
				var c float32
				decoded[i], c = clones[i].Forward(G, source, target)
				ma_cost.Add(c)
				G.Backward()
				w.Done()
			}(i, source[i], target[i])
		}
		w.Wait()
		stop := time.Now()

		// update
		parameters := model.GetParameters()
		for i := range clones {
			cloneParameters := clones[i].GetParameters()
			// share parameters
			for k, v := range parameters {
				assembler.Sxpy(cloneParameters[k].DW, v.DW)
				assembler.Sclean(cloneParameters[k].DW)
			}
		}
		gortex.ScaleGradient(parameters, 1/float32(batchSize))
		optimizer.Step(parameters)
		parameters = model.GetParameters()
		for i := range clones {
			cloneParameters := clones[i].GetParameters()
			// share parameters
			for k, v := range parameters {
				cloneParameters[k].W = v.W
				//if &(cloneParameters[k].W) != &(v.W) {
				//	panic(k)
				//}
				//cloneParameters[k].W = v.W
			}
		}
		// update end
		theCost := ma_cost.Avg()
		if math.IsNaN(float64(theCost)) {
			panic(fmt.Errorf("Cost is: &f\n", theCost))
		}
		if step%5000 == 0 && step > 0 {

			model.LearningRate = optimizer.LearningRate
			if err = model.Save(modelFile); err != nil {
				panic(err)
			}
			optimizer.LearningRate *= 0.999
			log.Printf("Model saved %s", modelFile)
		}
		if step%100 == 0 && step > 0 {
			log.Printf("E: %d S: %d LR: %f Loss: %f Speed: %f\n", epoch, step, optimizer.LearningRate, theCost, float64(numberOfTokens)/stop.Sub(start).Seconds())
			log.Printf("Source: %s\n", model.Dic.First.Decode(source[0], delimiter))
			log.Printf("Target: %s\n", model.Dic.Second.Decode(target[0], delimiter))
			log.Printf("Decode: %s\n\n", decoded[0])
			if step%1000 == 0 && step > 0 {
				clones[0].Encoder.Print(1)
			}
			//log.Printf("Attention:\n")
			//for i := range model.Attention.W {
			//	log.Printf("\t%d %f\n", i, model.Attention.W[i])
			//}
			//optimizer.LearningRate *= 0.999
		}
	}
}

func sample(c, h *gortex.Matrix, dic *gortex.Dictionary, fast_net *gortex.MultiplicativeLSTM, LookupTable *gortex.Matrix, temperature float32, multinomial bool) string {

	G := gortex.Graph{NeedsBackprop: false}
	term_id := dic.IDByToken(" ")
	//space := dic.IDByToken(" ")
	point := dic.IDByToken(".")
	ask := dic.IDByToken("?")
	eks := dic.IDByToken("!")

	//slow_ct := gortex.RandMat(hidden_size, 1)
	ct := c
	ht := h

	//fast_ct := gortex.RandMat(hidden_size, 1)
	generated := ""
	var logits *gortex.Matrix

	// fast network
	for j := 0; j < 50; j++ { // max word len 32
		//		fast_ht, fast_ct, logits = fast_net.Step(&G, G.Lookup(LookupTable, term_id), fast_ht, fast_ct)
		ht, ct, logits = fast_net.Step(&G, G.Lookup(LookupTable, int(term_id)), ht, ct)
		probs := gortex.SoftmaxT(logits, temperature)
		if multinomial {
			term_id = gortex.Multinomial(probs)
		} else {
			term_id, _ = gortex.MaxIV(probs)
		}
		//vprint(probs.W, dic.TokenByID(term_id))
		generated += dic.TokenByID(term_id)

		if term_id == point || term_id == ask || term_id == eks {
			break
		}
	}

	return generated
}

func main() {
	trainCommand := flag.NewFlagSet(train, flag.ExitOnError)
	trainCommand.StringVar(&input, "i", "~/data/ru-en.txt", "bisequence file to train from")
	trainCommand.StringVar(&experiment, "e", "d2s", "name of the experiment")

	trainAutoencoderCommand := flag.NewFlagSet(train, flag.ExitOnError)
	trainAutoencoderCommand.StringVar(&input, "i", "input.txt", "file, - one sequence per line")
	trainAutoencoderCommand.StringVar(&experiment, "e", "s2s", "name of the experiment")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
		fmt.Fprint(os.Stderr, "sequence2sequence <command> arguments\n")
		fmt.Fprint(os.Stderr, "commands are:\n")

		fmt.Fprintf(os.Stderr, "%s\n", train)
		trainCommand.PrintDefaults()

		fmt.Fprintf(os.Stderr, "%s\n", trainAutoencoder)
		trainAutoencoderCommand.PrintDefaults()

		flag.PrintDefaults()
	}
	flag.Parse()
	log.SetOutput(os.Stderr)

	if len(os.Args) == 1 {
		flag.Usage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case train:
		trainCommand.Parse(os.Args[2:])
	case trainAutoencoder:
		trainAutoencoderCommand.Parse(os.Args[2:])
	default:
		log.Printf("%q is not valid command.\n", os.Args[1])
		os.Exit(1)
	}

	// BUILD COMMAND ISSUED
	if trainCommand.Parsed() {
		if input == "" || experiment == "" {
			trainCommand.PrintDefaults()
			return
		}
		// train new model
		Train()
		return
	}
	// BUILD COMMAND ISSUED
	if trainAutoencoderCommand.Parsed() {
		if input == "" || experiment == "" {
			trainAutoencoderCommand.PrintDefaults()
			return
		}
		// train new autoencoder model
		Train()
		return
	}
}
