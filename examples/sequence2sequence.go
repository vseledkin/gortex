package main

import (
	"flag"
	"fmt"
	"os"
	"log"
	"github.com/vseledkin/gortex"
	"github.com/vseledkin/gortex/models"
	"bufio"
	"strings"
	"time"
	"math/rand"
)

const (
	train     = "train"
	translate = "translate"
	epochs    = 100
	window    = 32
)

var input, experiment string

func Sample(trainFile string, dic *gortex.BiDictionary, tokenizer gortex.Tokenizer) (func() (epoch, step int, source, target []uint), error) {
	f, e := os.Open(trainFile)
	if e != nil {
		return nil, e
	}
	// count number of samples
	r := bufio.NewReader(f)
	epoch := 0
	step := -1

	return func() (int, int, []uint, []uint) {
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
			pair := strings.Split(line, "<->")
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

			step++
			return epoch, step, source, target
		} else {
			panic(fmt.Errorf("corpus is broken, empty line"))
		}
	}, nil
}

func Train() {
	rand.Seed(time.Now().UnixNano())
	dicFile := experiment + ".dic.json"
	modelFile := experiment + ".model.json"
	var dic *gortex.BiDictionary
	if s, err := os.Stat(dicFile); err == nil && !s.IsDir() {
		dic, err = dic.Load(dicFile)
		if err != nil {
			panic(err)
		}
	} else {
		dic, err = gortex.BiDictionary{}.FromFile(input, gortex.CharSplitter{})
		if err != nil {
			panic(err)
		}
		dic.Top(64)
		dic.Second.Add(gortex.EOS)
		dic.Second.Add(gortex.BOS)
		dic.Save(dicFile)
	}
	log.Printf("First SubDictionary has %d tokens", dic.First.Len())
	log.Printf("Second SubDictionary has %d tokens", dic.Second.Len())
	optimizer := gortex.NewOptimizer(gortex.OpOp{Method: gortex.WINDOWGRAD, Momentum: gortex.DefaultMomentum, LearningRate: 0.001, Clip: 7})
	model := models.Seq2seq{EmbeddingSize: 256, HiddenSize: 256, EncoderOutputSize: 64, Window: window, Dic: dic}.Create()
	train_set, err := Sample(input, dic, gortex.CharSplitter{})
	if err != nil {
		panic(err)
	}
	batch_size := 4
	ma_cost := gortex.NewMovingAverage(64)
	for true {
		epoch, step, source, target := train_set()
		G := &gortex.Graph{NeedsBackprop: true}
		decoded, cost := model.Forward(G, source, target)
		ma_cost.Add(cost)
		G.Backward()
		var num_clip int
		if step%batch_size == 0 && step > 0 {
			//gortex.ScaleGradient(model.Parameters, 1/float32(batch_size))
			num_clip = optimizer.Step(model.Parameters).NumClipped
		}
		if step == 5000 {
			err = gortex.SaveModel(modelFile, model.Parameters)
			if err != nil {
				panic(err)
			}
			log.Printf("Model saved %s", modelFile)
		}
		if step%100 == 0 && step > 0 {
			log.Printf("E: %d S: %d LR: %f Loss: %f \n", epoch, step, optimizer.LearningRate, ma_cost.Avg())
			log.Printf("Source: %s\n", dic.First.Decode(source, ""))
			log.Printf("Target: %s\n", dic.Second.Decode(target, ""))
			log.Printf("Decode: %s\n", decoded)
			log.Printf("Clip: %d\n", num_clip)
			log.Printf("Attention:\n")
			//for i := range model.Attention.W {
			//	log.Printf("\t%d %f\n", i, model.Attention.W[i])
			//}
			optimizer.LearningRate *= 0.999
		}
	}
}

func main() {
	trainCommand := flag.NewFlagSet(train, flag.ExitOnError)
	trainCommand.StringVar(&input, "i", "~/data/ru-en.txt", "bisequence file to train from")
	trainCommand.StringVar(&experiment, "e", "s2s", "name of the experiment")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
		fmt.Fprint(os.Stderr, "sequence2sequence <command> arguments\n")
		fmt.Fprint(os.Stderr, "commands are:\n")

		fmt.Fprintf(os.Stderr, "%s\n", train)
		trainCommand.PrintDefaults()

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
}
