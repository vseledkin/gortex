package main

import (
	"fmt"
	"github.com/vseledkin/gortex"
	"github.com/vseledkin/gortex/assembler"
	"log"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"
)

const embedding_size = 128
const hidden_fast = 128
const hidden_slow = 128

//const trainFile = "64.unique.txt"
const trainFile = "input.txt"
const batch_size = 16
const L1 = 0.0

var learning_rate = float32(0.0001)

const anneal_rate = 0.9999
const Clip = 7.0

func main() {
	assembler.Init(true)
	// start from random
	rand.Seed(time.Now().UnixNano())

	//trainFile := "64.unique.txt"
	modelName := fmt.Sprintf("SlowFastLM.GRU%d-%d", hidden_fast, hidden_slow)
	fastModelName := modelName + "_fast"
	slowModelName := modelName + "_slow"
	dicName := "dic"
	dic, e := gortex.LoadDictionary(dicName + ".dic")
	if e != nil {
		dic, e = gortex.DictionaryFromFile(trainFile, gortex.CharSplitter{})
		if e != nil {
			log.Fatal(e)
		}
		e = gortex.SaveDictionary(dicName+".dic", dic)
		if e != nil {
			log.Fatal(e)
		}
	}

	fmt.Printf("Dictionary has %d tokens\n", dic.Len())
	fmt.Printf("%s\n", dic)

	optimizer := gortex.NewOptimizer(gortex.OpOp{Method: gortex.WINDOWGRAD, LearningRate: 0.01})

	//	fast_net := gortex.MakeGRU(embedding_size, hidden_size, dic.Len())
	//	slow_net := gortex.MakeGRU(hidden_size, hidden_size, hidden_size)
	fast_net := gortex.MakeGRU(embedding_size, hidden_fast, dic.Len())

	slow_net := gortex.MakeGRU(hidden_fast, hidden_slow, hidden_fast)

	h0_fast := gortex.Mat(hidden_fast, 1) // vector of zeros
	h0_slow := gortex.Mat(hidden_slow, 1) // vector of zeros
	// define model parameters
	var fast_model map[string]*gortex.Matrix
	var slow_model map[string]*gortex.Matrix
	LookupTable := gortex.RandMat(embedding_size, dic.Len()) // Lookup Table matrix
	if _, err := os.Stat(fastModelName); err == nil {
		weights, e := gortex.LoadModel(fastModelName)
		if e != nil {
			log.Fatal(e)
		}
		copy(LookupTable.W, weights["LookupTable"].W)
		// fast_model["LookupTable"]
		e = fast_net.SetParameters(fastModelName, weights)
		if e != nil {
			log.Fatal(e)
		}
		fast_model = fast_net.GetParameters(fastModelName)
		fast_model["LookupTable"] = LookupTable
	} else {
		fast_model = fast_net.GetParameters(fastModelName)
		fast_model["LookupTable"] = LookupTable
		gortex.InitWeights(fast_model, 0.05)

	}
	if _, err := os.Stat(slowModelName); err == nil {
		weights, e := gortex.LoadModel(slowModelName)
		if e != nil {
			log.Fatal(e)
		}
		e = slow_net.SetParameters(slowModelName, weights)
		if e != nil {
			log.Fatal(e)
		}
		slow_model = slow_net.GetParameters(slowModelName)
	} else {
		slow_model = slow_net.GetParameters(slowModelName)
		gortex.InitWeights(slow_model, 0.05)

	}
	count := 0
	ma_ppl := gortex.NewMovingAverage(1000)
	ma_nll := gortex.NewMovingAverage(1000)
	ma_bpc := gortex.NewMovingAverage(1000)
	ma_fast_speed := gortex.NewMovingAverage(1000)
	ma_slow_speed := gortex.NewMovingAverage(1000)

	ma_duration := gortex.NewMovingAverage(1000)

	threads := batch_size
	type result struct {
		cost       float32
		perplexity float32
		bpc        float32
		time       time.Duration
		fast_speed float32
		slow_speed float32
		fast_len   int
		slow_len   int
	}
	license := make(chan interface{}, threads)
	results := make(chan *result, batch_size*threads)
	for i := 0; i < threads; i++ {
		license <- 1
	}
	go func() {
		var update_count, slow_steps, fast_steps, samples_count int
		start := time.Now()
		// read channel in a nonblocking way
		for true {
			select {
			case r := <-results:
				//<-license
				update_count++
				samples_count++
				slow_steps += r.slow_len
				fast_steps += r.fast_len
				ma_bpc.Add(r.bpc)
				ma_ppl.Add(r.perplexity)
				ma_nll.Add(r.cost)
				ma_fast_speed.Add(r.fast_speed)
				ma_slow_speed.Add(r.slow_speed)
				ma_duration.Add(float32(r.time.Seconds()))
				if update_count == batch_size {
					//for i := 0; i < threads; i++ { // freeze train
					//	<-license
					//}
					ma_fast_speed.Add(float32(fast_steps) / float32(time.Now().Sub(start).Seconds()))
					ma_slow_speed.Add(float32(slow_steps) / float32(time.Now().Sub(start).Seconds()))
					update_count = 0
					//gortex.ScaleGradient(fast_model, 1/float32(fast_steps))
					//gortex.ScaleGradient(slow_model, 1/float32(slow_steps))
					fast_steps = 0
					slow_steps = 0
					if samples_count < 100 {
						optimizer.Step(fast_model)
						optimizer.Step(fast_model)
					} else {
						if samples_count%1000 == 0 {
							gortex.PrintGradient(fast_model)
							gortex.PrintGradient(slow_model)
						}
						optimizer.Step(fast_model)
						optimizer.Step(fast_model)
						fmt.Printf("step: %d nll: %0.2f ppl: %0.2f bpc: %0.2f lr: %f fspeed:%0.2f fspeed: %0.2f sspeed:%0.2f sspeed: %0.2f time:%0.2f s.\n", samples_count, ma_nll.Avg(), ma_ppl.Avg(), ma_bpc.Avg(), learning_rate, r.fast_speed, ma_fast_speed.Avg(), r.slow_speed, ma_slow_speed.Avg(), ma_duration.Avg())
					}

					learning_rate = learning_rate * anneal_rate
					if samples_count%100 == 0 {
						sample(dic, slow_net, fast_net, LookupTable, true)
						sample(dic, slow_net, fast_net, LookupTable, true)
						sample(dic, slow_net, fast_net, LookupTable, false)
						sample(dic, slow_net, fast_net, LookupTable, false)
					}
					if samples_count%1000 == 0 {
						gortex.SaveModel(fastModelName, fast_model)
						gortex.SaveModel(slowModelName, slow_model)
					}
					for i := 0; i < threads; i++ { // release train
						license <- 1
					}

					start = time.Now()

				}
			default:
				//skip
			}
		}
	}()
	var w sync.WaitGroup
	w.Add(1)
	go func() {
		gortex.WordSampleVisitor(trainFile, gortex.WordSplitter{}, dic, func(x [][]int) {
			if len(x) > 10 {
				<-license
				go func(x [][]int) {
					//fmt.Printf("%v\n",x)
					start := time.Now()
					// map term indexes in dictionary to embedding vectors
					var x_cost float32
					// make forward through rnn through time
					G := &gortex.Graph{NeedsBackprop: true}
					space := dic.IDByToken(" ")
					slow_ht := h0_slow
					//slow_ct := h0
					fast_ht := h0_fast
					//fast_ct := h0
					// slow network
					slow_len := 0
					fast_len := 0
					x_text := "|"
					y_text := "|"
					for j, word := range x {
						G.NeedsBackprop = j > 0
						//slow_ht, slow_ct, fast_ct = slow_net.Step(G, fast_ct, slow_ht, slow_ct)
						slow_ht, fast_ht = slow_net.Step(G, fast_ht, slow_ht)
						// fast network
						//add space to word at end and beginning of word
						word = append([]int{space}, word...)
						word = append(word, space)

						for i, char_id := range word[:len(word)-1] {
							x_text += dic.TokenByID(char_id)

							var fast_y *gortex.Matrix
							//fast_ht, fast_ct, fast_y = fast_net.Step(G, G.Lookup(LookupTable, char_id), fast_ht, fast_ct)
							fast_ht, fast_y = fast_net.Step(G, G.Lookup(LookupTable, char_id), fast_ht)
							// out task at each time step is to predict next symbol from rnn output
							cost, _ := G.Crossentropy(fast_y, word[i+1])
							y_id, _ := gortex.MaxIV(gortex.Softmax(fast_y))
							y_text += dic.TokenByID(y_id)
							if G.NeedsBackprop {
								x_cost += cost
								fast_len++
							}
						}
						if G.NeedsBackprop {
							slow_len++
						}
					}
					if count%100 == 0 {
						fmt.Printf("y: %s\n", y_text)
						fmt.Printf("x:%s\n", x_text)
					}
					G.Backward()

					gortex.PrintZeroGradient(fast_model)
					gortex.PrintZeroGradient(slow_model)

					x_cost /= float32(fast_len)
					x_perplexity := float32(math.Exp(float64(x_cost)))
					duration := time.Now().Sub(start)
					results <- &result{
						cost:       x_cost,
						perplexity: x_perplexity,
						bpc:        x_cost / math.Ln2,
						time:       duration,
						fast_speed: float32(fast_len) / float32(duration.Seconds()),
						slow_speed: float32(slow_len) / float32(duration.Seconds()),
						fast_len:   fast_len,
						slow_len:   slow_len,
					}
					//license <- 1
				}(x)

				count++

			}
		})
	}()
	w.Wait()
	for i := 0; i < threads; i++ {
		<-license
	}
}

func sample(dic *gortex.Dictionary, slow_net, fast_net *gortex.GRU, LookupTable *gortex.Matrix, multinomial bool) {

	fmt.Printf("MODEL GENERATED TEXT: ")
	G := gortex.Graph{NeedsBackprop: false}
	term_id := dic.IDByToken(" ")
	space := dic.IDByToken(" ")
	point := dic.IDByToken(".")
	ask := dic.IDByToken("?")
	eks := dic.IDByToken("!")
	slow_ht := gortex.RandMat(hidden_slow, 1)
	assembler.Sscale(0.01, slow_ht.W)
	//slow_ct := gortex.RandMat(hidden_size, 1)
	fast_ht := gortex.RandMat(hidden_fast, 1)
	assembler.Sscale(0.01, fast_ht.W)
	//fast_ct := gortex.RandMat(hidden_size, 1)
	var logits *gortex.Matrix
	for i := 0; i < 20; i++ { // generate 10 words
		term_id = space
		//	slow_ht, slow_ct, fast_ct = slow_net.Step(&G, fast_ct, slow_ht, slow_ct)
		slow_ht, fast_ht = slow_net.Step(&G, fast_ht, slow_ht)
		// fast network
		for j := 0; j < 32; j++ { // max word len 32
			//		fast_ht, fast_ct, logits = fast_net.Step(&G, G.Lookup(LookupTable, term_id), fast_ht, fast_ct)
			fast_ht, logits = fast_net.Step(&G, G.Lookup(LookupTable, term_id), fast_ht)
			if multinomial {
				term_id = gortex.Multinomial(gortex.Softmax(logits))
			} else {
				term_id, _ = gortex.MaxIV(gortex.Softmax(logits))

			}

			fmt.Printf("%s", dic.TokenByID(term_id))
			if term_id == space {
				break
			}
			if term_id == point || term_id == ask || term_id == eks {
				goto finish
			}
		}
	}
finish:
	fmt.Printf("\n")
}
