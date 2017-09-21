package main

import (
	"time"
	"fmt"
	"math/rand"
	g "github.com/vseledkin/gortex"
	"log"
	"bufio"
	"os"
	"strings"
)

func main() {
	// maintain random seed
	rand.Seed(time.Now().UnixNano())
	tokenizer := g.CharSplitter{}
	trainFile := "train.txt"
	dic, e := g.DictionaryFromFile(trainFile, tokenizer)
	if e != nil {
		log.Fatal(e)
	}
	g.SaveDictionary("dic.json", dic)
	hidden_size := 128
	fmt.Printf("Dictionary has %d tokens\n", dic.Len())
	fmt.Printf("%s\n", dic)

	optimizer := g.NewOptimizer(g.OpOp{Method: g.WINDOWGRAD, LearningRate: 0.001, Momentum: g.DefaultMomentum, Clip: 4})

	encoder := g.MakeOutputlessRNN(dic.Len(), hidden_size)
	Who := g.RandXavierMat(2, hidden_size)
	Bho := g.RandXavierMat(2, 1)
	// define model parameters
	encoderModel := encoder.GetParameters("Encoder")
	encoderModel["out"] = Who
	encoderModel["outBias"] = Bho

	count := 0
	ma_loss := g.NewMovingAverage(100)

	learning_rate := float32(0.001)
	anneal_rate := float32(0.999)

	g.CharClassifierSampleVisitor(trainFile, 1, true, tokenizer, dic, func(x []uint, label string) {
		if rand.NormFloat64() < 0.4 {
			x = append([]uint{52}, x...)
		}
		// read sample
		sample := ""
		for i := range x {
			sample += dic.TokenByID(x[i])
		}
		G := &g.Graph{NeedsBackprop: true}
		ht := g.Mat(hidden_size, 1) // vector of zeros
		// encode sequence into z
		for i := range x {
			oneHot := g.Mat(dic.Len(), 1)
			oneHot.W[x[i]] = 1.0
			ht = encoder.Step(G, oneHot, ht)
		}
		logits := G.Add(G.Mul(Who, ht), Bho)
		var target uint
		if label == "__label__ru" {
			target = 1
		}
		predicted_class, _ := g.MaxIV(g.Softmax(logits))
		cost, prob := G.Crossentropy(logits, target)

		G.Backward()

		optimizer.Step(encoderModel)

		count++

		ma_loss.Add(cost)

		avg_cost := ma_loss.Avg()
		if count%50 == 0 {
			fmt.Printf("step: %d lr: %f  loss: %f label: [%s] [%d] ? %d %f sample: [%s]\n",
				count, learning_rate, avg_cost, label, target, predicted_class, prob, sample)
			learning_rate = learning_rate * anneal_rate
		}

		if avg_cost < 1e-5 { // TEST
			var trueLabels, predictedLabels []uint
			g.CharClassifierSampleVisitor("test.txt", 1, false, tokenizer, dic, func(x []uint, label string) {
				// read sample
				sample := ""
				for i := range x {
					sample += dic.TokenByID(x[i])
				}

				G := &g.Graph{NeedsBackprop: false}
				ht := g.Mat(hidden_size, 1) // vector of zeros
				// encode sequence into z
				for i := range x {
					oneHot := g.Mat(dic.Len(), 1)
					oneHot.W[x[i]] = 1.0
					ht = encoder.Step(G, oneHot, ht)
				}
				logits := G.Add(G.Mul(Who, ht), Bho)
				var target uint
				if label == "__label__ru" {
					target = 1
				}
				trueLabels = append(trueLabels, target)

				predicted_class, _ := g.MaxIV(g.Softmax(logits))
				predictedLabels = append(predictedLabels, predicted_class)
				//fmt.Printf("sample %d %d %s\n", predicted_class, target, sample)
			})
			F1, message := g.F1Score(trueLabels, predictedLabels, []string{"ru", "en"}, nil)
			fmt.Printf("\n\nF1: %f %s\n\n", F1, message)

		}
		if avg_cost < 1e-5 {
			g.SaveModel("model.json", encoderModel)
			for true {
				r := bufio.NewReader(os.Stdin)
				line, e := r.ReadString('\n')
				if e != nil {
					break
				}
				line = strings.TrimSpace(line)
				println(line)
				G := &g.Graph{NeedsBackprop: false}
				ht := g.Mat(hidden_size, 1) // vector of zeros
				// encode sequence into z
				for _, char := range tokenizer.Split(line) {
					char_id := dic.IDByToken(char)
					oneHot := g.Mat(dic.Len(), 1)
					oneHot.W[char_id] = 1.0
					ht = encoder.Step(G, oneHot, ht)
				}
				logits := G.Add(G.Mul(Who, ht), Bho)

				predicted_label_index, probability := g.MaxIV(g.Softmax(logits))
				predicted_label := ""
				if predicted_label_index == 0 {
					predicted_label = "en"
				} else {
					predicted_label = "ru"
				}
				fmt.Printf("Input: [%s] Predicted label: %s Probabilty: %f\n", line, predicted_label, probability)
			}
		}
	})
}
