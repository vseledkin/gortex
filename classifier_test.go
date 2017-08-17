package gortex

import (
	"fmt"
	"math/rand"
	"testing"
	"time"
)

func TestClassifier(t *testing.T) {
	// maintain random seed
	rand.Seed(time.Now().UnixNano())
	trainFile := "train.txt"
	dic, e := DictionaryFromFile(trainFile, CharSplitter{})
	if e != nil {
		t.Fatal(e)
	}
	hidden_size := 128
	fmt.Printf("Dictionary has %d tokens\n", dic.Len())
	fmt.Printf("%s\n", dic)

	optimizer := NewOptimizer(OpOp{Method: WINDOWGRAD, LearningRate: 0.0003, Momentum: DefaultMomentum, Clip: 4})

	encoder := MakeGRU(dic.Len(), hidden_size, 1)
	Who := RandXavierMat(2, hidden_size)
	Bho := RandXavierMat(2, 1)
	// define model parameters
	encoderModel := encoder.GetParameters("Encoder")
	encoderModel["out"] = Who
	encoderModel["outBias"] = Bho

	count := 0
	ma_d := NewMovingAverage(256)

	learning_rate := float32(0.001)
	anneal_rate := float32(0.999)

	CharClassifierSampleVisitor(trainFile, 10, true, CharSplitter{}, dic, func(x []uint, label string) {
		// read sample
		sample := ""
		for i := range x {
			sample += dic.TokenByID(x[i])
		}
		G := &Graph{NeedsBackprop: true}
		ht := Mat(hidden_size, 1) // vector of zeros
		// encode sequence into z
		for i := range x {
			oneHot := Mat(dic.Len(), 1)
			oneHot.W[x[i]] = 1.0
			ht, _ = encoder.Step(G, oneHot, ht)
		}
		logits := G.Add(G.Mul(Who, ht), Bho)
		var target uint
		if label == "__label__Task" {
			target = 1
		}
		predicted_class, _ := MaxIV(Softmax(logits))
		cost, prob := G.Crossentropy(logits, target)

		G.Backward()
		//ScaleGradient(encoderModel, 1.0/float32(len(x)))
		optimizer.Step(encoderModel)

		count++

		ma_d.Add(cost)

		avg_cost := ma_d.Avg()
		if count%50 == 0 {
			fmt.Printf("label: [%s] [%d] ? %d %f\n", label, target, predicted_class, prob)
			fmt.Printf("encoded: [%s]\n", sample)
			fmt.Printf("step: %d loss: %f lr: %f\n", count, avg_cost, learning_rate)
			learning_rate = learning_rate * anneal_rate
		}

		if count%1000 == 0 { // TEST
			var trueLabels, predictedLabels []uint
			CharClassifierSampleVisitor("test.txt", 10, false, CharSplitter{}, dic, func(x []uint, label string) {
				// read sample
				sample := ""
				for i := range x {
					sample += dic.TokenByID(x[i])
				}
				G := &Graph{NeedsBackprop: false}
				ht := Mat(hidden_size, 1) // vector of zeros
				// encode sequence into z
				for i := range x {
					oneHot := Mat(dic.Len(), 1)
					oneHot.W[x[i]] = 1.0
					ht, _ = encoder.Step(G, oneHot, ht)
				}
				logits := G.Add(G.Mul(Who, ht), Bho)
				var target uint
				if label == "__label__Task" {
					target = 1
				}
				trueLabels = append(trueLabels, target)

				predicted_class, _ := MaxIV(Softmax(logits))
				predictedLabels = append(predictedLabels, predicted_class)
				//fmt.Printf("sample %d %d %s\n", predicted_class, target, sample)
			})
			F1, message := F1Score(trueLabels, predictedLabels, []string{"Neutral", "Task"}, nil)
			fmt.Printf("\n\nF1: %f %s\n\n", F1, message)
		}
	})
}
