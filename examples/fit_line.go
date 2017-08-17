package main

import (
	"fmt"
	g "github.com/vseledkin/gortex"
)

func main() {
	// train data two pints
	X := []float32{1, 0}
	Y := []float32{1, 2}
	// model parameters
	A := g.RandMat(1, 1)
	B := g.RandMat(1, 1)
	modelParameters := map[string]*g.Matrix{"A": A, "B": B}
	// optimizer
	optimizer := g.NewOptimizer(g.OpOp{Method: g.SGD, LearningRate: 0.01})
	epoch, maxEpochs := 0, 10000
	for true {
		cost := float32(0)
		for i := range X {
			y_vector := g.Mat(1, 1)
			y_vector.Set(0, 0, Y[i])
			x_vector := g.Mat(1, 1)
			x_vector.Set(0, 0, X[i])
			// construct y=Ax+b model
			graph := &g.Graph{NeedsBackprop: true}
			result := graph.Add(graph.Mul(A, x_vector), B)
			currentSampleError := graph.MSE(result, y_vector)
			cost += currentSampleError
			graph.Backward()

			optimizer.Step(modelParameters)
			fmt.Printf("Epoch: %d Sample %d A*x+B=y %f*%f + %f = %f error=%f\n",
				epoch, i, A.Get(0, 0), x_vector.Get(0, 0), B.Get(0, 0), result.Get(0, 0), currentSampleError)
		}
		cost /= 2 // per sample cost
		epoch++
		if cost < 1e-4 || epoch == maxEpochs {
			fmt.Printf("Final loss: %f \n", cost)
			fmt.Printf("A = %f, B = %f\n", A.Get(0, 0), B.Get(0, 0))
			break
		}
	}
}
