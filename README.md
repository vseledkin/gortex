# gortex

Pure go neural network library

## Example code
    
    Features: 
        auto backpropagation
        pure golang
        optimized

## Example 1: fit linear model to two points
```go
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
			// make data vectors
			y_vector := g.Mat(1, 1)
			y_vector.Set(0, 0, Y[i])
			x_vector := g.Mat(1, 1)
			x_vector.Set(0, 0, X[i])
			// make calculation graph, use true flag to denote we need to maintain back-propagation graph to use graph.Backward() later
			// use false at inference time to save memory/calculations
			graph := &g.Graph{NeedsBackprop: true}
			// construct y=Ax+b model
			result := graph.Add(graph.Mul(A, x_vector), B)
			// use mean square error loss MSE(A*x+B, target_value)
			currentSampleError := graph.MSE(result, y_vector)
			// accumulate cost
			cost += currentSampleError
			// auto-back-propagate graph
			graph.Backward()

			optimizer.Step(modelParameters)
			fmt.Printf("Epoch: %d Sample %d A*x+B=y %f*%f + %f = %f error=%f\n",
				epoch, i, A.Get(0, 0), x_vector.Get(0, 0), B.Get(0, 0), result.Get(0, 0), currentSampleError)
		}
		cost /= 2 // per sample cost average
		epoch += 1
		if cost < 1e-4 || epoch == maxEpochs { // we are aimed to get low absolute error value
			// final error
			fmt.Printf("Final loss: %f \n", cost)
			// print solution
			fmt.Printf("A = %f, B = %f\n", A.Get(0, 0), B.Get(0, 0))
			break
		}
	}
}



```
## Warning: Beta

Very beta




