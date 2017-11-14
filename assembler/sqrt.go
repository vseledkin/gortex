//+build !amd64 noasm

package assembler

import "math"

func Sqrt(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}
