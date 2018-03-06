//+build amd64,!noasm

package assembler

import "log"

var Sxpy func(X, Y []float32)

func init() {
	if useAVX {
		Sxpy = SxpyAvx
		if logging {
			log.Print("Using SxpyAvx")
		}
	} else if useSSE4 {
		Sxpy = SxpySSE4
		if logging {
			log.Print("Using SxpySSE4")
		}
	} else {
		panic("no candidate for Sxpuley!")
	}
}
func SxpySSE4(X, Y []float32)

func SxpyAvx(X, Y []float32)
