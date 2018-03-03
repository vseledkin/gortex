//+build amd64,!noasm

package assembler

import "log"

var Saxpy func(a float32, X []float32, Y []float32)

func init() {
	if useAVX {
		Saxpy = SaxpyAvx
		if logging {
			log.Print("Using SaxpyAvx")
		}
	} else if useSSE4 {
		Saxpy = SaxpySSE4
		if logging {
			log.Print("Using SaxpySSE4")
		}
	} else {
		panic("no candidate for Saxpy!")
	}
}

func SaxpySSE4(a float32, X []float32, Y []float32)

func SaxpyAvx(a float32, X []float32, Y []float32)
