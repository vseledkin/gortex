//+build amd64,!noasm

package assembler

import "log"

var Sxmuley func(X, Y []float32)

func init() {
	if useAVX {
		Sxmuley = SxmyAvx
		if logging {
			log.Print("Using SxmyAvx")
		}
	} else if useSSE4 {
		Sxmuley = SxmySSE4
		if logging {
			log.Print("Using SxmySSE4")
		}
	} else {
		panic("no candidate for Sxmuley!")
	}
}

func SxmySSE4(X, Y []float32)

func SxmyAvx(X, Y []float32)
