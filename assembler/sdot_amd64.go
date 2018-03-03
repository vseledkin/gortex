//+build amd64,!noasm

package assembler

import "log"

var Sdot func(X, Y []float32) float32

func init() {
	if useAVX {
		Sdot = SdotAvx
		if logging {
			log.Print("Using SdotAvx")
		}
	} else if useSSE4 {
		Sdot = SdotSSE4
		if logging {
			log.Print("Using SdotSSE4")
		}
	} else {
		panic("no candidate for Sdot!")
	}
}

func SdotSSE4(X, Y []float32) float32

func SdotAvx(X, Y []float32) float32
