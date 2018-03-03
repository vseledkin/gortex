package assembler

import "log"

var useAVX2, useAVX, useSSE4 bool

func init() {
	useAVX2 = supportsAVX2()
	useAVX = supportsAVX()
	//useAVX = false
	useSSE4 = supportsSSE4()
}

var logging bool = true

func supportsSSE4() bool
func supportsAVX() bool
func supportsAVX2() bool

var Isamax func(x []float32) int
var Ismax func(x []float32) int

func Init(optimize bool) {
	if optimize {
		Isamax = isamax_asm
		Ismax = ismax_asm
	} else {
		Isamax = isamax
		Ismax = ismax
	}
}

func init() {
	Init(true)
	log.Printf("SSE4: %v", useSSE4)
	log.Printf("AVX: %v", useAVX)
	log.Printf("AVX2: %v", useAVX2)
}
