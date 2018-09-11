package assembler

import (
	"log"
	"golang.org/x/sys/cpu"
)

var useAVX2, useAVX, useSSE4 bool

func init() {

	useSSE4 = cpu.X86.HasSSE41
	useAVX = cpu.X86.HasAVX
	useAVX2 = cpu.X86.HasAVX2

	Init(true)
	log.Printf("SSE4: %v", useSSE4)
	log.Printf("AVX: %v", useAVX)
	log.Printf("AVX2: %v", useAVX2)

}

var logging bool = true

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
