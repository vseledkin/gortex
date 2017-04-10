package assembler

var L1 func(x []float32) float32
var Sscale func(v float32, x []float32)
var Isamax func(x []float32) int
var Ismax func(x []float32) int

func Init(optimize bool) {
	if optimize {
		L1 = l1asm
		Sscale = sscale_asm
		Isamax = isamax_asm
		Ismax = ismax_asm
	} else {
		L1 = l1
		Sscale = sscale
		Isamax = isamax
		Ismax = ismax
	}
}

func init() {
	Init(false)
}
