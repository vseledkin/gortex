package assembler

var L1 func(x []float32) float32
var Sscale func(v float32, x []float32)
var Isamax func(x []float32) int
var Ismax func(x []float32) int
var Sxmulely func(X, Y []float32)
var Sxmulelyplusz func(X, Y, Z []float32)

func Init(optimize bool) {
	if optimize {
		L1 = l1asm
		Sscale = sscale_asm
		Isamax = isamax_asm
		Ismax = ismax_asm
		Sxmulely = sxmulely_asm
		Sxmulelyplusz = sxmulelyplusz_asm
	} else {
		L1 = l1
		Sscale = sscale
		Isamax = isamax
		Ismax = ismax
		Sxmulely = sxmulely
		Sxmulelyplusz = sxmulelyplusz
	}
}

func init() {
	Init(false)
}
