package assembler

var Sum func(x []float32) float32
var L1 func(x []float32) float32
var Sscale func(v float32, x []float32)
var Isamax func(x []float32) int
var Ismax func(x []float32) int
var Sxmuley func(X, Y []float32)
var Sxmuleyplusz func(X, Y, Z []float32)

func Init(optimize bool) {
	if optimize {
		L1 = l1asm
		Sum = sumasm
		Sscale = sscale_asm
		Isamax = isamax_asm
		Ismax = ismax_asm
		Sxmuley = sxmuley_asm
		Sxmuleyplusz = sxmuleyplusz_asm
	} else {
		L1 = l1
		Sum = sum
		Sscale = sscale
		Isamax = isamax
		Ismax = ismax
		Sxmuley = sxmuley
		Sxmuleyplusz = sxmuleyplusz
	}
}

func init() {
	Init(true)
}
