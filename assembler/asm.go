package assembler



var Isamax func(x []float32) int
var Ismax func(x []float32) int
var Sxmuley func(X, Y []float32)
var Sxmuleyplusz func(X, Y, Z []float32)

func Init(optimize bool) {
	if optimize {

		Isamax = isamax_asm
		Ismax = ismax_asm
		Sxmuley = sxmuley_asm
		Sxmuleyplusz = sxmuleyplusz_asm
	} else {

		Isamax = isamax
		Ismax = ismax
		Sxmuley = sxmuley
		Sxmuleyplusz = sxmuleyplusz
	}
}

func init() {
	Init(true)
}
