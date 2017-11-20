package assembler



var Isamax func(x []float32) int
var Ismax func(x []float32) int

var Sxmuleyplusz func(X, Y, Z []float32)

func Init(optimize bool) {
	if optimize {

		Isamax = isamax_asm
		Ismax = ismax_asm

		Sxmuleyplusz = sxmuleyplusz_asm
	} else {

		Isamax = isamax
		Ismax = ismax
		Sxmuleyplusz = sxmuleyplusz
	}
}

func init() {
	Init(true)
}
