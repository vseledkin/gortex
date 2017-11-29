package assembler



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
}
