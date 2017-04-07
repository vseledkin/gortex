package assembler

var L1 func(x []float32) float32
var Sscale func(x []float32) float32

func Init(optimize bool) {
	if optimize {
		L1 = l1asm
		Sscale = sscale_asm
	} else {
		L1 = l1
		Sscale = sscale
	}
}

func init(){
	Init(false)
}