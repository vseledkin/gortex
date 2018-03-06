//+build amd64,!noasm

package assembler

//Sset  set all components of a vector to a
func Sset(a float32, x []float32)

func SsetAVX(a float32, x []float32)
