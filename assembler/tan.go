package assembler

/*
	Floating-point tangent.
*/

// The original C code, the long comment, and the constants
// below were from http://netlib.sandia.gov/cephes/cmath/sin.c,
// available from http://www.netlib.org/cephes/cmath.tgz.
// The go code is a simplified version of the original C.
//
// For more information see:
// https://github.com/golang/go/blob/master/src/math/tan.go
//
// Cephes Math Library Release 2.8:  June, 2000
// Copyright 1984, 1987, 1989, 1992, 2000 by Stephen L. Moshier
//
// The readme file at http://netlib.sandia.gov/cephes/ says:
//    Some software in this archive may be from the book _Methods and
// Programs for Mathematical Functions_ (Prentice-Hall or Simon & Schuster
// International, 1989) or from the Cephes Mathematical Library, a
// commercial product. In either event, it is copyrighted by the author.
// What you see here may be used freely but it comes with no support or
// guarantee.
//
//   The two known misprints in the book are repaired here in the
// source listings for the gamma function and the incomplete beta
// integral.
//
//   Stephen L. Moshier
//   moshier@na-net.ornl.gov
import "math"

const (
	uvnan    = 0x7FC00001
	uvinf    = 0x7F800000
	uvneginf = 0xFF800000
)

// Inf returns positive infinity if sign >= 0, negative infinity if sign < 0.
func Inf(sign int) float32 {
	var v uint32
	if sign >= 0 {
		v = uvinf
	} else {
		v = uvneginf
	}
	return math.Float32frombits(v)
}

// NaN returns an IEEE 754 ``not-a-number'' value.
func NaN() float32 { return math.Float32frombits(uvnan) }

// IsNaN reports whether f is an IEEE 754 ``not-a-number'' value.
func IsNaN(f float32) (is bool) {
	return f != f
}

// IsInf reports whether f is an infinity, according to sign.
// If sign > 0, IsInf reports whether f is positive infinity.
// If sign < 0, IsInf reports whether f is negative infinity.
// If sign == 0, IsInf reports whether f is either infinity.
func IsInf(f float32, sign int) bool {
	return sign >= 0 && f > math.MaxFloat32 || sign <= 0 && f < -math.MaxFloat32
}

// tan coefficients
var _tanP = [...]float32{
	-1.30936939181383777646E4, // 0xc0c992d8d24f3f38
	1.15351664838587416140E6,  // 0x413199eca5fc9ddd
	-1.79565251976484877988E7, // 0xc1711fead3299176
}
var _tanQ = [...]float32{
	1.00000000000000000000E0,
	1.36812963470692954678E4,  //0x40cab8a5eeb36572
	-1.32089234440210967447E6, //0xc13427bc582abc96
	2.50083801823357915839E7,  //0x4177d98fc2ead8ef
	-5.38695755929454629881E7, //0xc189afe03cbe5a31
}

// Tan returns the tangent of the radian argument x.
//
// Special cases are:
//	Tan(±0) = ±0
//	Tan(±Inf) = NaN
//	Tan(NaN) = NaN
func Tan(x float32) float32

func tan(x float32) float32 {
	const (
		PI4A = 7.85398125648498535156E-1                             // 0x3fe921fb40000000, Pi/4 split into three parts
		PI4B = 3.77489470793079817668E-8                             // 0x3e64442d00000000,
		PI4C = 2.69515142907905952645E-15                            // 0x3ce8469898cc5170,
		M4PI = 1.273239544735162542821171882678754627704620361328125 // 4/pi
	)
	// special cases
	switch {
	case x == 0 || IsNaN(x):
		return x // return ±0 || NaN()
	case IsInf(x, 0):
		return NaN()
	}

	// make argument positive but save the sign
	sign := false
	if x < 0 {
		x = -x
		sign = true
	}

	j := int64(x * M4PI) // integer part of x/(Pi/4), as integer for tests on the phase angle
	y := float32(j)      // integer part of x/(Pi/4), as float

	/* map zeros and singularities to origin */
	if j&1 == 1 {
		j += 1
		y += 1
	}

	z := ((x - y*PI4A) - y*PI4B) - y*PI4C
	zz := z * z

	if zz > 1e-14 {
		y = z + z*(zz*(((_tanP[0]*zz)+_tanP[1])*zz+_tanP[2])/((((zz+_tanQ[1])*zz+_tanQ[2])*zz+_tanQ[3])*zz+_tanQ[4]))
	} else {
		y = z
	}
	if j&2 == 2 {
		y = -1 / y
	}
	if sign {
		y = -y
	}
	return y
}

var TanGo = tan
