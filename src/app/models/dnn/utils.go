package dnn

import (
    "math"
    "fmt"
)

type Vector []float64
type Frame []Vector

// DotProduct calcula el producto punto entre dos vectores
func DotProduct(a, b Vector) float64 {
    if len(a) != len(b) {
        panic(fmt.Errorf(
			"cannot dot product arrays of unequal length: %d, %d",
			len(a),
			len(b),
		))
    }
    var res float64
    for i := range a {
        res += a[i] * b[i]
    }
    return res
}

// Subtract resta dos vectores
func (a Vector) Subtract(b Vector) Vector {
    if len(a) != len(b) {
        panic("vectors must be of the same length")
    }
    result := make(Vector, len(a))
    for i := range a {
        result[i] = a[i] - b[i]
    }
    return result
}

// Scalar multiplica un vector por un escalar
func (a Vector) Scalar(scalar float64) Vector {
    result := make(Vector, len(a))
    for i := range a {
        result[i] = a[i] * scalar
    }
    return result
}

// Apply aplica una función a cada elemento del vector
func (a Vector) Apply(fn func(float64) float64) Vector {
    result := make(Vector, len(a))
    for i := range a {
        result[i] = fn(a[i])
    }
    return result
}

// ElementwiseProduct realiza el producto elemento a elemento de dos vectores
func (a Vector) ElementwiseProduct(b Vector) Vector {
    if len(a) != len(b) {
        panic("vectors must be of the same length")
    }
    result := make(Vector, len(a))
    for i := range a {
        result[i] = a[i] * b[i]
    }
    return result
}

// Subtract resta dos matrices
func (a Frame) Subtract(b Frame) Frame {
    if len(a) != len(b) {
        panic("frames must be of the same length")
    }
    result := make(Frame, len(a))
    for i := range a {
        result[i] = a[i].Subtract(b[i])
    }
    return result
}

// Scalar multiplica una matriz por un escalar
func (a Frame) Scalar(scalar float64) Frame {
    result := make(Frame, len(a))
    for i := range a {
        result[i] = a[i].Scalar(scalar)
    }
    return result
}

//lineal activaction
func Lineal(x float64) float64 {
    return x
}
//lineal derivative
func LinealDerivative(x float64) float64 {
    return 1
}

func ReLU(x float64) float64 {
    if x < 0 {
        return 0
    }
    return x
}

func ReLUDerivative(x float64) float64 {
    if x > 0 {
        return 1
    }
    return 0
}

// Sigmoid es la función de activación sigmoide
func Sigmoid(x float64) float64 {
    return 1 / (1 + float64(math.Exp(float64(-x))))
}

// SigmoidDerivative calcula la derivada de la función sigmoide
func SigmoidDerivative(x float64) float64 {
    sig := Sigmoid(x)
    return sig * (1 - sig)
}

// CalculateAccuracy calcula la precisión de las predicciones
func CalculateAccuracy(predictions Frame, actual []float64) float64 {
    correct := 0
    for i, prediction := range predictions {
        predictedLabel := 0
        if prediction[0] >= 0.5 {
            predictedLabel = 1
        }
        if float64(predictedLabel) == actual[i] {
            correct++
        }
    }
    return float64(correct) / float64(len(actual))
}