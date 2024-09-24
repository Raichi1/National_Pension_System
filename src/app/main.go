package main

import (
	"fmt"
	"tp/models/deep"
	"tp/panditas"
	"tp/utils"
)

func main() {

	dnn := deep.NeuralNetwork{}
	dnn.Initialize([]int{5, 10, 5, 1}, 0.0001)

	df, _ := panditas.ReadCSV("../dataset/dataset_clean.csv")

	features, label, _ := df.GetFeaturesAndLabels("target")

	X_train, X_test, Y_train, Y_test, _ := utils.SplitData(features, label, 0.2)

	dnn.TrainConcurrently(X_train, Y_train, 5, 32)
	// Entrenar la red

	// Calcular precision %
	accuracy := dnn.Accuracy(X_test, Y_test)
	fmt.Println("Accuracy:", accuracy)

	// Console for input data from the user
	var input []float64
	columns := df.Headers
	i := 1
	for {
		fmt.Printf("\nPrediccion %d : \n", i)
		for i := 0; i < len(columns)-1; i++ {
			fmt.Printf("Ingrese el valor de %s \n", columns[i])
			var value float64
			fmt.Scan(&value)
			input = append(input, value)
		}
		// Predict
		predict := dnn.PredictUser(input)
		fmt.Printf("\nRemuneracion Esperada: S/. %.2f\n", predict[0])
		// clean input
		input = nil
		i++
	}

}
