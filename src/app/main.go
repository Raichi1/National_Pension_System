package main

import (
	"fmt"
	"tp/models/deep"
	"tp/panditas"
	"tp/utils"
)

func main() {
	var learningRate float64
	var layers []int

	// Mensaje de bienvenida
	fmt.Println("\n-----------------------------------------------------")
	fmt.Println("Bienvenido al sistema de predicción de remuneraciones")
	fmt.Println("---------------------------------------------------")

	for {
		fmt.Println("\nMenú de opciones:")
		fmt.Println("1. Ingresar datos para predicción")
		fmt.Println("2. Modificar parámetros de la red neuronal")
		fmt.Println("3. Salir")
		fmt.Print("Seleccione una opción: ")

		var option int
		fmt.Scan(&option)

		switch option {
		case 1:
			if learningRate == 0 || len(layers) == 0 {
				fmt.Println("Primero debe configurar los parámetros de la red neuronal.")
				continue
			}

			dnn := deep.NeuralNetwork{}
			dnn.Initialize(layers, learningRate)

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
			it := 1

			for {
				fmt.Printf("\nPredicción %d : \n", it)
				for i := 0; i < len(columns)-1; i++ {
					switch i {
					case 0:
						fmt.Printf("Ingrese el valor de %s (0: No declaró importe, 1: Sí declaró importe): ", columns[i])
					case 2:
						fmt.Printf("Ingrese el valor de %s (0: Privado, 1: Público): ", columns[i])
					case 4:
						fmt.Printf("Ingrese el valor de %s (número de meses): ", columns[i])
					default:
						fmt.Printf("Ingrese el valor de %s: ", columns[i])
					}

					var value float64
					_, err := fmt.Scan(&value)
					if err != nil {
						fmt.Println("Entrada no válida. Por favor, ingrese un número.")
						i-- // Repetir la entrada para el mismo índice
						continue
					}
					input = append(input, value)
				}

				// Predict
				predict := dnn.PredictUser(input)
				fmt.Printf("\nRemuneración Esperada: S/. %.2f\n", predict[0])

				// Limpiar input
				input = nil
				it++

				// Preguntar si desea realizar otra predicción
				var continueOption string
				fmt.Print("¿Desea realizar otra predicción? (s/n): ")
				fmt.Scan(&continueOption)
				if continueOption != "s" && continueOption != "S" {
					break
				}
			}

		case 2:
			fmt.Print("Ingrese la tasa de aprendizaje (learning rate): ")
			fmt.Scan(&learningRate)

			fmt.Print("Ingrese la cantidad de capas intermedias: ")
			var numLayers int
			fmt.Scan(&numLayers)

			layers = []int{5} // La primera capa siempre es 5
			for i := 0; i < numLayers; i++ {
				fmt.Printf("Ingrese el número de neuronas para la capa %d: ", i+1)
				var neurons int
				fmt.Scan(&neurons)
				layers = append(layers, neurons)
			}
			layers = append(layers, 1) // La última capa siempre es 1
			fmt.Println("Parámetros actualizados.")

		case 3:
			fmt.Println("Gracias por usar el sistema de predicción de remuneraciones. ¡Hasta luego!")
			return

		default:
			fmt.Println("Opción no válida. Por favor, seleccione una opción del menú.")
		}
	}
}
