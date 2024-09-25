package utils

import (
	"encoding/gob"
	"errors"
	"math/rand"
	"os"
	"time"
	"tp/models/ann"
)

// Función para dividir los datos
func SplitData(data [][]float64, labels []float64, percentage float64) ([][]float64, [][]float64, []float64, []float64, error) {
	// Verificar que las longitudes coincidan
	if len(data) != len(labels) {
		return nil, nil, nil, nil, errors.New("la longitud de los datos y las etiquetas no coinciden")
	}

	// Verificar que el porcentaje esté en el rango [0.0, 1.0]
	if percentage < 0.0 || percentage > 1.0 {
		return nil, nil, nil, nil, errors.New("el porcentaje debe estar entre 0.0 y 1.0")
	}

	total := len(data)
	splitIndex := int(float64(total) * percentage)

	// Crear índices aleatorios
	rand.Seed(time.Now().UnixNano()) // Usar una semilla para garantizar aleatoriedad
	indices := rand.Perm(total)

	// Inicializar slices con la capacidad adecuada
	trainData := make([][]float64, 0, total-splitIndex)
	trainLabels := make([]float64, 0, total-splitIndex)
	testData := make([][]float64, 0, splitIndex)
	testLabels := make([]float64, 0, splitIndex)

	// Llenar los conjuntos de entrenamiento y prueba
	for i, idx := range indices {
		if i >= splitIndex {
			trainData = append(trainData, data[idx])
			trainLabels = append(trainLabels, labels[idx])
		} else {
			testData = append(testData, data[idx])
			testLabels = append(testLabels, labels[idx])
		}
	}

	return trainData, testData, trainLabels, testLabels, nil
}

func SaveModel(Ann *ann.ANN, filename string) error {
	// Crear el archivo
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Codificar el modelo usando gob
	encoder := gob.NewEncoder(file)
	err = encoder.Encode(Ann)
	if err != nil {
		return err
	}

	return nil
}

func LoadModel(filename string) (*ann.ANN, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Decodificar el modelo usando gob
	var ann ann.ANN
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&ann)
	if err != nil {
		return nil, err
	}

	return &ann, nil
}

// TrainTestSplit divides the data into training and testing sets
func TrainTestSplit(xData [][]float64, yData []float64, testSize float64) (trainX [][]float64, trainY []float64, testX [][]float64, testY []float64) {
    // Seed the random number generator to ensure reproducibility
    rand.Seed(time.Now().UnixNano())

    // Calculate the number of test samples
    totalSamples := len(xData)
    numTestSamples := int(testSize * float64(totalSamples))

    // Generate a list of indices and shuffle them
    indices := rand.Perm(totalSamples)

    // Split the indices into training and testing indices
    testIndices := indices[:numTestSamples]
    trainIndices := indices[numTestSamples:]

    // Initialize slices for the output data
    trainX = make([][]float64, len(trainIndices))
    trainY = make([]float64, len(trainIndices))
    testX = make([][]float64, len(testIndices))
    testY = make([]float64, len(testIndices))

    // Fill the training data
    for i, idx := range trainIndices {
        trainX[i] = xData[idx]
        trainY[i] = yData[idx]
    }

    // Fill the testing data
    for i, idx := range testIndices {
        testX[i] = xData[idx]
        testY[i] = yData[idx]
    }
    return
}

func MinMaxScaler(data [][]float64) ([][]float64, []float64, []float64) {
    min := make([]float64, len(data[0]))
    max := make([]float64, len(data[0]))

    for j := range data[0] {
        min[j] = data[0][j]
        max[j] = data[0][j]
    }

    for _, row := range data {
        for j, val := range row {
            if val < min[j] {
                min[j] = val
            }
            if val > max[j] {
                max[j] = val
            }
        }
    }

    scaledData := make([][]float64, len(data))
    for i, row := range data {
        scaledRow := make([]float64, len(row))
        for j, val := range row {
            scaledRow[j] = (val - min[j]) / (max[j] - min[j])
        }
        scaledData[i] = scaledRow
    }

    return scaledData, min, max
}

func ScaleData(data [][]float64, min, max []float64) [][]float64 {
    scaledData := make([][]float64, len(data))
    for i, row := range data {
        scaledRow := make([]float64, len(row))
        for j, val := range row {
            scaledRow[j] = (val - min[j]) / (max[j] - min[j])
        }
        scaledData[i] = scaledRow
    }
    return scaledData
}

func MinMaxScalerSingle(data []float64) ([]float64, float64, float64) {
    min := data[0]
    max := data[0]

    for _, val := range data {
        if val < min {
            min = val
        }
        if val > max {
            max = val
        }
    }

    scaledData := make([]float64, len(data))
    for i, val := range data {
        scaledData[i] = (val - min) / (max - min)
    }

    return scaledData, min, max
}

func ScaleDataSingle(data []float64, min, max float64) []float64 {
    scaledData := make([]float64, len(data))
    for i, val := range data {
        scaledData[i] = (val - min) / (max - min)
    }
    return scaledData
}