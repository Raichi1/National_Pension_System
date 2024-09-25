package main

import (
    "fmt"
    //ann "tp/models/ann"
    "tp/models/dnn"
    "tp/panditas"
    "tp/utils"
)

func main() {
    df, _ := panditas.ReadCSV("../dataset/dataset_clean.csv")

    features, label, _ := df.GetFeaturesAndLabels("target")
    trainX, trainY, testX, testY := utils.TrainTestSplit(features, label, 0.2)

    // Escalar los datos de entrenamiento
    scaledTrainX, minX, maxX := utils.MinMaxScaler(trainX)
    scaledTestX := utils.ScaleData(testX, minX, maxX)

    // Escalar las etiquetas de entrenamiento
    trainYFloat := make([]float64, len(trainY))
    for i, val := range trainY {
        trainYFloat[i] = float64(val)
    }
    scaledTrainY, minY, maxY := utils.MinMaxScalerSingle(trainYFloat)

    // Escalar las etiquetas de prueba
    testYFloat := make([]float64, len(testY))
    for i, val := range testY {
        testYFloat[i] = float64(val)
    }
    scaledTestY := utils.ScaleDataSingle(testYFloat, minY, maxY)

    trainXFrame := make(dnn.Frame, len(scaledTrainX))
    for i, row := range scaledTrainX {
        vector := make(dnn.Vector, len(row))
        for j, val := range row {
            vector[j] = float64(val)
        }
        trainXFrame[i] = vector
    }
    trainYFrame := make(dnn.Frame, len(scaledTrainY))
    for i, val := range scaledTrainY {
        vector := make(dnn.Vector, 1)
        vector[0] = float64(val)
        trainYFrame[i] = vector
    }
    testXFrame := make(dnn.Frame, len(scaledTestX))
    for i, row := range scaledTestX {
        vector := make(dnn.Vector, len(row))
        for j, val := range row {
            vector[j] = float64(val)
        }
        testXFrame[i] = vector
    }
    testYFrame := make(dnn.Frame, len(scaledTestY))
    for i, val := range scaledTestY {
        vector := make(dnn.Vector, 1)
        vector[0] = float64(val)
        testYFrame[i] = vector
    }

    inputSize := len(trainX[0])
    outputSize := 1
    epochs := 1
    nn := &dnn.MLPSequencial{
        Layers: []*dnn.LayerSequencial{
            {Name: "Input Layer", Width: inputSize},
            {Name: "Hidden Layer", Width: 10, ActivationFunction: dnn.ReLU, ActivationFunctionDeriv: dnn.ReLUDerivative},
            {Name: "Output Layer", Width: outputSize, ActivationFunction: dnn.Lineal, ActivationFunctionDeriv: dnn.LinealDerivative},
        },
        LearningRate: 0.1,
        Introspect: func(step dnn.StepSequencial) {
            fmt.Printf("Epoch: %d, Loss: %f\n", step.Epoch, step.LossSequencial)
        },
    }

    //fmt.Println(trainXFrame)
    //fmt.Println(trainYFrame)
    loss, err := nn.TrainSequencial(epochs, trainXFrame, trainYFrame)
    if err != nil {
        fmt.Println("Error durante el entrenamiento:", err)
        return
    }

    fmt.Printf("Entrenamiento completado con pérdida final: %f\n", loss)

	// Crear y escalar testXFrame2
    testXFrame2 := [][]float64{
        {1,96.0,120.41,202303.0,25638.41},
    }
    scaledTestXFrame2 := utils.ScaleData(testXFrame2, minX, maxX)
    // Convertir [][]float64 a dnn.Frame
    scaledTestXFrame2DNN := make(dnn.Frame, len(scaledTestXFrame2))
    for i, row := range scaledTestXFrame2 {
        scaledTestXFrame2DNN[i] = make(dnn.Vector, len(row))
        for j, val := range row {
            scaledTestXFrame2DNN[i][j] = val
        }
    }
    // Hacer predicciones
    predictions := nn.PredictSequencial(testXFrame)
    testing := nn.PredictSequencial(scaledTestXFrame2DNN)

	descaledTesting := make(dnn.Frame, len(testing))
    for i, pred := range testing {
        descaledTesting[i] = make(dnn.Vector, len(pred))
        for j, val := range pred {
            descaledTesting[i][j] = val*(maxY-minY) + minY
        }
    }
    fmt.Println("Predicciones testing:", descaledTesting)


    // Desescalar las predicciones
    descaledPredictions := make(dnn.Frame, len(predictions))
    for i, pred := range predictions {
        descaledPredictions[i] = make(dnn.Vector, len(pred))
        for j, val := range pred {
            descaledPredictions[i][j] = val*(maxY-minY) + minY
        }
    }

    // Desescalar las etiquetas de prueba
    descaledTestY := make(dnn.Frame, len(testYFrame))
    for i, val := range testYFrame {
        descaledTestY[i] = make(dnn.Vector, len(val))
        for j, v := range val {
            descaledTestY[i][j] = v*(maxY-minY) + minY
        }
    }

    //fmt.Println("Predicciones:", descaledPredictions)
    //fmt.Println("Etiquetas reales:", descaledTestY)

    mae := dnn.MeanAbsoluteError(descaledPredictions, descaledTestY)
    mse := dnn.MeanSquaredError(descaledPredictions, descaledTestY)
    r2 := dnn.R2Score(descaledPredictions, descaledTestY)

    fmt.Printf("Mean Absolute Error: %f\n", mae)
    fmt.Printf("Mean Squared Error: %f\n", mse)
    fmt.Printf("R² Score: %f\n", r2)
}