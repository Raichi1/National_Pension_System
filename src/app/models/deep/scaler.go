package deep

// MinMaxScaler estructura para almacenar los valores mínimo y máximo de cada característica
type MinMaxScaler struct {
	min []float64
	max []float64
}

// FitTransform calcula los valores mínimo y máximo para cada característica y escala los datos
func (scaler *MinMaxScaler) FitTransform(data [][]float64) [][]float64 {
	if len(data) == 0 {
		return nil
	}

	numFeatures := len(data[0])
	scaler.min = make([]float64, numFeatures)
	scaler.max = make([]float64, numFeatures)

	// Inicializa min y max
	for j := 0; j < numFeatures; j++ {
		scaler.min[j] = data[0][j]
		scaler.max[j] = data[0][j]
	}

	// Calcular min y max para cada característica
	for i := 1; i < len(data); i++ {
		for j := 0; j < numFeatures; j++ {
			if data[i][j] < scaler.min[j] {
				scaler.min[j] = data[i][j]
			}
			if data[i][j] > scaler.max[j] {
				scaler.max[j] = data[i][j]
			}
		}
	}

	// Escalar los datos
	scaledData := make([][]float64, len(data))
	for i := 0; i < len(data); i++ {
		scaledData[i] = make([]float64, numFeatures)
		for j := 0; j < numFeatures; j++ {
			scaledData[i][j] = (data[i][j] - scaler.min[j]) / (scaler.max[j] - scaler.min[j])
		}
	}

	return scaledData
}

// Function to Trasform
func (scaler *MinMaxScaler) Transform(data [][]float64) [][]float64 {
	if len(data) == 0 {
		return nil
	}

	scaledData := make([][]float64, len(data))
	for i := 0; i < len(data); i++ {
		scaledData[i] = make([]float64, len(scaler.min))
		for j := 0; j < len(scaler.min); j++ {
			scaledData[i][j] = (data[i][j] - scaler.min[j]) / (scaler.max[j] - scaler.min[j])
		}
	}
	return scaledData
}

// function to Trasform []float64
func (scaler *MinMaxScaler) TransformY(data []float64) []float64 {
	if len(data) == 0 {
		return nil
	}

	scaledData := make([]float64, len(data))
	for i := 0; i < len(data); i++ {
		scaledData[i] = (data[i] - scaler.min[0]) / (scaler.max[0] - scaler.min[0])
	}
	return scaledData
}

// InverseTransform revierte la transformación aplicada a los datos escalados
func (scaler *MinMaxScaler) InverseTransform(data [][]float64) [][]float64 {
	if len(data) == 0 {
		return nil
	}

	inverseData := make([][]float64, len(data))
	for i := 0; i < len(data); i++ {
		inverseData[i] = make([]float64, len(scaler.min))
		for j := 0; j < len(scaler.min); j++ {
			inverseData[i][j] = data[i][j]*(scaler.max[j]-scaler.min[j]) + scaler.min[j]
		}
	}
	return inverseData
}

// function to train with data []float64
func (scaler *MinMaxScaler) FitTransformY(data []float64) []float64 {
	if len(data) == 0 {
		return nil
	}

	scaler.min = make([]float64, 1)
	scaler.max = make([]float64, 1)

	// Inicializa min y max
	scaler.min[0] = data[0]
	scaler.max[0] = data[0]

	// Calcular min y max para cada característica
	for i := 1; i < len(data); i++ {
		if data[i] < scaler.min[0] {
			scaler.min[0] = data[i]
		}
		if data[i] > scaler.max[0] {
			scaler.max[0] = data[i]
		}
	}

	// Escalar los datos
	scaledData := make([]float64, len(data))
	for i := 0; i < len(data); i++ {
		scaledData[i] = (data[i] - scaler.min[0]) / (scaler.max[0] - scaler.min[0])
	}

	return scaledData
}
