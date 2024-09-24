package main

import (
	"fmt"
	ann "tp/models/ann"
	"tp/panditas"
)

func main() {
	df, _ := panditas.ReadCSV("../dataset/dataset_clean.csv")

	features, label, _ := df.GetFeaturesAndLabels("target")

	// var model *ann.ANN
	// fmt.Println(label)

	model := ann.ANNConcurrent(features, label)

	pred := model.ForwardConcurrent(features[0])
	fmt.Println("Predicción:", pred[0])
}
