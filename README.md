The CtMGAM repository was created in order to train a model based on binary classifications of CtMGAM inhibitors, and then use the model to screen a filtered ChEMBL dataset for new inhibitors.

The easiest way to replicate the script is to download either the package manager Conda or Mamba, and then run the following line to create the necessary env for the scripts:<br>
"**conda env create -f environment.yaml**"

To run the ML model, from the **src directory** type the prompt: <br>
**python3 rf_mlmodel.py ../data/training_data.csv ../data/filtered_ChemBL.csv**<br>

If you want to run the script wih your own files, first make sure your data headers match. Then when running the rf_model.py, the first file is the training data, and the second is the file you want to screen.

The jupyter notebooks are interactive, and there are commented out sections where you will need to provide file paths. For drawing the molecules, you must specifiy a bit location you would like highlighted within the scitpt as well.
