# Presentation (german)
https://sites.google.com/view/alpha-mind-games/startseite/projekt-helm/einleitung
## Create Virtual Environment
```bash
    python3 -m venv .venv
    source .venv/bin/activate
```

## Install Dependencies
```python3 -m pip -r install requirements.txt```

## Setup data
Single ``.csv``can be combined via the ``.concatCSV.sh``script located in the root directory.
This script will concat all ``.csv``files in the ``/data`` folder and output a file named
``all.csv``. This file will most likely be the input parameter for the training process.

In order to predict and validate the trained model, a separated validation data file should be placed 
outside the ``/data`` folder to not be included in the combined .csv file.

## Start program
### train
```python3 src/main.py -t data/all.csv```

Preprocessed data will be saved as ``.npy`` files in the ``data`` folder. This will save
a lot of computation time in successive training runs (in case mode parameters got changed).
If a different data set should be used, the user has to remove those files in order to generate
new ``.npy``files.

### predict
```python3 src/main.py -p data/toPredict.csv -m models/fullTrained1212.h5```

### live (not implemented yet, functionality is given by the other project)
```python3 src/main.py -l -m models/fullTrained1212.h5```
