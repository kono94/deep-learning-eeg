## Create Virtual Environment
```bash
    python3 -m venv .venv
    source .venv/bin/activate
```

## Install Dependencies
```python3 -m pip -r install requirements.txt```


## Start program
### train
```python3 src/main.py -t data/all.csv```
```python3 src/main.py --train data/all.csv```

### predict
```python3 src/main.py -p data/toPredict.csv -m models/fullTrained1212.h5```
```python3 src/main.py --predict data/toPredict.csv --model models/fullTrained1212.h5```

### live
```python3 src/main.py -l -m models/fullTrained1212.h5```