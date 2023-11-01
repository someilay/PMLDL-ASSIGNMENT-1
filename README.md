## PMLDL ASSIGNMENT 1
#### Ilia Milioshin, i.mileshin@innopolis.university, B20-RO-01

## Setup

* Clone repo: 
```bash
git clone https://github.com/someilay/PMLDL-ASSIGNMENT-1.git
```
* Create a virtual environment

For Unix/macOS
```bash
python3 -m venv venv
```
For Windows
```bash
py -m venv env
```
* Activating a virtual environment


For Unix/macOS
```bash
source env/bin/activate
```
For Windows
```bash
.\env\Scripts\activate
```
* Installing a packages
```bash
pip install -r requirements.txt
```
* Install [pytorch](https://pytorch.org/get-started/locally/#start-locally)
* Unzip content of [models.zip](https://drive.google.com/file/d/1j6PMjJHWQmy6-AbqHTzY9IMO1JsBAHmN/view?usp=sharing) into [models](models)

* Create datasets
```bash
python src/data/make_dataset.py
```

## Training a final model (add --help for more parameters)
```bash
python src/models/train_model.py
```

## Do de-toxification
```bash
python src/models/predict_model.py
```

## Get charts and data visualization (results would be stored [here](src/visualization))
```bash
python src/visualization/visualize.py
```
