# Music Emotion Recognition 

Music Emotion Recognition is a python library for dealing with predictions of the emotion for music on the MTG-Jamendo dataset.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
pip install -r requirements.txt
```

## Train a model (Transformer)

```python
# under audio_classification/code/models folder

python trsf_mood_classification.py
```

## Evaluate a model (Transformer)

```python
# under audio_classification/code/models folder

python evaluate.py
```

## Predict a music mood 

```python
# under audio_classification/code/models folder

streamlit run api.py
```
