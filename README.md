# Finetuning for Sarcasm Detection using a Pruned Dataset

## Models
All the models are available in `Models` directory. In addition to models reported in the paper, we have implented `XLNet` and `Electra` models, which are available in in the models directory.

To run the models, first make sure to install the requirements using below command:
```
pip install -r requirements.txt
```
Then run each models by using calling `python` for the model name like:
```
python ./Models/model-name/model-name.py
```

Download SARC (train-balanced-sarcasm.csv) from: https://www.kaggle.com/datasets/sherinclaudia/sarcastic-comments-on-reddit

## To Run Inference on SARC
```
python ./Models/RoBERTa/RoBERTa_Reddit_Ranking.py
```

## To Reproduce best results in the paper
```
python ./Models/RoBERTa/RoBERTa.py
```

Codebase is fork of: https://github.com/AmirAbaskohi/SemEval2022-Task6-Sarcasm-Detection ; Read their paper: https://arxiv.org/pdf/2204.08198.pdf
