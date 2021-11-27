import pickle
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

os.environ['SKLEARN_CPP_MIN_LOG_LEVEL'] = '3'



def clean3(text):
    text = text.lower()
    t = ""

    if len(text) > 3:
        text = text[0:text.index(".")]

        for i in range(len(text) - 2):
            t += text[i:i + 3] + " "

        t += "\n"

    return t


def one_site_check(domain):
    _domain = []
    filename_path = 'multi_sgd/modelV2021isp_multi2.sav'
    model = pickle.load(open(filename_path, 'rb'))
    _domain.append(clean3(domain))
    score = model.predict_proba(_domain)
    temp = int(model.predict(_domain)[0])

    if temp == 0:
        val = 'legit'

    elif temp == 1:
        val = 'crypt'

    elif temp == 2:
        val = 'goz'

    elif temp == 3:
        val = 'newgoz'

    return f"Subclass: {val} \nScore: {np.max(score)}"


def model_eval_sgd(model_path, x, y_label, threshold=0.5):
    result = {}
    model = pickle.load(open(model_path, 'rb'))
    predicted = model.predict(x)
    predicted = predicted.astype('int32')
    result['cm'] = confusion_matrix(y_label, predicted)
    y_prob = model.predict_proba(x)
    result['accuracy_score'] = accuracy_score(y_true=y_label, y_pred=predicted)
    result['roc_auc_score'] = roc_auc_score(y_label, y_prob, multi_class='ovo')

    return result
