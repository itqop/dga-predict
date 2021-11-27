import sys
import pickle
import pandas as pd
from test1 import clean3
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, average_precision_score, roc_auc_score


def aione():
    predd = []
    filename = 'models/modelV2021isp_multi2.sav'
    model = pickle.load(open(filename, 'rb'))
    pred = clean3(input())
    predd.append(pred)
    a = model.predict_proba(predd)
    temp = int(model.predict(predd)[0])
    if temp == 0:
        print("legit")
    elif temp == 1:
        print("Crypt")
    elif temp == 2:
        print("goz")
    elif temp == 3:
        print("newgoz")
    print("Score: ", end=" ")
    print(np.max(a))


def model_eval(model_path, x, y_label, threshold=0.5):
    result = {}
    model = pickle.load(open(model_path, 'rb'))
    predicted = model.predict(x)
    predicted = predicted.astype('int32')
    result['cm'] = confusion_matrix(y_label, predicted)
    #result['f1_score'] = f1_score(y_label, predicted)
    #result['precision_score'] = precision_score(y_label, predicted)
    #result['recall_score'] = recall_score(y_label, predicted)
    y_prob = model.predict_proba(x)
    result['accuracy_score'] = accuracy_score(y_true=y_label, y_pred=predicted)
    result['roc_auc_score'] = roc_auc_score(y_label, y_prob, multi_class='ovo')

    return result


if __name__ == '__main__':
    data_temp = pd.read_csv('datasets/testmultiv2.csv')
    x = data_temp.iloc[:, 0].values
    y = data_temp.iloc[:, -1].values
    filename = 'models/modelV2021isp_multi2.sav'
    res = model_eval(filename, x, y)
    for (x, y) in res.items():
        print((x, y))
    #sys.exit(aione()) # Набрать сайт вручную
