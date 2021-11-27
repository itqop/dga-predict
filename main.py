import pandas as pd
import numpy as np
import warnings
import os
import tensorflow as tf
import sys
import re
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, average_precision_score, \
    roc_auc_score
from multi_sgd.predict import one_site_check
from multi_sgd.prepareone import preprocessing_csv
from multi_sgd.predict import model_eval_sgd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def warn(*args, **kwargs):
    pass


warnings.warn = warn

dictinary = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
             '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13,
             'e': 14, 'f': 15, 'g': 16, 'h': 17, 'i': 18, 'j': 19, 'k': 20,
             'l': 21, 'm': 22, 'n': 23, 'o': 24, 'p': 25, 'q': 26, 'r': 27,
             's': 28, 't': 29, 'u': 30, 'v': 31, 'w': 32, 'x': 33, 'y': 34,
             'z': 35, '!': 36, '-': 37, '.': 38, '_': 39, 'S': 40}


def _pad(domain, max_length=63):
    if len(domain) < max_length:
        domain = '!' * (max_length - len(domain)) + domain

    return domain


# Функция по преобразованию домена
def pad_domain(data, pad_fn=_pad):
    data['domain'] = data['domain'].map(pad_fn)

    return data


def _split_domain(domain, sep='?'):
    result = []

    for ch in domain:
        result.append(ch)

    result = list(map(lambda x: x.lower(), result))

    return sep.join(result)


def split_domain(data, split_fn=_split_domain, sep='?'):
    data = pd.concat([pd.DataFrame(data['domain'].map(split_fn).values), data['subclass']], axis=1)
    data.columns = ['domain', 'subclass']

    cols = ['domain%d' % d for d in range(0, 63)]
    data[cols] = data['domain'].str.split(sep, expand=True)
    data = data[cols + ['subclass']]

    return data


def id_encoding(data, dictionary, embeding_cols):
    data[embeding_cols] = data[embeding_cols].apply(lambda x: x.map(dictionary))
    data = data[embeding_cols + ['subclass']]

    return data


def resd(model_path, x, threshold=0.5):
    model = tf.keras.models.load_model(model_path)
    y_pred_value = model.predict(x=x)
    y_pred = np.where(y_pred_value > threshold, 1, 0)

    return y_pred


def resd_csv(model_path, x, y_label, threshold=0.5):
    result = {}
    model = tf.keras.models.load_model(model_path)

    y_pred_value = model.predict(x=x)
    y_pred = np.where(y_pred_value > threshold, 1, 0)
    result['cm'] = confusion_matrix(y_label, y_pred)
    result['f1_score'] = f1_score(y_label, y_pred)
    result['precision_score'] = precision_score(y_label, y_pred)
    result['recall_score'] = precision_score(y_label, y_pred)
    result['average_precision_score'] = average_precision_score(y_label, y_pred)
    result['roc_auc_score'] = roc_auc_score(y_label, y_pred)

    return result


def prepair_test_data(x, model='gru') -> pd.DataFrame:
    return helped(pd.read_csv(x) if '.csv' in x and model != 'sgd'
                  else [x] if '.csv' in x
                  else x, model=model)


def helped(data: str or pd.DataFrame, model='gru') -> pd.DataFrame:
    if isinstance(data, str):
        if model == 'sgd':
            return one_site_check(data)

        data_temp = re.sub(r'\.[a-z]*', '', data)
        temp = split_domain(pad_domain((pd.DataFrame({'domain': [data_temp], 'subclass': [0]}))))
        em_cols = ['domain%d' % d for d in range(0, 63)]
        temp = id_encoding(temp, dictinary, em_cols)
        x = temp.iloc[:, 0:63].values.astype(np.float32)

        if model == 'gru':
            res = resd(r'./checkpoint', x)[0][0]

        elif model == 'lstm':
            res = resd(r'./checkpointLSTM', x)[0][0]

        return pd.DataFrame({'domain': [data], 'subclass': ['dga' if res else 'legit']})

    elif isinstance(data, pd.DataFrame):
        data['domain'] = data['domain'].apply(lambda x: re.sub(r'\.[a-z]*', '', x))
        data_temp = split_domain(pad_domain(data))
        em_cols = ['domain%d' % d for d in range(0, 63)]
        data_temp = id_encoding(data_temp, dictinary, em_cols)
        x = data_temp.iloc[:, 0:63].values.astype(np.float32)
        y = data_temp.iloc[:, -1].values

        if model == 'gru':
            result = resd_csv(r'./checkpoint', x, y)

        elif model == 'lstm':
            result = resd_csv(r'./checkpointLSTM', x, y)

        return result

    elif isinstance(data, list):
        buff = preprocessing_csv(data[0])

        return model_eval_sgd('multi_sgd/modelV2021isp_multi2.sav', buff[0], buff[1])

    raise NotImplementedError(
        print("Неправильный ввод"),
    )


if __name__ == '__main__':
    try:
        domain = sys.argv[1]
        model = sys.argv[2]

    except:
        print("usage: main.py <(choose from domain path to csv)> <(choose from 'gru', 'sgd', 'lstm')>")
        domain = 'mtuci.ru'
        model = 'lstm'

    if model not in ('gru', 'lstm', 'sgd'):
        print("Такой модели не существует")

    else:
        print(prepair_test_data(domain, model=model))
