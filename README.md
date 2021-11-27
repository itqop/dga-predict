# 4 Сессия
## Команда Cerebro: Власов Глеб, Калентьев Леон, Арустамян Александр

<!-- #region -->
## Installation guide
```sh
git clone https://github.com/L1chik/cerebro-dga
```

```sh
cd /home/l1chik/cerebro-dga
```

```sh
docker run --gpus all -p 9999:8888 -it --rm -v /home/l1chik/cerebro-dga:/app nvcr.io/nvidia/tensorflow:21.11-tf2-py3
```

/// *заменить /home/l1chik на верный путь*

## Usage
```sh
python main.py <(domain | path_to_csv)>  <( gru | lstm | sgd)> 
```

*Default params domain= google.com , model= gru*

Если указать конкретный домен программа выдаст подкласс к которому он относится.
Если указать csv файлл с размеченныи признаком subclass, программа выведет метрики своей работы. 

*LSTM & GRU - бинарные, legit=0, dga=1. SGD - мультиклассовый legit=0, cryptolocker=1, goz=2, newgoz=3*

<!-- #endregion -->
