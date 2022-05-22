# EarthQuake Seismic Classification

# 安装

pip install -r requirements.txt


# 2D shape data

6000 -> 3x50x40

## resnet

```
python resnet_test.py   or python main_all.py resnet
```

precision recall  F1 score in micro: (0.9830611010284331, 0.9830611010284331, 0.9830611010284331

## vgg

```
python main_all.py vgg
```

precision recall  F1 score in micro: (0.9742891712038717, 0.9742891712038717, 0.9742891712038717

## inception

```
python main_all.py inception   
```

precision recall  F1 score in micro: (0.9686932849364791, 0.9686932849364791, 0.9686932849364791

## AlexNet 和 LeNet

只支持特定shape的数据， 不能支持

# MiniRocket

```
python minirocket_pytorch.py
```

precision recall  F1 score in micro: (0.9783726557773744, 0.9783726557773744, 0.9783726557773744,
