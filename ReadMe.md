# EarthQuake Seismic Classification

# 安装

pip install -r requirements.txt

# 数据划分

训练:验证:测试  0.63: 0.07:0.3

(13883, 3, 50, 40) (1543, 3, 50, 40) (6612, 3, 50, 40)


# 2D shape data

6000 -> 3x50x40



## resnet

```
python resnet_test.py   or python main_all.py resnet
```

30 epochs takes 210.65927863121033 seconds 0:03:30.659279  by GPU

precision recall  F1 score in micro: (0.9830611010284331, 0.9830611010284331, 0.9830611010284331

Note: specify 指定 GPU id  CUDA_VISIBLE_DEVICES=4   4 is the GPU id you want to use

```
CUDA_VISIBLE_DEVICES=0   python resnet_test.py
```
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

30 epoch training time takes: 0:00:32.0839

total time(feature init + 30 epochs training + evaluation) takes 62 seconds,  0:01:02.968496

## MiniRocket + KMeans

```
python minirocket_un.py
```

Test accuracy 0.7277323998444185

total time(feature init + KMeans training + evaluate cluster accuracy) takes 95 seconds,  0:01:35.068857

Much slower, because sklearn KMeans uses multi-cores CPUs. So please use servers, instead of your own PC/laptop 不要用自己的电脑，容易卡死


# Observations

## Normalization 

Normalization (in data_util.py) is crucial for resnet_test.py/main_all.py (98% -> 50%).

But MiniRocket works well without normalization(supervised: 97.8% -> 97.0%,  unsupervised: 72.7% -> 68.7%)
