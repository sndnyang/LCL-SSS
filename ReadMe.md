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

- precision recall  F1 score in micro: (0.9774652147610405, 0.9774652147610405, 0.9774652147610405, None)
- precision recall  F1 score in macro: (0.9742350942479397, 0.9784838179920086, 0.9762684751952175, None)
- precision recall  F1 score in weighted: (0.9777493925150258, 0.9774652147610405, 0.9775278578108789, None)

30 epochs takes 210.65927863121033 seconds 0:03:30.659279  by GPU

Note: specify 指定 GPU id  CUDA_VISIBLE_DEVICES=4   4 is the GPU id you want to use

Note: about the micro and macro, weighted modes: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html

```
CUDA_VISIBLE_DEVICES=0   python resnet_test.py
```
## vgg

```
python main_all.py vgg
```

- precision recall  F1 score in micro: (0.9807924984875983, 0.9807924984875983, 0.9807924984875983, None)
- precision recall  F1 score in macro: (0.9807680111517341, 0.979085444315043, 0.9799133389389202, None)
- precision recall  F1 score in weighted: (0.9808440645335728, 0.9807924984875983, 0.9808039652414201, None)

30 epoch takes 228 seconds,  0:03:48.908452

## inception

```
python main_all.py inception   
```

- precision recall  F1 score in micro: (0.9640048396854205, 0.9640048396854205, 0.9640048396854205, None)
- precision recall  F1 score in macro: (0.9675209970511508, 0.9558780226812885, 0.9610584015381325, None)
- precision recall  F1 score in weighted: (0.9643438850153745, 0.9640048396854205, 0.9636412281604345, None)

30 epoch takes 155 seconds,  0:02:35.260036 by Titan RTX GPU

# MiniRocket

```
python minirocket_pytorch.py
```

- precision recall  F1 score in micro: (0.9765577737447066, 0.9765577737447066, 0.9765577737447066, None)
- precision recall  F1 score in macro: (0.9740317458166179, 0.9760858799525116, 0.975037360831401, None)
- precision recall  F1 score in weighted: (0.9766728008981629, 0.9765577737447066, 0.9765968085618346, None)


30 epoch training time takes: 0:00:32.0839

total time(feature init + 30 epochs training + evaluation) takes 62 seconds,  0:01:02.968496

## MiniRocket + KMeans

```
python minirocket_un.py
```

Valid (Train data)

- precision recall  F1 score in micro: (0.72650071308181, 0.72650071308181, 0.72650071308181, None)
- precision recall  F1 score in macro: (0.768166980467447, 0.7778092485870426, 0.7269018362922858, None)
- precision recall  F1 score in weighted: (0.8061755802259496, 0.72650071308181, 0.7177761482591168, None)


Test (unseen data)

- precision recall  F1 score in micro: (0.7280701754385965, 0.7280701754385965, 0.7280701754385965, None)
- precision recall  F1 score in macro: (0.7688228818383352, 0.7821433192910333, 0.7306720810775338, None)
- precision recall  F1 score in weighted: (0.8059558883062758, 0.7280701754385965, 0.7192791769292779, None)


total time(feature init + KMeans training + evaluate cluster accuracy) takes 100 seconds,  0:01:40.92

Much slower, because sklearn KMeans uses multi-cores CPUs. So please use servers, instead of your own PC/laptop 不要用自己的电脑，容易卡死


## MiniRocket + Gaussian Mixture

Valid(Train data)

- precision recall  F1 score in micro: (0.7281861791780111, 0.7281861791780111, 0.7281861791780111, None)
- precision recall  F1 score in macro: (0.772654993160231, 0.7804282970534618, 0.7288347746132154, None)
- precision recall  F1 score in weighted: (0.8115435180379689, 0.7281861791780111, 0.7195912425526907, None)

Test (unseen data)

- precision recall  F1 score in micro: (0.6677253478523896, 0.6677253478523896, 0.6677253478523896, None)
- precision recall  F1 score in macro: (0.4913682236442349, 0.5983473703037866, 0.5340606981630641, None)
- precision recall  F1 score in weighted: (0.5391397590979354, 0.6677253478523896, 0.5924183550697695, None)


total time(feature init + GM training + evaluate cluster accuracy) takes 341 seconds,  0:05:41.382244

# Observations

## Normalization 

Normalization (in data_util.py) is crucial for resnet_test.py/main_all.py (98% -> 50%).

But MiniRocket works well without normalization(supervised: 97.8% -> 97.0%,  unsupervised: 72.7% -> 68.7%)
