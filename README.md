# TRUSTED MULTI-VIEW CLASSIFICATION

## 目录

- [1. 简介]()
- [2. 数据集和复现精度]()
- [3. 准备数据与环境]()
    - [3.1 准备环境]()
    - [3.2 准备数据]()
    - [3.3 准备模型]()
- [4. 开始使用]()
    - [4.1 模型训练]()
    - [4.2 模型评估]()
    - [4.3 模型预测]()
- [5. 模型推理部署]()
    - [5.1 基于Inference的推理]()
    - [5.2 基于Serving的服务化部署]()
- [6. 自动化测试脚本]()
- [7. LICENSE]()
- [8. 参考链接与文献]()

## 1. 简介

多视图分类（MVC）通常侧重于改进分类通过使用来自不同视图的信息，通常将它们集成到下游任务的统一综合表示。然而，它也是动态评估不同样本的视图质量至关重要，以便提供可靠的不确定性估计，表明预测是否可以可信的。为此，提出了一种新的多视图分类方法，称为可信多视图分类，为多视图分类提供了新的范例通过在证据层面上动态整合不同观点进行学习。这个该算法联合使用多个视图来提高分类的可靠性通过整合来自每个视图的证据，提高可靠性。为了实现这一点Dirichlet分布用于模拟类概率的分布，通过不同视角的证据进行参数化，并与Dempster集成-沙弗理论。统一的学习框架会导致精确的不确定性和不确定性因此，该模型具有可靠性和鲁棒性样品。大量的实验结果验证了该方法的有效性该模型具有较高的准确性、可靠性和鲁棒性。

注意：在给出参考repo的链接之后，建议添加对参考repo的开发者的致谢。

**论文:** [Trusted Multi-View Classification](https://arxiv.org/abs/2102.02051)

**参考repo:** [TMC](https://github.com/hanmenghan/TMC)

在此非常感谢tmc repo的 hanmenghan等人贡献的[TMC](https://github.com/hanmenghan/TMC)，提高了本repo复现论文的效率。

**aistudio体验教程:** [地址](https://aistudio.baidu.com/aistudio/projectdetail/3756056)


## 2. 数据集和复现精度

- 数据集大小：该数据集包含从荷兰实用地图集合中提取的手写数字（“0”--“9”）的特征。每类 200 个图案（总共 2,000 个图案）已被数字化为二进制图像。
- 数据集下载链接：[**Multiple Features Data Set**](https://archive.ics.uci.edu/ml/datasets/Multiple+Features)
- 数据格式：该数据集每个样本包含六个特征：
  - mfeat-fou：字符形状的 76 个傅立叶系数；
  - mfeat-fac：216个剖面相关性；
  - mfeat-kar：64个Karhunen-Love系数；
  - mfeat-pix：2 x 3 窗口中的 240 像素平均值；
  - mfeat-zer：47 个 Zernike 时刻；
  - mfeat-mor：6个形态特征。
- 复现精度：（学习率设为0.003）

| Task               | Paper（ACC） | source code(ACC) | this repo(ACC) |
| ------------------ | ------------ | ---------------- | -------------- |
| handwritten_6views | 98.51±0.15  | 98.25            | 98.50          |



## 3. 准备数据与环境


### 3.1 准备环境

首先介绍下支持的硬件和框架版本等环境的要求，格式如下：

- 硬件：GPU: Tesla V100 Mem 16GB, CPU 2cores RAM 16GB (aistudio高级GPU)
- 框架：
  - PaddlePaddle >= 2.2.0

- 使用如下命令安装依赖：(本项目在aistudio中运行，此步骤可忽略)

```
pip install -r requirements.txt
```

### 3.2 准备数据

由于数据量较小，已经放在repo里面了，路径如下所示：

```
# 全量数据： datasets/handwritten_6views.mat
```


### 3.3 准备模型


预训练模型默认保存到output/model_best下。


## 4. 开始使用


### 4.1 模型训练

```
# python
python train.py 
```



```
The number of training images = 8
W0410 22:18:59.794243 10217 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
W0410 22:18:59.798430 10217 device_context.cc:465] device: 0, cuDNN Version: 7.6.
[Epoch 50, iter: 3] acc: 0.97250, lr: 0.00300, loss: 12.97460, avg_reader_cost: 0.00011 sec, avg_batch_cost: 0.00078 sec, avg_samples: 800.0, avg_ips: 1027386.16044 images/sec.
[Epoch 50, iter: 7] acc: 0.97500, lr: 0.00300, loss: 13.06705, avg_reader_cost: 0.00010 sec, avg_batch_cost: 0.00051 sec, avg_samples: 1600.0, avg_ips: 3143272.31850 images/sec.
[Epoch 100, iter: 3] acc: 0.98875, lr: 0.00300, loss: 11.41495, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.00081 sec, avg_samples: 800.0, avg_ips: 986895.05882 images/sec.
[Epoch 100, iter: 7] acc: 0.98375, lr: 0.00300, loss: 11.45024, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.00071 sec, avg_samples: 1600.0, avg_ips: 2241445.02338 images/sec.
epoch: 100 ,test_loss:11.64425, test_acc: 0.9825
[Epoch 150, iter: 3] acc: 0.98500, lr: 0.00300, loss: 10.83840, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.00089 sec, avg_samples: 800.0, avg_ips: 899341.51702 images/sec.
[Epoch 150, iter: 7] acc: 0.98750, lr: 0.00300, loss: 10.57193, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.00078 sec, avg_samples: 1600.0, avg_ips: 2060450.23027 images/sec.
[Epoch 200, iter: 3] acc: 0.98750, lr: 0.00300, loss: 10.05518, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.00080 sec, avg_samples: 800.0, avg_ips: 997456.36147 images/sec.
[Epoch 200, iter: 7] acc: 0.98875, lr: 0.00300, loss: 10.26580, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.00066 sec, avg_samples: 1600.0, avg_ips: 2430599.92756 images/sec.
epoch: 200 ,test_loss:10.51799, test_acc: 0.9825
[Epoch 250, iter: 3] acc: 0.99000, lr: 0.00300, loss: 9.51728, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.00081 sec, avg_samples: 800.0, avg_ips: 988348.51252 images/sec.
[Epoch 250, iter: 7] acc: 0.98875, lr: 0.00300, loss: 9.52120, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.00070 sec, avg_samples: 1600.0, avg_ips: 2277192.53478 images/sec.
[Epoch 300, iter: 3] acc: 0.99000, lr: 0.00300, loss: 8.99130, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.00081 sec, avg_samples: 800.0, avg_ips: 985445.87372 images/sec.
[Epoch 300, iter: 7] acc: 0.99000, lr: 0.00300, loss: 9.22736, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.00070 sec, avg_samples: 1600.0, avg_ips: 2284945.99932 images/sec.
epoch: 300 ,test_loss:10.00660, test_acc: 0.9825
[Epoch 350, iter: 3] acc: 0.98875, lr: 0.00300, loss: 8.77700, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.00076 sec, avg_samples: 800.0, avg_ips: 1053514.34851 images/sec.
[Epoch 350, iter: 7] acc: 0.99125, lr: 0.00300, loss: 9.04363, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.00065 sec, avg_samples: 1600.0, avg_ips: 2478170.75332 images/sec.
[Epoch 400, iter: 3] acc: 0.99125, lr: 0.00300, loss: 9.03489, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.00074 sec, avg_samples: 800.0, avg_ips: 1078227.24936 images/sec.
[Epoch 400, iter: 7] acc: 0.99187, lr: 0.00300, loss: 9.06323, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.00065 sec, avg_samples: 1600.0, avg_ips: 2458200.14652 images/sec.
epoch: 400 ,test_loss:9.74508, test_acc: 0.9825
[Epoch 450, iter: 3] acc: 0.99125, lr: 0.00300, loss: 8.55426, avg_reader_cost: 0.00013 sec, avg_batch_cost: 0.00075 sec, avg_samples: 800.0, avg_ips: 1059836.76563 images/sec.
[Epoch 450, iter: 7] acc: 0.99250, lr: 0.00300, loss: 8.42972, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.00064 sec, avg_samples: 1600.0, avg_ips: 2488278.23508 images/sec.
[Epoch 500, iter: 3] acc: 0.99000, lr: 0.00300, loss: 8.92419, avg_reader_cost: 0.00014 sec, avg_batch_cost: 0.00077 sec, avg_samples: 800.0, avg_ips: 1035311.07683 images/sec.
[Epoch 500, iter: 7] acc: 0.99313, lr: 0.00300, loss: 8.60470, avg_reader_cost: 0.00015 sec, avg_batch_cost: 0.00070 sec, avg_samples: 1600.0, avg_ips: 2286503.03237 images/sec.
epoch: 500 ,test_loss:9.60446, test_acc: 0.9850
====> acc: 0.9850
```

### 4.2 模型评估

```
!python eval.py # 1个batch
```

```
The number of training images = 8
W0411 10:35:22.982743  7446 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
W0411 10:35:22.987826  7446 device_context.cc:465] device: 0, cuDNN Version: 7.6.
loss: 6.8701171875 acc: 0.9900
```

### 4.3 模型预测

```
!python predict.py
```

```
The number of training images = 8
W0411 10:37:17.383888  7620 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
W0411 10:37:17.389173  7620 device_context.cc:465] device: 0, cuDNN Version: 7.6.
====> acc: 0.9850
```


## 5. 模型推理部署

如果repo中包含该功能，可以按照Inference推理、Serving服务化部署再细分各个章节，给出具体的使用方法和说明文档。


## 6. 自动化测试脚本

介绍下tipc的基本使用以及使用链接


## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 8. 参考链接与文献
