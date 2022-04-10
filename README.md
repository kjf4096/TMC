# TRUSTED MULTI-VIEW CLASSIFICATION

## 1. 简介

多视图分类（MVC）通常侧重于改进分类通过使用来自不同视图的信息，通常将它们集成到下游任务的统一综合表示。然而，它也是动态评估不同样本的视图质量至关重要，以便提供可靠的不确定性估计，表明预测是否可以可信的。为此，提出了一种新的多视图分类方法，称为可信多视图分类，为多视图分类提供了新的范例通过在证据层面上动态整合不同观点进行学习。这个该算法联合使用多个视图来提高分类的可靠性通过整合来自每个视图的证据，提高可靠性。为了实现这一点Dirichlet分布用于模拟类概率的分布，通过不同视角的证据进行参数化，并与Dempster集成-沙弗理论。统一的学习框架会导致精确的不确定性和不确定性因此，该模型具有可靠性和鲁棒性样品。大量的实验结果验证了该方法的有效性该模型具有较高的准确性、可靠性和鲁棒性。

##2. 数据集和复现精度

* 数据集大小：该数据集包含从荷兰实用地图集合中提取的手写数字（“0”--“9”）的特征。每类 200 个图案（总共 2,000 个图案）已被数字化为二进制图像。

* 数据集下载链接:https://aistudio.baidu.com/aistudio/datasetdetail/137482

* 数据格式：该数据集每个样本包含六个特征：

** mfeat-fou：字符形状的 76 个傅立叶系数；

** mfeat-fac：216个剖面相关性；

** mfeat-kar：64个Karhunen-Love系数；

** mfeat-pix：2 x 3 窗口中的 240 像素平均值；

** mfeat-zer：47 个 Zernike 时刻；

** mfeat-mor：6个形态特征。

* 复现精度：（学习率设为0.003）

Task|	Paper（ACC）|	source code(ACC)|	this repo(ACC)
handwritten_6views	|98.51+-0.15|	98.25|	98.50
