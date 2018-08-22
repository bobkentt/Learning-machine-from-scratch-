# 一些自动特征工程和模型选择和超参数优化的工具

![](http://inews.gtimg.com/newsapp_bt/0/4935913359/641)

## 特征工程
1. [Featuretools](https://github.com/Featuretools/featuretools)
> Featuretools是一个自动特征工程的工具，它可以根据一组相关的表来自动构造特征。这个开源库是一个面向企业客户的商用前端服务的子集。
2. [Boruta-py](https://github.com/scikit-learn-contrib/boruta_py)
> Boruta-py是Brouta特征降维策略的一种实现，以“全相关”方式来解决问题。
3. [Categorical-encoding](https://github.com/scikit-learn-contrib/categorical-encoding)
> 这个库扩展了许多分类编码方法，可作为Scikit-learn中数据转换器的接口。
它还实现了常见的分类编码方法，如one-hot编码和hash编码，以及很多niche编码方法（包括base n编码和target编码）。
4. [Tsfresh]
(https://github.com/blue-yonder/tsfresh)
> 这个库专注于时间序列数据的特征生成，它由一个德国零售分析公司支持，是他们数据分析流程中的一步。

## 超参数优化
1. [Skopt]
(https://scikit-optimize.github.io/)
> Skopt是一个超参数优化库，包括随机搜索、贝叶斯搜索、决策森林和梯度提升树。
2. [Hyperopt]
(https://github.com/hyperopt/hyperopt-sklearn)
> Hyperopt是一个超参数优化库，针对具有一定条件或约束的搜索空间进行调优，其中包括随机搜索和Tree Parzen Estimators（贝叶斯优化的变体）等算法。

3. [Ray.tune]
(https://github.com/ray-project/ray/tree/master/python/ray/tune)
> Ray.tune是一个超参数优化库，主要适用于深度学习和强化学习模型。它结合了许多先进算法，如Hyperband算法（最低限度地训练模型来确定超参数的影响）、基于群体的训练算法（Population Based Training，在共享超参数下同时训练和优化一系列网络）、Hyperopt方法和中值停止规则（如果模型性能低于中等性能则停止训练）。


## 全流程解决方案
1. [ATM]
(https://github.com/HDI-Project/ATM)
> Auto-Tune Models是麻省理工学院HDI项目开发出的框架，可用于机器学习模型的快速训练，仅需很小的工作量。
它使用贝叶斯优化和Bandits库，利用穷举搜索和超参数优化来实现模型选择。要注意，ATM仅支持分类问题，也支持AWS上的分布式计算。
2. [MLBox]
(https://github.com/AxeldeRomblay/MLBox)
> MLBox是一个新出的框架，其目标是为自动机器学习提供一个最新和最先进的方法。
> 除了许多现有框架实现的特征工程外，它还提供数据采集、数据清理和训练-测试漂移检测等功能。
> 此外，它使用Tree Parzen Estimators来优化所选模型的超参数。
3. [auto_ml]
(https://github.com/ClimbsRocks/auto_ml)
> Auto_ml是一种实用工具，旨在提高从数据中获取的信息量，且无需除数据清洗外的过多工作。
该框架使用进化网格搜索算法来完成特征处理和模型优化的繁重工作。它利用其它成熟函数库（如XGBoost、TensorFlow、Keras、LightGBM和sklearn）来提高计算速度，还宣称只需最多1毫秒来实现预测，这也是这个库的亮点。
> 该框架可快速洞察数据集（如特征重要性）来创建初始预测模型。
4. [auto-sklearn]
(https://github.com/automl/auto-sklearn)
> Auto-sklearn使用贝叶斯搜索来优化机器学习流程中使用的数据预处理器、特征预处理器和分类器，并把多个步骤经过训练后整合成一个完整模型。
> 这个框架由弗莱堡大学的ML4AAD实验室编写，且其中的优化过程使用同一实验室编写的SMAC3框架完成。
> 顾名思义，这个模型实现了sklearn中机器学习算法的自动构建。Auto-sklearn的主要特点是一致性和稳定性。
5. [H2O]
(https://github.com/h2oai/h2o-3)
> H2O是一个用Java编写的机器学习平台，它和sklearn等机器学习库的使用体验相似。但是，它还包含一个自动机器学习模块，这个模块利用其内置算法来创建机器学习模型。
> 该框架对内置于H2O系统的预处理器实施穷举搜索，并使用笛卡尔网格搜索或随机网格搜索来优化超参数。
> H2O的优势在于它能够形成大型计算机集群，这使得它在规模上有所增长。它还可在python、javascript、tableau、R和Flow（web UI）等环境中使用。
6. [TPOT]
(https://github.com/EpistasisLab/tpot)
> TPOT为基于树的流程优化工具，是一种用于查找和生成最佳数据科学流程代码的遗传编程框架。TPOT和其他自动机器学习框架一样，从sklearn库中获取算法。
> TPOT的优势在于其独特的优化方法，可以提供更有效的优化流程。
> 它还包括一个能把训练好的流程直接转换为代码的工具，这对希望能进一步调整生成模型的数据科学家来说是一个主要亮点。


**原文：https://medium.com/georgian-impact-blog/automatic-machine-learning-aml-landscape-survey-f75c3ae3bbf2**
