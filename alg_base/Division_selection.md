# 决策树的划分选择(Division selection)

决策树学习的关键在于如何选择最优的划分属性。

一般而言，随着划分的不断进行，我们希望决策树的分支结点所包含的样本尽可能属于同一类别（即结点的纯度(purity)越来越高）。

判断结点的纯度的主要方法：
* [信息增益](https://github.com/bobkentt/Learning-machine-from-scratch-/blob/master/alg_base/InformationGain.md)；
* [增益比](https://github.com/bobkentt/Learning-machine-from-scratch-/blob/master/alg_base/InformationGainRadio.md)；
* [基尼指数](https://github.com/bobkentt/Learning-machine-from-scratch-/blob/master/alg_base/GiniIndex.md)。

## 主要算法
决策树的划分选择主要包括ID3、C4.5和CART。
* ID3决策树算法使用信息增益判断纯度；
* C4.5决策树使用增益比
* CART决策树使用基尼指数
