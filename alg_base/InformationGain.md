

## 信息增益（Information Gain）

信息增益是按某属性分裂前的熵与按该属性分裂后的熵的差值。

### 公式

```

gain()=infobeforeSplit()–infoafterSplit()

```

信息增益选择方法有一个很大的缺陷，它总是会倾向于选择属性值多的属性，如果训练集中加一个姓名属性，假设14条记录中的每个人姓名不同，那么信息增益就会选择姓名作为最佳属性，因为按姓名分裂后，每个组只包含一条记录，而每个记录只属于一类（要么购买电脑要么不购买），因此纯度最高，以姓名作为测试分裂的结点下面有14个分支。但是这样的分类没有意义，它没有任何泛化能力。增益比率对此进行了改进，它引入一个分裂信息：

## 信息增益比（Information Gain Ratio）

信息增益比是信息增益除以按某属性分裂前的熵。

### 公式

```

GainRatio=gain() / infobeforeSplit()

```
我们找GainRatio最大的属性作为最佳分裂属性。如果一个属性的取值很多，那么infobeforeSplit()会大，从而使GainRatio变小。不过增益比率也有缺点，infobeforeSplit(D)可能取0，此时没有计算意义；且当infobeforeSplit(D)趋向于0时，GainRatio的值变得不可信，改进的措施就是在分母加一个平滑，这里加一个所有分裂信息的平均值：

 公式

```

GainRatio=gain() / infobeforeSplit() + 

```

### 基尼系数
