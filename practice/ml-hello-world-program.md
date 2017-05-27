
# ml-hello-world-program
使用开源库scikit-learn，来写机器学习的第一个hello world程序。


解决的问题是：
> Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。

scikit-learn中刚好提供Iris数据集，也可以去下面地址下载：

[Iris数据集下载地址](http://archive.ics.uci.edu/ml/datasets/Iris)

在scikit-learn中，分类的预测器是一个Python对象，来实现fit(X, y)和predict(T)方法。下面这个预测器的例子是classsklearn.svm.SVC，实现了支持向量机分类 。


```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#Author: bobkentt@163.com
#Date:

from sklearn import svm
from sklearn import datasets

def main():
    clf = svm.SVC()
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)
    print(clf.predict([X[0]]))
    exit(0)


if __name__ == "__main__":
    main()
```
