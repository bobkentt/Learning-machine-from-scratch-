# scikit-plot

scikit-plot是一个画图神器，可以轻松的花常见的：
* 混淆矩阵
* PR-曲线
* ROC曲线
* 学习曲线
* 特征重要性
* 聚类的肘点等等

参考代码
```
# 画roc
predicted_probas = best_tree.predict_proba(x_test)

import scikitplot as skplt
skplt.metrics.plot_roc(y_test, predicted_probas)
plt.show()

# 画PR曲线
skplt.metrics.plot_precision_recall_curve(y_test, predicted_probas)
plt.show()

# 求AUC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, predicted_probas[:,-1])
```

链接地址：

https://github.com/reiinakano/scikit-plot
