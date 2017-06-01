# axis

axis()函数给出了形如[xmin,xmax,ymin,ymax]的列表，指定了坐标轴的范围。

## example

```
import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()
```

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/practice/pic/20170601101424.jpg)
