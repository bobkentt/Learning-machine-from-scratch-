
# BPE

BPE，（byte pair encoder）字节对编码，也可以叫做digram coding双字母组合编码，主要目的是为了数据压缩，算法描述为字符串里频率最常见的一对字符被一个没有在这个字符中出现的字符代替的层层迭代过程。具体在下面描述。该算法首先被提出是在Philip Gage的C Users Journal的 1994年2月的文章“A New Algorithm for Data Compression”。

算法过程

这个算法个人感觉很简单，下面就来讲解下：

比如我们想编码：

aaabdaaabac

我们会发现这里的aa出现的词数最高（我们这里只看两个字符的频率），那么用这里没有的字符Z来替代aa：

```
ZabdZabac

Z=aa
```

此时，又发现ab出现的频率最高，那么同样的，Y来代替ab：

```
ZYdZYac

Y=ab

Z=aa
```

同样的，ZY出现的频率大，我们用X来替代ZY：

```
XdXac

X=ZY

Y=ab

Z=aa
```

最后，连续两个字符的频率都为1了，也就结束了。就是这么简单。

解码的时候，就按照相反的顺序更新替换即可。

参考https://cloud.tencent.com/developer/article/1089017
