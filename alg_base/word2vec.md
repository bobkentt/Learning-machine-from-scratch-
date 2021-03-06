# word2vec
## 什么是word2vec
word2vec 是 Google 在 2013 年年中开源的一款将词表征为实数值向量的高效 工具，采用的模型有 CBOW(Continuous Bag-Of-Words，即连续的词袋模型)和 Skip-Gram 两种。word2vec 代码链接为:https://code.google.com/p/word2vec/， 遵循 Apache License 2.0 开源协议，是一种对商业应用友好的许可，当然需要充 分尊重原作者的著作权。

word2vec 一般被外界认为是一个 Deep Learning(深度学习)的模型，究其原因，可能和 word2vec 的作者 Tomas Mikolov 的 Deep Learning 背景以及 word2vec 是一种神经网络模型相关，但我们谨慎认为该模型层次较浅，严格来说还不能算 是深层模型。当然如果word2vec上层再套一层与具体应用相关的输出层，比如 Softmax，此时更像是一个深层模型。

word2vec 通过训练，可以把对文本内容的处理简化为 K 维向量空间中的向量 运算，而向量空间上的相似度可以用来表示文本语义上的相似度。因此，word2vec 输出的词向量可以被用来做很多 NLP 相关的工作，比如聚类、找同义词、词性分 析等等。而 word2vec 被人广为传颂的地方是其向量的加法组合运算(Additive Compositionality )， 官 网 上 的 例 子 是 : vector('Paris') - vector('France') + vector('Italy')≈vector('Rome')，vector('king') - vector('man') + vector('woman')≈ vector('queen')。但我们认为这个多少有点被过度炒作了，很多其他降维或主题 模型在一定程度也能达到类似效果，而且 word2vec 也只是少量的例子完美符合 这种加减法操作，并不是所有的 case 都满足。

word2vec 大受欢迎的另一个原因是其高效性，Mikolov 在论文[2]中指出一个 优化的单机版本一天可训练上千亿词。

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/practice/pic/word2vec.png)

## 背景知识
### 词向量
1. [One-hot representation](https://github.com/bobkentt/Learning-machine-from-scratch-/blob/master/alg_base/One-hot-Representation.md)
2. [Distributed Representation](https://github.com/bobkentt/Learning-machine-from-scratch-/blob/master/alg_base/Distributed_Representation.md)
### 主要思想
**说起来word2vec，其实就是把词映射成一定维度的稠密向量，同时保持住词和词之间的关联性，主要体现在(欧式)距离的远近上。**

### 统计语言模型
传统的统计语言模型是表示语言基本单位(一般为句子)的概率分布函数， 这个概率分布也就是该语言的生成模型。一般语言模型可以使用各个词语条件概 率的形式表示:

𝑝(s) = 𝑝(𝑤1𝑇) = 𝑝(𝑤1, 𝑤2,... , 𝑤𝑇)=∏𝑇𝑡=1 𝑝( 𝑤𝑡|Context)

其中 Context 即为上下文，根据对 Context 不同的划分方法，可以分为五大类: 

#### (1)上下文无关模型(Context=NULL) 
该模型仅仅考虑当前词本身的概率，不考虑该词所对应的上下文环境。这是一种最简单，易于实现，但没有多大实际应用价值的统计语言模型。

𝑝(𝑤𝑡|Context)= 𝑝(𝑤𝑡)=𝑁𝑤𝑡 / 𝑁

这个模型不考虑任何上下文信息，仅仅依赖于训练文本中的词频统计。它是 n-gram 模型中当 n=1 的特殊情形，所以有时也称作 Unigram Model(一元文法统计模型)。实际应用中，常被应用到一些商用语音识别系统中。

#### (2)n-gram 模型(Context= 𝑤𝑡−n+1, 𝑤𝑡−n+2,... , 𝑤𝑡−1)
n=1 时，就是上面所说的上下文无关模型，这里 n-gram 一般认为是 N>=2 是 的上下文相关模型。当 n=2 时，也称为 Bigram 语言模型，直观的想，在自然语 言中 “白色汽车”的概率比“白色飞翔”的概率要大很多，也就是 p(汽车|白色)> p(飞翔|白色)。n>2 也类似，只是往前看 n-1 个词而不是一个词。

一般 n-gram 模型优化的目标是最大 log 似然，即:

∏𝑇𝑡=1 𝑝𝑡( 𝑤𝑡|𝑤𝑡−n+1, 𝑤𝑡−n+2,... , 𝑤𝑡−1)log𝑝𝑚(𝑤𝑡|𝑤𝑡−n+1, 𝑤𝑡−n+2,... , 𝑤𝑡−1)

n-gram 模型的优点包含了前 N-1 个词所能提供的全部信息，这些信息对当前词出现具有很强的约束力。同时因为只看 N-1 个词而不是所有词也使得模型的效率较高。

n-gram 语言模型也存在一些问题:

1. n-gram 语言模型无法建模更远的关系，语料的不足使得无法训练更高阶的 语言模型。大部分研究或工作都是使用 Trigram，就算使用高阶的模型，其统计 到的概率可信度就大打折扣，还有一些比较小的问题采用 Bigram。
2. 这种模型无法建模出词之间的相似度，有时候两个具有某种相似性的词， 如果一个词经常出现在某段词之后，那么也许另一个词出现在这段词后面的概率 也比较大。比如“白色的汽车”经常出现，那完全可以认为“白色的轿车”也可 能经常出现。
3. 训练语料里面有些 n 元组没有出现过，其对应的条件概率就是 0，导致计 算一整句话的概率为 0。解决这个问题有两种常用方法:

* 方法一为平滑法。最简单的方法是把每个 n 元组的出现次数加 1，那么原来 出现 k 次的某个 n 元组就会记为 k+1 次，原来出现 0 次的 n 元组就会记为出现 1 次。这种也称为 Laplace 平滑。当然还有很多更复杂的其他平滑方法，其本质都 是将模型变为贝叶斯模型，通过引入先验分布打破似然一统天下的局面。而引入 先验方法的不同也就产生了很多不同的平滑方法。

* 方法二是回退法。有点像决策树中的后剪枝方法，即如果 n 元的概率不到， 那就往上回退一步，用 n-1 元的概率乘上一个权重来模拟。

#### (3)n-pos模型(Context= 𝑐(𝑤𝑡−n+1),𝑐(𝑤𝑡−n+2),...,𝑐(𝑤𝑡−1))
严格来说 n-pos 只是 n-gram 的一种衍生模型。n-gram 模型假定第 t 个词出现 概率条件依赖它前 N-1 个词，而现实中很多词出现的概率是条件依赖于它前面词 的语法功能的。n-pos 模型就是基于这种假设的模型，它将词按照其语法功能进 行分类，由这些词类决定下一个词出现的概率。这样的词类称为词性 (Part-of-Speech，简称为 POS)。n-pos 模型中的每个词的条件概率表示为:

𝑝(s) = 𝑝(𝑤1𝑇)= 𝑝(𝑤1,𝑤2,...,𝑤𝑇)=∏𝑇𝑡=1𝑝(𝑤𝑡|𝑐(𝑤𝑡−n+1),𝑐(𝑤𝑡−n+2),...,𝑐(𝑤𝑡−1)) 

c 为类别映射函数，即把 T 个词映射到 K 个类别。

实际上 n-Pos使用了一种聚类的思想，使得原来 n-gram 中𝑤𝑡−n+1, 𝑤𝑡−n+2,... , 𝑤𝑡−1中的可能为

TN-1 减少到𝑐(𝑤𝑡−n+1),𝑐(𝑤𝑡−n+2),...,𝑐(𝑤𝑡−1)中的 KN-1，同时这种减少还采用了语义有意义的类别。

#### (4)基于决策树的语言模型
上面提到的上下文无关语言模型、n-gram 语言模型、n-pos 语言模型等等，
都可以以统计决策树的形式表示出来。而统计决策树中每个结点的决策规则是一 个上下文相关的问题。这些问题可以是“前一个词时 w 吗?”“前一个词属于类 别 ci 吗?”。当然基于决策树的语言模型还可以更灵活一些，可以是一些“前一 个词是动词?”，“后面有介词吗?”之类的复杂语法语义问题。

基于决策树的语言模型优点是:分布数不是预先固定好的，而是根据训练预 料库中的实际情况确定，更为灵活。缺点是:构造统计决策树的问题很困难，且 时空开销很大。

#### (5)最大熵模型
最大熵原理是E.T. Jayness于上世纪50年代提出的，其基本思想是:对一个 随机事件的概率分布进行预测时，在满足全部已知的条件下对未知的情况不做任 何主观假设。从信息论的角度来说就是:在只掌握关于未知分布的部分知识时， 应当选取符合这些知识但又能使得熵最大的概率分布。
𝑝(w|Context) =𝑒∑𝑖 𝜆𝑖𝑓𝑖(𝑐𝑜𝑛𝑡𝑒𝑥𝑡,𝑤) 𝑍(𝐶𝑜𝑛𝑡𝑒𝑥𝑡)
其中𝜆𝑖是参数，𝑍(𝐶𝑜𝑛𝑡𝑒𝑥𝑡)为归一化因子，因为采用的是这种 Softmax 形式， 所以最大熵模型有时候也称为指数模型。
#### (6)自适应语言模型
前面的模型概率分布都是预先从训练语料库中估算好的，属于静态语言模型。 而自适应语言模型类似是 Online Learning 的过程，即根据少量新数据动态调整模 型，属于动态模型。在自然语言中，经常出现这样现象:某些在文本中通常很少 出现的词，在某一局部文本中突然大量地出现。能够根据词在局部文本中出现的 情况动态地调整语言模型中的概率分布数据的语言模型成为动态、自适应或者基 于缓存的语言模型。通常的做法是将静态模型与动态模型通过参数融合到一起， 这种混合模型可以有效地避免数据稀疏的问题。

还有一种主题相关的自适应语言模型，直观的例子为:专门针对体育相关内 容训练一个语言模型，同时保留所有语料训练的整体语言模型，当新来的数据属 于体育类别时，其应该使用的模型就是体育相关主题模型和整体语言模型相融合 的混合模型。

## NNLM
NNLM 是 Neural Network Language Model 的缩写，即神经网络语言模型。神 经网络语言模型方面最值得阅读的文章是 Deep Learning 人物Bengio 的《A Neural Probabilistic Language Model》，JMLR 2003。

NNLM 采用的是 Distributed Representation，即每个词被表示为一个浮点向量。 其模型图如下:

![](https://pic3.zhimg.com/61f31ca272dcd40ad0a3d05efc3172a2_b.jpg)


> 本文参考blog[Deep Learning实战之word2vec](http://techblog.youdao.com/?p=915#LinkTarget_699)
