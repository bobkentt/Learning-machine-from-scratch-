# Distributed Representation
Distributed representation 最早由 Hinton 在 1986 年提出[8]。其基本思想是 通过训练将每个词映射成 K 维实数向量(K 一般为模型中的超参数)，通过词之 间的距离(比如 cosine 相似度、欧氏距离等)来判断它们之间的语义相似度。而 word2vec 使用的就是这种 Distributed representation 的词向量表示方式。

这种向量一般长成这个样子：[0.792, −0.177, −0.107, 0.109, −0.542, ...]，也就是普通的向量表示形式。维度以50维和100维比较常见。
