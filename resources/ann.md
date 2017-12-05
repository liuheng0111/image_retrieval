#### 近似最近邻搜索 Approximate Nearest Neighbor Search(ANN)

**基于树的方法**
- KD 树
空间纬度比较低的时候，KD 树比较高效，当空间纬度较高，可以采用哈希或矢量量化的方法
- BK 树

**基于哈希的方法**
哈希：将连续的实值散列化为 0、1 的离散值
监督、无监督和半监督
评估：可以使用 knn 得到的近邻作为 ground truth，也可以使用样本自身的类别作为 ground truth
[** https://github.com/FALCONN-LIB/FALCONN/wiki/LSH-Primer局部敏感哈希 LSH]()

**基于矢量量化的方法**

#### Reference
1. [图像检索：再叙 ANN Search](http://yongyuan.name/blog/ann-search.html?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io)

