#### 基于内容的图像检索（CBIR）


#### 参考文献
1. [wikipedia: content-based image retrieval](https://en.wikipedia.org/wiki/Content-based_image_retrieval)



#### 截图识别
1. 计算图片的像素大小，设定阈值


#### Near-duplicated images detection

**感知哈希 perceptual hashing**

- **Average Hashing**


- **Perceptive Hashing(pHash)**
    - 更鲁棒

- **Difference Hashing**

计算相邻像素之间的亮度差异并确定相对梯度

- **Wavelet Hashing**

**存储结构**

- **BK-Tree**


1. [Duplicate image detection with perceptual hashing in Python / BK-Tree](http://tech.jetsetter.com/2017/03/21/duplicate-image-detection/)
2. [Perceptual hash algorithms(aHash/pHash/dHash)](http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html)
3. [pHash 开源](http://phash.org/)
4. [数据结构：BK-Tree](https://signal-to-noise.xyz/post/bk-tree/)



#### Similar images retrieval

**Content Based Image Retrieval (CBIR)**
    1. [Deep learning of binary hash codes for fast image retrieval](https://zhuanlan.zhihu.com/p/23891866)
    2. [车辆精确检索](https://github.com/iamhankai/vehicle-retrieval-kCNNs)
    3. [FaceBook AI: Billion-scale similarity search with GPUs](https://arxiv.org/pdf/1702.08734.pdf)
    4. [FaceBook AI 开源大规模 GPU 加速图片检索](https://github.com/facebookresearch/faiss)
    5. [图像检索：基于内容的图像检索技术](http://yongyuan.name/blog/cbir-technique-summary.html)
    6. [图像检索：再叙ANN Search](http://yongyuan.name/blog/ann-search.html?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io)
    7. [图像检索：layer选择与fine-tuning性能提升验证](http://yongyuan.name/blog/layer-selection-and-finetune-for-cbir.html)
    8. [图像检索 paper list](https://github.com/willard-yuan/awesome-cbir-papers)

    10. [基于 CNN 的相似图片内容检索](https://blogs.technet.microsoft.com/machinelearning/2016/11/28/deep-learning-part-4-content-based-similar-image-retrieval-using-cnn/)
    11. [Deep Sketch Hashing: Fast Free-hand Sketch-Based Image Retrieval](https://github.com/ymcidence/DeepSketchHashing)
    12. [Deep sketch hashing](https://github.com/ymcidence/DeepSketchHashing)
    13. [End-to-end Learning of Deep Visual Representations for Image Retrieval](https://arxiv.org/abs/1610.07940)
    14. [Content-based image retrieval tutorial](https://arxiv.org/abs/1608.03811)
    15. [SSDH: Semi-supervised Deep Hashing for Large Scale Image Retrieval]
    16. [CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard Examples](http://cmp.felk.cvut.cz/cnnimageretrieval/)
    17. [CVPR 2016 Tutorial: Image Tag Assignment, Refinement and Retrieval](http://www.micc.unifi.it/tagsurvey/downloads/CVPR2016-Image-Tag-Assignment-Refinement-Retrieval.pdf)
    18. [Semantic Image Retrieval via Active Grounding of Visual Situations]()
    19. [基于复杂网络的图像检索](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4667282/)
    20. [Content-Based Image Retrieval using Pivotal HD with HAWQ](https://content.pivotal.io/blog/content-based-image-retrieval-using-pivotal-hd-with-hawq)

**Survey**
- **2006**
    1. [A survey of content-based image retrieval with high-level semantics](http://www.baskent.edu.tr/~hogul/RA1.pdf)


- **2007**
    1. [Content-based Multimedia Information Retrieval: State of the Art and Challenges](http://www.ugmode.com/prior_art/lew2006cbm.pdf)
    2. [Image retrieval: Ideas, Influences, and Trends of the New Age](http://infolab.stanford.edu/~wangz/project/imsearch/review/JOUR/)

- **2016**
    1. [知乎：基于深度学习的视觉实例搜索研究进展](https://zhuanlan.zhihu.com/p/22265265)


**Reference**
- **2007**
    1. [基于 topic: Image Retrieval on Large-Scale Image Databases]
    2. [博客介绍：Content-Based Image Retrieval using Pivotal HD with HAWQ](https://content.pivotal.io/blog/content-based-image-retrieval-using-pivotal-hd-with-hawq)



- **2009**
    1. [Bundling Features for Large Scale Partial-Duplicate Web Image Search](https://www.microsoft.com/en-us/research/publication/bundling-features-for-large-scale-partial-duplicateweb-image-search/?from=http%3A%2F%2Fresearch.microsoft.com%2Fpubs%2F80803%2Fcvpr_2009_bundle.pdf)
    2. [wikipedia: Reverse image search](https://en.wikipedia.org/wiki/Reverse_image_search#cite_note-7)

- **2011**
    1. [Fundamental Media Understanding](http://www.gbv.de/dms/weimar/toc/669061174_toc.pdf)

    图片相似度定义

    2.

- **2014**
    1. [Multi-Scale Orderless Pooling of Deep Convolutional Activation Features] 将CNN特征与无序的VLAD编码方法相结合

- **2015**
    1. [Exploiting Local Features from Deep Networks for Image Retrieval]
    2. [Aggregating Deep Convolutional Features for Image Retrieval]

- **2016**
    1. [基于卷积特征的实例搜索(Bags of Local Convolutional Features for Scalable Instance Search)](https://github.com/imatge-upc/retrieval-2016-icmr)
    2. [Faster R-CNN Features for Instance Search](https://github.com/imatge-upc/retrieval-2016-deepvision)
    3. [DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations](http://personal.ie.cuhk.edu.hk/~lz013/projects/DeepFashion.html)
    4. [NetVLAD: CNN architecture for weakly supervised place recognition](http://www.di.ens.fr/willow/research/netvlad/)
    5. [Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles]
    6. [Where to Focus: Query Adaptive Matching for Instance Retrieval Using Convolutional Feature Maps]
    7. [Instance-Level Coupled Subspace Learning for Fine-Grained Sketch-Based Image Retrieval]
    9. [鉴黄、视频、图片去重、图像搜索业务分析与实践](https://yq.aliyun.com/articles/64959?utm_campaign=wenzhang&utm_medium=article&utm_source=QQ-qun&utm_content=m_7920)




- **2017**
    1. [+ * Class-Weighted Convolutional Features for Visual Instance Search](https://github.com/imatge-upc/retrieval-2017-cam)
    2. [+ Deep Region Hashing for Efficient Large-scale Instance Search from Images](https://arxiv.org/abs/1701.07901)
    3. [高效图片搜索与识别：WINE-O.AI开发实例](https://github.com/mlgill/scipy_2017_wine-o.ai),
       [车辆精确检索](https://github.com/iamhankai/vehicle-retrieval-kCNNs)
    4. [Compression of Deep Neural Networks for Image Instance Retrieval](https://arxiv.org/abs/1701.04923)
    5. [* 基于图像检索的行人重识别](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247486312&idx=1&sn=41016c176811ec61b5613e252c568680&chksm=96e9d4e8a19e5dfe3d9c25a9a7a05ec1526ac7bf79cf50768631cc405bab1050712705ecca7a#rd)
    6. [Transformation-Invariant Reverse Image Search](https://github.com/pippy360/transformationInvariantImageSearch)
    7. [基于递归注意力模型的卷积神经网络：让精细化物体分类成为现实](https://weibo.com/ttarticle/p/show?id=2309404133830159114110)
    8. [VGG Image Search Engine (VISE)](http://www.robots.ox.ac.uk/~vgg/software/vise/)
    9. [魏秀参：细粒度、无监督图像检索](https://mp.weixin.qq.com/s/XAfwhfQHDIhv1b4FxD_X0g)
    10. [A Revisit on Deep Hashings for Large-scale Content Based Image Retrieval](https://arxiv.org/pdf/1711.06016.pdf)
    11. [+ Face++: 行人重识别度量学习方法](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247486474&idx=1&sn=b8391ab277c543acf183acd6213138f4&chksm=96e9d38aa19e5a9c28d8f22e0246442099f1dd6de3dbf3c0b3dc0d68a78f5cda60281cc06074#rd)
    12. [行人重识别: Unsupervised Adaptation for Deep Stereo](http://www.paperweekly.site/papers/1084)
    13. [+ Unsupervised Triplet Hashing for Fast Image Retrieval](https://arxiv.org/abs/1702.08798)
    14. [SUBIC: A supervised, structured binary code for image search]()
    15. [基于深度学习的端到端相似图像检索 TiefVision - End-to-end deep learning image-similarity search engine](https://github.com/paucarre/tiefvision)



- **监督**
    1. [深度离散哈希算法（监督学习）](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650732724&idx=5&sn=c4761aa946af3f714df57396d8172454&chksm=871b3ccab06cb5dccd43d63a2a1a7693acae62a0f98ab60f229896ed933afe0bbdb3544f8792#rd)
    2. [Neural Codes for Image Retrieval](https://arxiv.org/pdf/1404.1777.pdf)


- **无监督**
    1. [* 卷积去噪自编码器](https://blog.sicara.com/keras-tutorial-content-based-image-retrieval-convolutional-denoising-autoencoder-dc91450cc511)


#### 近似最近邻搜索 Approximate Nearest Neighbor Search(ANN)

**基于树的方法**
- KD 树
空间纬度比较低的时候，KD 树比较高效，当空间纬度较高，可以采用哈希或矢量量化的方法

**基于哈希的方法**
哈希：将连续的实值散列化为 0、1 的离散值
监督、无监督和半监督
评估：可以使用 knn 得到的近邻作为 ground truth，也可以使用样本自身的类别作为 ground truth
[** https://github.com/FALCONN-LIB/FALCONN/wiki/LSH-Primer局部敏感哈希 LSH]()

**基于矢量量化的方法**


**参考资料**
    1. [图像检索：再叙 ANN Search](http://yongyuan.name/blog/ann-search.html?hmsr=toutiao.io&utm_medium=toutiao.io&utm_source=toutiao.io)



#### metric learning


#### watermark


