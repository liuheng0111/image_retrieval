#### Pipeline
- `data`: 数据集
    - `train`: 训练数据集，即用于图片搜索的图片库
    - `test`: 测试数据集，即 query 图片。初期，通过数据增强的方法生成测试数据集
    - `models`: 用于存放训练好的模型，包括计算的感知哈希，预训练的神经网络等
    
- `model`: 模型，提取图像特征
    - `perceptral_hash`: 感知哈希算法
    - `neural_network`: 神经网络算法
    - `other_features`: 其他的手工特征
    
- `ann`: Approximate nearest neighbour search 近似最近邻搜索
    - `tree`: tree based model
        - `kd_tree`
        - `bk_tree`
    - `lsh`: lsh based model
    - `vector_quantization`: 矢量量化
        - `product_quantization`
    
- `utils`: 实用工具
    - `configures`: 配置
    - `get_image_path`: 获取图片路径
    - `img_download`: 下载图片
    - `evaluation`: 评价指标
    
