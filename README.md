# Global_Linear_Model
## 一、目录文件
    ./data/:
        train.conll: 训练集
        dev.conll: 开发集
    ./big_data/:
        train.conll: 训练集
        dev.conll: 开发集
        test.conll: 测试集
    ./:
        global_liner_model.py: 代码(未进行特征抽取优化)
        global_liner_model_v2.py: 代码(特征抽取优化)
    ./README.md: 使用说明

## 二、运行
### 1.运行环境
    python 3
### 2.运行方法
    各个参数
    'train_data_file': 'data/train.conll', #训练集文件,大数据改为'../big_data/train.conll'
    'dev_data_file': 'data/dev.conll',     #开发集文件,大数据改为'../big_data/dev.conll'
    'test_data_file': 'data/dev.conll',    #测试集文件,大数据改为'../big_data/test.conll'
    'iterator': 100,                       # 最大迭代次数
    'stop_iterator': 10                    # 连续多少个迭代没有提升就退出
    'shuffle': True                        # 每次迭代是否打乱数据
    'averaged': False/True                 # 连续多少个迭代没有提升就退出

    
### 3.参考结果
注：</br>
1.迭代次数均从0开始计算。</br>
2.每次迭代时间为train/test/dev的时间总和。</br>

#### (1)小数据测试
```
训练集：data/train.conll
测试集：data/test.conll
开发集：data/dev.conll
```
| partial feature | averaged percetron | 迭代次数  | train 准确率 | test 准确率 | dev 准确率  | 时间/迭代 |
| :-------------: | :----------------: | :------: | :----------: | :--------: | :---------: | :-------: |
|        ×        |         ×          |  16/26   |    100%      |   84.92%   |   86.52%    |   34.1s   |
|        ×        |         √          |  17/27   |    99.75%    |   85.32%   |   86.69%    |   33.1s   |
|        √        |         ×          |  21/31   |    100%      |   85.91%   |   87.32%    |   10.0s   |
|        √        |         √          |  25/35   |    99.97%    |   86.51%   |   87.31%    |   11.6s   |

#### (2)大数据测试
```
训练集：big-data/train.conll
测试集：big-data/test.conll
开发集：big-data/dev.conll
```
| partial feature | averaged percetron | 迭代次数  | train 准确率 | test 准确率 | dev 准确率  | 时间/迭代 |
| :-------------: | :----------------: | :------: | :----------: | :--------: | :---------: | :-------: |
|        ×        |         ×          |  25/36   |    98.95%    |   93.44%   |   93.18%    |   15min   |
|        ×        |         √          |   8/19   |    98.19%    |   94.27%   |   94.10%    |   17min   |
|        √        |         ×          |  30/41   |    99.10%    |   93.66%   |   93.34%    |  3.5min   |
|        √        |         √          |  15/26   |    99.18%    |   94.26%   |   94.12%    |  3.5min   |
