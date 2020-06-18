# netvlad-in-speech
An end-to-end framework using NetVLAD for language identification and speaker recognition. 

## 运行环境
  ```
  Python 3.6+
  pytorch v1.5.0
  ```
  
## 数据准备与特征提取
可用使用kaldi进行数据准备和特征提取，计算64维的fbank特征，并用```kaldi_ark2npy.py```脚本转为numpy的数据格式，方便数据加载。
以LRE07任务为例，数据文件准备参考目录 ```data/lre07_demo```。已实现多线程流水线式的数据载入模块，见``speech_dataloader.py``。

## 模型框架

![End-to-End framework for LID and SRE](https://github.com/jkchen79/netvlad-in-speech/blob/master/images/end-to-end_framework_for_LID_SRE.png)
（图片摘自第二篇参考论文）

提取fbank声学特征，以Resnet为帧基本的特征提取器，以NetVLAD为编码层，将变长的输入数据整合为语句级别的固定维度的embedding表示。输出层根据具体的语种识别或说话人识别任务作相应调整。

## NetVLAD编码层
NetVLAD可以高效地从大小变化的二维输入数据中提取多个维度的信息，并整合输出维度固定的向量表示，可在语种识别和声纹识别中应用（参考论文1和3）。
  - LRE07语种识别结果（摘自参考论文1）
  
|ID| Model | 3s Cavg (%) / EER(%) | 3s Cavg (%) / EER(%) |3s Cavg (%) / EER(%)   
|---|--------- | --- | --- | ------    
|1| Resnet34 + AvgPool | 9.24 / 10.91 | 3.39 / 5.58 | 1.83 / 3.64 
|2| Resnet34 + NetVLAD 64 |**8.59 / 8.08** | **2.80 / 2.50** | **1.32 / 1.02** 

  - voxceleb说话人识别结果
    - 待更新，线下验证中


## 参考论文
  - [**Jinkun Chen**, Weicheng Cai et al. "End-to-end Language Identification using NetFV and NetVLAD." 2018 11th International Symposium on Chinese Spoken Language Processing (ISCSLP). IEEE, 2018.](https://arxiv.org/abs/1809.02906)
  - [Weicheng Cai, **Jinkun Chen**, and Ming Li. "Exploring the encoding layer and loss function in end-to-end speaker and language recognition system." arXiv preprint arXiv:1804.05160 (2018).](https://arxiv.org/abs/1804.05160)
  - [Xie, Weidi, et al. "Utterance-level aggregation for speaker recognition in the wild." ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019.](https://arxiv.org/abs/1902.10107)
