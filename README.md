# CSCI-566-Final-Project

## 下载文件及数据
因git push文件大小限制，AudioCLIP-Partial-Training.pt 与 MELD.Raw没有添加进来

获取AudioCLIP-Partial-Training.pt，运行AudioCLIP-master/demo/AudioCLIP.ipynb的第一个代码框：

![image](https://github.com/user-attachments/assets/b21d3edf-5d23-45f2-bf1c-9ce2bf8795fa)


AudioCLIP-Partial-Training.pt文件将被正确存放在AudioCLIP-master/assets中

MELD.Raw从 https://www.kaggle.com/datasets/zaber666/meld-dataset 下载，解压完成后将MELD.Raw移动到data/MELD/MELD_kaggle文件夹下

## 环境依赖
data_preprocess.ipynb的运行环境为data_preprocess_requirements.txt

python=3.10.15

main_MELD.ipynb的运行环境为main_requirements.txt

由于版本过低，有些依赖项与高版本python有冲突，所以分三步：

第一步：创建的虚拟环境python版本=3.8.20

第二步：安装requirements.txt依赖

第三步：将python版本降回3.7.12
