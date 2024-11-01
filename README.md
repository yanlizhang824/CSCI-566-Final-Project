# CSCI-566-Final-Project

因git push文件大小限制，AudioCLIP-Partial-Training.pt 与 MELD.Raw没有添加进来

获取AudioCLIP-Partial-Training.pt，运行AudioCLIP-master/demo/AudioCLIP.ipynb的第一个代码框：

! wget -P ../assets/ https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/bpe_simple_vocab_16e6.txt.gz
! wget -P ../assets/ https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt

AudioCLIP-Partial-Training.pt文件将被正确存放在AudioCLIP-master/assets中

MELD.Raw从 https://www.kaggle.com/datasets/zaber666/meld-dataset 下载，解压完成后将MELD.Raw移动到data/MELD/MELD_kaggle文件夹下

main_MELD.ipynb的运行环境为main_requirements.txt
data_preprocess.ipynb的运行环境为data_preprocess_requirements.txt
