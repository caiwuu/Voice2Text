# Voice2Text
### Voice2Text是一款基于OpenAI whisper的语音转文字应用，支持音频、视频、实时语音转文字

### 使用方法

1.拉取代码

```
git clone https://github.com/caiwuu/Voice2Text
cd ./Voice2Text
```

2.创建python虚拟环境

```
conda create  -p ./env  python==3.11.9
conda activate ./env
```

3.安装依赖

```
pip install -r requirements.txt
```

4.下载模型到models文件中，模型仓库：[https://huggingface.co/Systran](https://huggingface.co/Systran)，默认使用 faster-whisper-large-v2，其他模型自己代码里改下模型名称

5.启动

```
python webUI.py
```

![image-20240902014651428](https://cdn.jsdelivr.net/gh/caiwuu/image/202409020146577.png)
