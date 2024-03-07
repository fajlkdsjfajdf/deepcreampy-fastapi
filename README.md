# deepcreampy-fastapi
这是一个基于deepcream的 web api，可以使用api快速对图片进去去码(去除马赛克)处理 

已经集成了deepcream去码的功能, 你不需要额外安装任何其他软件。

# 简介
<a target="_blank"  href="https://github.com/cookieY/DeepCreamPy">deepcreampy</a> 是个可以将二次元图片马赛克去除的工具, 专注于你懂得部位的马赛克去除， 可以去除 长条形状的单色码(一般在漫画中使用)以及通常马赛克(大部分时候在galgame和番剧中使用), 你可以前往deepcreampy了解具体细节。 
但是, deepcreampy 需要将想要修复的部位先手动用绿色或其他纯色涂抹,才能正确的识别并修复。 所以, 本工具结合了 <a href="https://github.com/natethegreate/hent-AI">hent-AI</a> 的自动涂抹马赛克位置功能, 完成后自动交给deepcreampy处理。
本工具使用fastapi 输出接口, 直接使用这些开放接口即可。

!! 注意, 本模型只适合二次元图片

![api 接口](https://raw.githubusercontent.com/fajlkdsjfajdf/deepcreampy-fastapi/main/images/api.png)


# 安装
本工具测试时使用的python 为3.10版本, tenserflow最大支持2.10
下载本代码后, cd 到代码主目录, 使用pip安装依赖

```pip copy
pip install -r requirements.txt
```
国内环境(使用清华源)
```pip copy
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
注意, 由于tenserflow>2.10 的版本在windows 中不再支持GPU加速, 所以requirements.txt 设置了最大版本为2.10, 如果是使用cpu或者再linux上运行,可以放开tenserflow版本限制。

安装完成依赖, 我们需要2个额外的模型来完成安装

模型下载地址

<a href="https://openmodeldb.info/models/4x-Fatal-Pixels">4x-Fatal-Pixels.pth</a>

下载完成后, 将其命名为"4x-Fatal-Pixels.pth" 放入 models/esrgan 文件下

<a href="https://github.com/natethegreate/hent-AI">hent-AI</a>

在# The Model 中找到对应步长的模型下载地址
下载完成后, 将其命名为"weights.h5" 放入 models/mrcnn 文件下

# 模型说明
esrgan模型, 该模型用于在马赛克修复时缩放整张图片, 用以提取马赛克位置的基本形状, 强化deepcreampy 的效果, 你可以在 openmodeldb.info 中下载到 Twittman 训练的对应的esrgan模型
下载完成后, 将其命名为"4x-Fatal-Pixels.pth" 放入 models/esrgan 文件下
<a href="https://openmodeldb.info/models/4x-Fatal-Pixels">4x-Fatal-Pixels.pth</a>

mask-r-cnn模型， Mask R-CNN 是一个实例分割（Instance segmentation）模型, 在这里，我们用其来搜索马赛克区域。
下载完成后, 将其命名为"weights.h5" 放入 models/mrcnn 文件下
你可以在 <a href="https://github.com/natethegreate/hent-AI">hent-AI</a> 的 # The Model 中找到对应步长的模型下载地址。

# 启动
完成安装后, 你可以直接使用 
```python copy
python server.py
```
来启动, 稍等片刻, 启动完成后, 即可在 
```html copy
http://localhost:8001
```
中查看所有接口

# 致谢
<a target="_blank"  href="https://github.com/cookieY/DeepCreamPy">deepcreampy</a>
<a target="_blank"  href="https://github.com/natethegreate/hent-AI">hent-AI</a>
<a target="_blank"  href="https://github.com/nanoskript/deepcreampy-onnx-docker">deepcreampy-onnx-docker</a>

