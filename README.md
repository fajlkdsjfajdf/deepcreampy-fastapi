# deepcreampy-fastapi
这是一个基于deepcream的 web api，可以使用api快速对图片进去去码处理

# 简介
<a target="_blank"  href="https://github.com/cookieY/DeepCreamPy">deepcreampy</a> 是个可以将图片马赛克去除的工具, 专注于你懂得部位的马赛克去除， 可以去除 长条形状的码或者马赛克, 你可以前往deepcreampy了解具体细节。 
但是, deepcreampy 需要将想要修复的部位先手动用绿色或其他纯色涂抹,才能正确的识别并修复。 所以, 本工具结合了 <a href="https://github.com/natethegreate/hent-AI">hent-AI</a> 的自动涂抹马赛克位置功能, 完成后自动交给deepcreampy处理。
本工具使用fastapi 输出接口, 直接使用这些开放接口即可。
