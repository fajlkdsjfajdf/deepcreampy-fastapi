# 使用 python3.10 作为基础镜像
FROM python:3.10

# 设置工作目录
WORKDIR /app/deepcreampy

# 将当前目录下的所有文件复制到镜像内的 /app/deepcreampy 目录下
COPY . /app/deepcreampy

# 安装 requirements.txt 中的依赖
RUN pip install -r requirements.txt

# 设置启动命令
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8001", "server:app"]
