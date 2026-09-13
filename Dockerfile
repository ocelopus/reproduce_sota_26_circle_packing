FROM python:3.12.13-slim-trixie

# move requirements.txt to /app
COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# docker build -t self_try_evolve:python-3.12.13-slim-trixie .