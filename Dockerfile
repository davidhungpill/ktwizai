FROM ubuntu:18.04

USER root

ENV TZ Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone 

RUN apt-get update -y -qq
RUN apt-get upgrade -y -qq
RUN apt-get install -y vim git curl
RUN apt-get install -y python3-pip python3.8 python3.8-dev
RUN python3.8 -m pip install --upgrade pip setuptools

RUN mkdir -p /home/workspace/ai_backend
WORKDIR /home/workspace/admin_backend
RUN	git clone "https://github.com/davidhungpill/ktwizai.git"
WORKDIR /home/workspace/admin_backend/ktwizai
RUN	pip install -r requirements.txt
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
EXPOSE 8000