FROM python:3

USER root

ENV TZ Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone 

RUN apt-get update -y -qq
RUN apt-get upgrade -y -qq
RUN apt-get install -y git 

RUN mkdir -p /home/workspace/ai_backend
WORKDIR /home/workspace/admin_backend
RUN	git clone https://github.com/davidhungpill/ktwizai.git ktwizai
WORKDIR /home/workspace/admin_backend/ktwizai
RUN	pip3 install -r requirements.txt
RUN python manage.py migrate

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
EXPOSE 8000