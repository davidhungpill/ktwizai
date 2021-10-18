FROM python:3.8.10

ADD  requirements.txt ${HOME}
ADD  devops/jenkins/pip.conf ${HOME}/.pip/
ADD  . ${HOME}

USER root

ENV TZ Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone 

RUN  pip install -r requirements.txt
RUN  chmod -R a+rw ${HOME}

RUN python manage.py migrate

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
EXPOSE 8000
