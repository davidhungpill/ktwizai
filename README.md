# ktwizai
kt wiz ai 이벤트 프로젝트 AI 

local 및 docker 미사용
1. pip install -r requirements.txt
-> 필요 라이브러리 설치
2. python manage.py migrate
-> DB설정
3. pthon manage.py runserver
-> 서버 실행 (직접 AI서버에 연동 없을 예정으로, WSGI 미사용)

docker 사용 시
1. docker image build -t ktwizdjango:0.1 .
-> docker image 생성 (base python3)
2. docker run -p 8000:8000 ktwizdjango:0.1
-> 서버 실행
