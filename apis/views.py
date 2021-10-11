from django.contrib.auth.models import User, Group
from django.http.response import HttpResponse
from rest_framework import viewsets
from rest_framework import permissions
from rest_framework.views import APIView

from .serializers import UserSerializer, GroupSerializer
from django.conf import settings

from crawler import views as cviews

import pandas as pd
from sqlalchemy import create_engine

SEASON_GAME_NUMBER = 144

host = settings.DATABASES['default']['HOST']
user = settings.DATABASES['default']['USER']
password = settings.DATABASES['default']['PASSWORD']
database_name = settings.DATABASES['default']['NAME']
database_url = f'postgresql://{user}:{password}@{host}:5432/{database_name}'


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]

class PredictSeasonResult(APIView):
    def post(self, request, format=None):
        update_all_predict_value()
        return HttpResponse('predicted and updated into DB')

    def get(self, request, format=None):
        update_all_predict_value()
        return HttpResponse('predict and updated into DB')

def update_choice_value(type):

    engine = create_engine(database_url, echo=False)
    question_df = pd.read_sql_table('question', engine)
    target_question = question_df.loc[question_df.content.str.contains(type)]
    question_id = target_question.question_id.values[0]
    choice = 'no_choice'

    if(type =='승'):
        choice = cviews.get_choices_season_win_number()
    elif(type =='타율'):
        choice = cviews.get_choices_season_win_number()
    elif(type =='방어율'):
        choice = cviews.get_choices_season_win_number()
    else:
        print('no question') 

    if(choice != 'no_choice'):
        engine.execute(f"update question set choices = '{choice}' where question_id = {question_id}")
    else:
        pass

def update_predict_value(type):

    engine = create_engine(database_url, echo=False)
    question_df = pd.read_sql_table('question', engine)
    target_question = question_df.loc[question_df.content.str.contains(type)]
    question_id = target_question.question_id.values[0]
   
    answer = 'no_answer'
    if(type =='승'):
        answer = cviews.get_season_win_number()
        # answer = f'{answer + -1}~{answer + 1}승' 
    elif(type =='타율'):
        answer = cviews.predict_season_avg()
    elif(type =='방어율'):
        answer = cviews.predict_season_era()
    else:
        print('no question') 

    if(answer != 'no_answer'):
        answer_df = pd.read_sql_table('aianswer', engine)
        answer_id = max(answer_df.aianswer_id.values) + 1
        engine.execute(f"insert into aianswer values({answer_id}, '{answer}', {question_id})")
    else:
        pass

def update_all_predict_value():
    
    predict_list = ['승','방어율','타율']
    for i in predict_list:
        update_predict_value(i)