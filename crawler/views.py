from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings

# Create your views here.
import datetime
import re
from pathlib import Path
import random
import itertools

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels import robust
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.arima_model import ARIMA
import statsmodels.tsa.arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima

from scipy import stats
from scipy.stats import trim_mean

import json

from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import requests
from urllib import parse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from sqlalchemy import create_engine
from time import sleep

import platform

SEASON_GAME_NUMBER = 144
team_initial = {'KT':'KT','키움':'WO','LG':'LG','롯데':'LT','SSG':'SK','두산':'OB','삼성':'SS', 'KIA':'HT', '한화' :'HH', 'NC':'NC'}

now_date_str = datetime.datetime.now().strftime('%Y%m%d')
now_date_datetime = datetime.datetime.now().strftime('%Y-%m-%d')

host = settings.DATABASES['default']['HOST']
user = settings.DATABASES['default']['USER']
password = settings.DATABASES['default']['PASSWORD']
database_name = settings.DATABASES['default']['NAME']
database_url = f'postgresql://{user}:{password}@{host}:5432/{database_name}'

if('Windows' in platform.system()):
    driver_path = './static/chromedriver.exe'
else:
    driver_path = './static/chromedriver'


def transform_schedule(schedule_df):
    transformed_df = schedule_df
    number_regex = re.compile('[\d]+') # number_regex = re.compile('[0-9]+')
    word_regex = re.compile('[가-힣A-Z]+')
    team_regex = re.compile('(한화|KIA|LG|롯데|삼성|KT|NC|두산|SSG|키움)')
    # month day
    month_day = number_regex.findall(transformed_df.date)
    month = month_day[0]
    day = month_day[1]
    
    # time
    time = schedule_df.time.split(':')
    if(len(time) > 1):
        hour = time[0]
        minute = time[1]
    else:
        hour = '00'
        minute = '00'
    
    date = datetime.datetime(2021, int(month), int(day), int(hour), int(minute))
    transformed_df['game_date'] = date
    # game
    game_result = schedule_df.result
    teams = team_regex.findall(game_result)
    score = number_regex.findall(game_result)
    if('없습니다' in game_result):
        result = 'NOGAME'
    elif(('취소' in game_result) | (len(score) == 0)):
        away_team = teams[0]
        home_team = teams[1]
        transformed_df['home_team'] = home_team
        transformed_df['away_team'] = away_team
    else:
        away_team = teams[0]
        home_team = teams[1]
        away_team_score = score[0]
        home_team_score = score[1]
        transformed_df['home_team'] = home_team
        transformed_df['away_team'] = away_team
        transformed_df['home_team_score'] = home_team_score
        transformed_df['away_team_score'] = away_team_score

    return transformed_df

def crawling_naver_kboSchedule(target_year = 2021):
    today_datetime = datetime.datetime.now().strftime('%Y%m%d')
    game_schedule_2021 = None
    for i in range(4, 13):
        url = f'https://sports.news.naver.com/kbaseball/schedule/index?date={today_datetime}&month=0{i}&year={target_year}&teamCode='
        schedule = pd.read_html(url)
        for j in range(1, len(schedule) - 1):
            if(game_schedule_2021 is None):
                game_schedule_2021 = schedule[j]
            else:
                game_schedule_2021 = pd.concat([game_schedule_2021, schedule[j]], axis=0)

    game_schedule_2021.columns = schedule[0].columns
    game_schedule_2021.drop_duplicates(inplace=True)
    game_schedule_2021 = game_schedule_2021.reindex(columns=['날짜','시간','구장','경기','중계/기록']).reset_index(drop=True).rename(columns={'날짜':'date','시간':'time','구장':'stadium','경기':'result','중계/기록':'note'})
    game_schedule_2021.reset_index(drop=True, inplace=True)
    return game_schedule_2021


def preprocessing_schedule(transformed_df):
    word_regex = re.compile('[가-힣A-Z]+')
    transformed_df =  transformed_df.reindex(columns=['game_date','away_team','away_team_score','home_team','home_team_score','stadium', 'note'])
    transformed_df['away_team_score'] = transformed_df['away_team_score'].astype('float')
    transformed_df['home_team_score'] = transformed_df['home_team_score'].astype('float')
    transformed_df['result'] = transformed_df[~transformed_df.isnull()].apply(lambda x: -1 if x.away_team_score > x.home_team_score else 1
                                                                     if x.home_team_score > x.away_team_score else 0 ,axis = 1)
    transformed_df['stadium'] = transformed_df[~transformed_df.stadium.isnull()].stadium.map(lambda x: word_regex.findall(x)[0])
    return transformed_df

def get_schedule_pre():
    engine = create_engine(database_url, echo=False)
    schedule_pre_df = pd.read_sql_table('ai_pre_kbo_schedule', engine)
    return schedule_pre_df

def get_remain_game_number(team = 'KT'):
    schedule_pre_df = get_schedule_pre()

    remains_game_number = len(schedule_pre_df.loc[((schedule_pre_df.home_team == team)  | (schedule_pre_df.away_team == team)) & (schedule_pre_df.game_date > now_date_datetime)])
    return remains_game_number

def get_played_game_number(team = 'KT'):
    schedule_pre_df = get_schedule_pre()

    played_game_number_away_kt = len(schedule_pre_df.loc[(schedule_pre_df.away_team == team) & (~schedule_pre_df.away_team_score.isnull())])
    played_game_number_home_kt = len(schedule_pre_df.loc[(schedule_pre_df.home_team == team) & (~schedule_pre_df.home_team_score.isnull())])
    played_game_number_kt = played_game_number_away_kt + played_game_number_home_kt
    
    return played_game_number_kt

def get_remain_game_df(team='KT'):
    schedule_pre_df = get_schedule_pre()
    remain_games = schedule_pre_df.loc[schedule_pre_df.game_date > now_date_datetime]
    season_away_remain = remain_games.loc[(remain_games.away_team == team), ['game_date','home_team']]
    season_away_remain['homegame'] = 0
    season_away_remain.rename(columns={'home_team' : 'op_team'}, inplace=True)
    season_home_remain = remain_games.loc[(remain_games.home_team == team), ['game_date','away_team']]
    season_home_remain['homegame'] = 1
    season_home_remain.rename(columns={'away_team' : 'op_team'}, inplace=True)

    season_remain = pd.concat([season_away_remain, season_home_remain])
    return season_remain

def get_season_history_df(team='KT'):
    schedule_pre_df = get_schedule_pre()

    away_df = schedule_pre_df.loc[(schedule_pre_df.away_team == team) & (~schedule_pre_df.away_team_score.isnull())]
    season_away_history = away_df.drop(columns='away_team').rename(columns={'away_team_score' : 'score', 'home_team' : 'op_team', 'home_team_score' : 'op_score'})
    season_away_history['result'] = season_away_history.result.map(lambda x: -(x))
    season_away_history['homegame'] = 0

    home_df = schedule_pre_df.loc[(schedule_pre_df.home_team == team) & (~schedule_pre_df.home_team_score.isnull())]
    season_home_history = home_df.drop(columns='home_team').rename(columns={'home_team_score' : 'score', 'away_team' : 'op_team', 'away_team_score' : 'op_score'})
    season_home_history['homegame'] = 1

    season_history = pd.concat([season_away_history, season_home_history]).sort_index()
    season_history = season_history.loc[season_history.game_date < datetime.datetime.now().strftime('%Y%m%d 00:00:00')]
    season_history.drop(columns= 'note', inplace=True)
    
    return season_history

def get_vs_statics_df(team='KT'):
    season_history = get_season_history_df(team)
    total_games = season_history.pivot_table(index=['result'], columns=['op_team'],values='homegame' ,aggfunc='count').fillna(0).T
    total_games.columns = ['패','무','승']
    total_games = total_games.reindex(columns = ['승','무','패'])
    total_games['승률'] = total_games.apply(lambda row: row.승 / (sum(row) - row.무), axis=1)
    return total_games

def get_season_win_number(team='KT'):
    remain_game_df = get_remain_game_df(team)
    history_game_df = get_season_history_df(team)
    vs_statics_df =  get_vs_statics_df(team)
    remain_game_table = remain_game_df.pivot_table(columns = 'op_team', values='game_date', aggfunc='count').T
    remain_game_table.columns = ['잔여경기']
    win_predict_table = vs_statics_df.join(remain_game_table).fillna(0)
    win_predict_table['추가예상승'] = np.round(win_predict_table['잔여경기'] * win_predict_table['승률'])
    return win_predict_table.sum().승 + win_predict_table.sum().추가예상승

def get_season_win_number_result(team = 'KT'):
    remain_game_df = get_remain_game_df(team)
    history_game_df = get_season_history_df(team)
    vs_statics_df =  get_vs_statics_df(team)
    win_number = int(get_season_win_number(team))
    remain_games_number = np.round(len(remain_game_df))
    result_str = f'현재 {team}의 2021년 정규시즌 잔여 경기는 {remain_games_number}경기이며, 최종 승리 예상 숫자는 {win_number-1} ~ {win_number+1} 승 입니다'
    
    return result_str

def get_win_static_table(target_team = 'KT'):
    remain_game_df = get_remain_game_df(target_team)
    remain_game_table = remain_game_df.pivot_table(columns='op_team',values='game_date', aggfunc='count').T
    remain_game_table.columns = ['잔여경기']
    win_result_support_tb1 =  get_vs_statics_df(target_team).join(remain_game_table).fillna(0)
    win_result_support_tb1 = win_result_support_tb1.iloc[:,[0,1,2,4,3]]
    
    return win_result_support_tb1

def get_schedule(request):
    # schedule crawling
    schedule_df = crawling_naver_kboSchedule()
    # schedule_df.to_csv(f'./data/ktwiz_kboSchedule_crawlingNaver_v{now_date_str}.csv', encoding='utf-8')
    schedule_df.head()

    schedule_pre_df = schedule_df.apply(transform_schedule, axis=1)
    schedule_pre_df = preprocessing_schedule(schedule_pre_df)

    # schedule_pre_df.to_csv(f'./data/ktwiz_schedule_pre_v{now_date_str}.csv', encoding='utf-8')
    schedule_pre_df.head()

    html = schedule_pre_df.to_html()
    return HttpResponse(html)

def save_df_into_table(name, df):

    engine = create_engine(database_url, echo=False)
    
    try:
        engine.execute(f'drop table {name}')
    except:
        print(f'{name} table is not exist')
    
    df.to_sql(name, con=engine)


def save_crawling_schedule():
    # schedule crawling
    schedule_df = crawling_naver_kboSchedule()
    save_df_into_table('ai_kbo_schedule', schedule_df)

def save_preprocessing_schedule():
    # schedule crawling
    engine = create_engine(database_url, echo=False)
    schedule_df = pd.read_sql_table('ai_kbo_schedule', engine)
    schedule_pre_df = schedule_df.apply(transform_schedule, axis=1)
    schedule_pre_df = preprocessing_schedule(schedule_pre_df)
    save_df_into_table('ai_pre_kbo_schedule', schedule_pre_df)

def predict_season_win_number(target_team = 'KT'): 
    win_result = get_season_win_number(target_team)
    return win_result

def get_choices_season_win_number():
    examples = np.arange(1,5) * sum(get_win_static_table().잔여경기) / 4
    examples = np.append(examples, 0)
    examples.sort()
    examples = [round(i) for i in examples]
    examples = np.array(examples) + sum(get_win_static_table().승)
    q1 = np.percentile(examples, 0)
    q2 = np.percentile(examples, 25)
    q3 = np.percentile(examples, 50)
    q4 = np.percentile(examples, 75)
    q5 = np.percentile(examples, 100)

    op1 = f'{q1}~{q2 -1}'
    op2 = f'{q2}~{q3 -1}'
    op3 = f'{q3}~{q4 -1}'
    op4 = f'{q4}~{q5}'
    choices = f'{op1}_{op2}_{op3}_{op4}'

    return choices


# avg, era

def get_game_review_url(team, game_summary_row):
    team_initial_name = team_initial[game_summary_row.op_team]
    homegame = game_summary_row.homegame
    game_dt = game_summary_row.game_date
    game_dt_str = str(game_dt).replace('-','')[:8]
    post_fix = game_summary_row.DH
    
    if(homegame == 1):
        game_id = game_dt_str + team_initial_name + team + post_fix
    else:
        game_id = game_dt_str + team + team_initial_name + post_fix

    url = f'https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx?gameDate={game_dt_str}&gameId={game_id}&section=REVIEW'
    return url

def get_game_review(driver, game_summary_row):
    try:
        print(game_summary_row.review_url)
        driver.get(game_summary_row.review_url)
        driver.implicitly_wait(1)
        page = pd.read_html(driver.page_source)

        away_batting_info = page[6].iloc[-1]
        away_batting_info = away_batting_info.add_prefix('batting_')
        away_pitching_info = page[10].iloc[-1][-11:]
        away_pitching_info = away_pitching_info.add_prefix('pitching_')
        away_info =  pd.concat([away_batting_info, away_pitching_info])
        away_info = away_info.add_prefix('away_')

        home_batting_info = page[9].iloc[-1]
        home_batting_info = home_batting_info.add_prefix('batting_')
        home_pitching_info = page[11].iloc[-1][-11:]
        home_pitching_info = home_pitching_info.add_prefix('pitching_')
        home_info =  pd.concat([home_batting_info, home_pitching_info])
        home_info = home_info.add_prefix('home_')

        info = pd.concat([away_info, home_info])
        return info
    except:
        driver = webdriver.Chrome(driver_path)
        print('except selenium', game_summary_row.review_url)
        driver.implicitly_wait(7)
        driver.get(game_summary_row.review_url)
        page = pd.read_html(driver.page_source)

        away_batting_info = page[6].iloc[-1]
        away_batting_info = away_batting_info.add_prefix('batting_')
        away_pitching_info = page[10].iloc[-1][-11:]
        away_pitching_info = away_pitching_info.add_prefix('pitching_')
        away_info =  pd.concat([away_batting_info, away_pitching_info])
        away_info = away_info.add_prefix('away_')

        home_batting_info = page[9].iloc[-1]
        home_batting_info = home_batting_info.add_prefix('batting_')
        home_pitching_info = page[11].iloc[-1][-11:]
        home_pitching_info = home_pitching_info.add_prefix('pitching_')
        home_info =  pd.concat([home_batting_info, home_pitching_info])
        home_info = home_info.add_prefix('home_')

        info = pd.concat([away_info, home_info])
        return info


def enrich_doubleHeader_info(scheduled_df):
    scheduled_df['game_date_post'] = scheduled_df.game_date.shift(-1)
    scheduled_df['game_date_prev'] = scheduled_df.game_date.shift(1)
    scheduled_df['DH'] = scheduled_df.apply(lambda x: '1' if(x.game_date.day == x.game_date_post.day) else '2' if(x.game_date.day == x.game_date_prev.day) else '0', axis=1)
    scheduled_df = scheduled_df[['game_date','DH','score','op_team','op_score','stadium','result','homegame']]
    return scheduled_df

def rename_detail_summary_columns(row):
    if(row.homegame == 1):
        row = row.rename(lambda x: x.replace('away_','op_'))
        row = row.rename(lambda x: x.replace('home_',''))
    elif(row.homegame ==0):
        row = row.rename(lambda x: x.replace('away_',''))
        row = row.rename(lambda x: x.replace('home_','op_'))       
    else:
        print('no change')
    return row

def reorg_detail_summary_dataframe(df):
    cols = df.columns
    season_summary_detail_cols = ['game_date', 'DH','stadium','homegame','score','op_team','op_score','result']
    op_cols = [i for i in cols  if (('op_batting' in i) | ('op_pitching' in i))]
    target_team_cols = [i for i in cols  if((('batting' in i) | ('pitching' in i)) & ('op_' not in i))]
    target_columns = season_summary_detail_cols + target_team_cols + op_cols
    df = df[target_columns]
    return df

def preprocessing_inning(inning):
    inning_str = str(inning)
    if('/' in inning_str):
        first_num = inning_str.split()[0]
        second_num, third_num = inning_str.split()[1].split('/')
        return float(first_num) + float(second_num) / float(third_num)
    else:
        return float(inning)

def crawling_game_detail(team = 'KT'):
    driver = webdriver.Chrome(driver_path)
    season_summary_df = enrich_doubleHeader_info(get_season_history_df(team).copy())
    season_summary_df['review_url'] = season_summary_df.apply(lambda x: get_game_review_url(team, x), axis=1)

    enriched_season_summary_df = season_summary_df.apply(lambda x: pd.concat([x, get_game_review(driver, x)]), axis=1)

    enriched_season_summary = enriched_season_summary_df.apply(rename_detail_summary_columns, axis=1)
    enriched_season_summary = reorg_detail_summary_dataframe(enriched_season_summary)

    enriched_season_summary['pitching_이닝'] = enriched_season_summary.pitching_이닝.map(preprocessing_inning)
    enriched_season_summary['op_pitching_이닝'] = enriched_season_summary.op_pitching_이닝.map(preprocessing_inning)
    enriched_season_summary['season_era'] = round((enriched_season_summary.pitching_자책.cumsum() * 9 / enriched_season_summary.pitching_이닝.cumsum()), 2)

    return enriched_season_summary

def crawling_game_detail_update(target_date, team = 'KT'):
    driver = webdriver.Chrome('./static/chromedriver.exe')
    season_summary_df = enrich_doubleHeader_info(get_season_history_df(team))
    season_summary_df['review_url'] = season_summary_df.apply(lambda x: get_game_review_url(team, x), axis=1)

    if(len(season_summary_df.loc[season_summary_df.game_date > target_date]) > 0):
        season_summary_df = season_summary_df.loc[season_summary_df.game_date > target_date]

        print(driver.get_network_conditions)
        enriched_season_summary_df = season_summary_df.apply(lambda x: pd.concat([x, get_game_review(driver, x)]), axis=1)

        enriched_season_summary = enriched_season_summary_df.apply(rename_detail_summary_columns, axis=1)
        enriched_season_summary = reorg_detail_summary_dataframe(enriched_season_summary)

        enriched_season_summary['pitching_이닝'] = enriched_season_summary.pitching_이닝.map(preprocessing_inning)
        enriched_season_summary['op_pitching_이닝'] = enriched_season_summary.op_pitching_이닝.map(preprocessing_inning)
        enriched_season_summary['season_era'] = round((enriched_season_summary.pitching_자책.cumsum() * 9 / enriched_season_summary.pitching_이닝.cumsum()), 2)
        
        return enriched_season_summary
    else:
        pass
    
    try:
        driver.close()
    except:
        driver.kill()

def save_season_history_detail():
    # schedule crawling
    engine = create_engine(database_url, echo=False)
    enriched_season_summary = crawling_game_detail()

    try:
        engine.execute(f'drop table ai_kbo_season_history')
    except:
        print('ai_kbo_season_history table is not exist')

    save_df_into_table('ai_kbo_season_history', enriched_season_summary)


def save_season_history_detail_op1():

    engine = create_engine(database_url, echo=False)
    season_history_df = pd.read_sql_table('ai_kbo_season_history', engine)
   
    if(len(season_history_df)>0):
        last_update_date = max(season_history_df.game_date.values)
        updated_enriched_season_summary = crawling_game_detail_update(last_update_date, 'KT')
        
        if(updated_enriched_season_summary is None):
            print(updated_enriched_season_summary, 'none', last_update_date)
        else:
            updated_enriched_season_summary.to_sql('ai_kbo_season_history', engine, if_exists='append', index=True) 
    else:
        save_season_history_detail()

    # try:
    #     engine.execute(f'drop table {name}')
    # except:
    #     print(f'{name} table is not exist')
    
    # df.to_sql(name, con=engine)

def predict_season_era():
   
    engine = create_engine(database_url, echo=False)
    season_history_df = pd.read_sql_table('ai_kbo_season_history', engine)
   
    season_pitching_era = season_history_df[['game_date', 'season_era']].set_index('game_date')
    remain_game_number = get_remain_game_number()

    model = SARIMAX(season_pitching_era, order=(1,1,2))
    model_fit = model.fit(trend='nc')
    fc = model_fit.forecast(remain_game_number, alpha=0.05)
    predicted_value = fc[-1:].values[0]

    # fc_series = pd.Series(fc, index=get_remain_game_df().game_date.values)
    return np.round(predicted_value, 2)

def predict_season_avg():
       
    engine = create_engine(database_url, echo=False)
    season_history_df = pd.read_sql_table('ai_kbo_season_history', engine)
   
    season_batting_avg = season_history_df[['game_date', 'batting_타율']].set_index('game_date')
    remain_game_number = get_remain_game_number()

    model = SARIMAX(season_batting_avg, order=(1,1,2))
    model_fit = model.fit(trend='nc')
    fc = model_fit.forecast(remain_game_number, alpha=0.05)
    predicted_value = fc[-1:].values[0]

    # fc_series = pd.Series(fc, index=get_remain_game_df().game_date.values)
    return np.round(predicted_value, 3)