from django.shortcuts import render
from django.http import HttpResponse

import pandas as pd

# Create your views here.
def show_player_statics(request):
    players_df = pd.read_csv('./static/csv/ktwiz_player_info_v202107211.csv', encoding='euc-kr')
    html = players_df.to_html()
    return HttpResponse(html)
