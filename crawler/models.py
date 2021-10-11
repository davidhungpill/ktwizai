from django.db import models

# Create your models here.
class KBOSchedule(models.Model):
    game_date = models.DateTimeField(auto_now_add=False)
    away_team = models.CharField(max_length = 100)
    away_team_score = models.IntegerField(max_length = 10)
    home_team = models.CharField(max_length = 100)
    home_team_score = models.IntegerField(max_length = 10)
    stadium = models.CharField(max_length = 100)
    result = models.CharField(max_length = 2)
    note = models.CharField(max_length = 1000)