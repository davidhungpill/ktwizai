from django.urls import include, path
from rest_framework import routers
from . import views

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'groups', views.GroupViewSet)
# router.register(r'groups', views.PredictSeasonResult)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),
    path('update_predict/', views.PredictSeasonResult.as_view(), name='predict_update'),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]