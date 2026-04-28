from django.urls import path
from . import views

urlpatterns = [
    path('parse-schedule/', views.parse_schedule),
    path('get-reminders/', views.get_reminders),
    path('list-timetables/', views.list_timetables),
    path('set-active-timetable/', views.set_active_timetable),
    path('delete-timetable/', views.delete_timetable),
    path('due-reminders/', views.due_reminders),
    path('test-email/', views.test_email),
    path('health/', views.health),
]