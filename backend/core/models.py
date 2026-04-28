from django.db import models


class Student(models.Model):
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.email


class Timetable(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name="timetables")
    title = models.CharField(max_length=200, default="My Timetable")
    uploaded_file_name = models.CharField(max_length=200, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.student.email} — {self.title}"


class TimetableEvent(models.Model):
    timetable = models.ForeignKey(Timetable, on_delete=models.CASCADE, related_name="events")
    subject = models.CharField(max_length=200)
    day = models.CharField(max_length=20)
    time = models.CharField(max_length=20)
    room = models.CharField(max_length=100, blank=True)
    teacher = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return f"{self.subject} — {self.day} at {self.time}"


class Reminder(models.Model):
    timetable_event = models.ForeignKey(TimetableEvent, on_delete=models.CASCADE, related_name="reminders")
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name="reminders")
    remind_at = models.DateTimeField()
    email_sent = models.BooleanField(default=False)
    popup_sent = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Reminder for {self.timetable_event.subject} at {self.remind_at}"