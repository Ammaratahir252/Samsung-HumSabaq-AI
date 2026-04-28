import os
import io
import json
import base64
import fitz

from datetime import datetime, timedelta

from django.utils import timezone
from django.conf import settings
from django.core.mail import send_mail
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from groq import Groq

from .models import Student, Timetable, TimetableEvent, Reminder
from . import scheduler as app_scheduler


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing")
    return Groq(api_key=api_key)


# ── Send reminder email ────────────────────────────────────────────────────
def send_class_reminder(email, subject, day, time, tips):
    tips_text = "\n  ▸ ".join(tips)
    message = f"""
Hi! This is HumSabaq AI 📚

━━━━━━━━━━━━━━━━━━━━━━━━━━━
📌 CLASS REMINDER — 30 minutes to go!
━━━━━━━━━━━━━━━━━━━━━━━━━━━

Subject : {subject}
Day     : {day}
Time    : {time}

📖 Study Tips:
  ▸ {tips_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━
Good luck! — HumSabaq AI 🎓
COMSATS University Islamabad
"""
    send_mail(
        subject=f"📚 HumSabaq AI — {subject} in 30 minutes!",
        message=message,
        from_email=settings.EMAIL_HOST_USER,
        recipient_list=[email],
        fail_silently=False,
    )
    print(f"📧 Email sent: {subject} to {email}")


# ── Send scheduled reminder (called by scheduler) ─────────────────────────
def send_scheduled_reminder_email(reminder_id):
    try:
        reminder = Reminder.objects.select_related("student", "timetable_event").get(id=reminder_id)

        if reminder.email_sent:
            return

        event = reminder.timetable_event
        tips = ["Revise your notes", "Practice key questions", "Stay focused"]

        try:
            client = get_groq_client()
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a study assistant. Return ONLY valid JSON, no extra text."
                    },
                    {
                        "role": "user",
                        "content": f'Give 3 short specific study tips for {event.subject}. Return ONLY JSON: {{"tips":["tip1","tip2","tip3"]}}'
                    }
                ],
                max_tokens=200
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1:
                data = json.loads(raw[start:end])
                tips = data.get("tips", tips)
        except Exception as e:
            print(f"AI tips fallback: {e}")

        send_class_reminder(
            reminder.student.email,
            event.subject,
            event.day,
            event.time,
            tips
        )

        reminder.email_sent = True
        reminder.save()

    except Exception as e:
        print(f"❌ Reminder send error: {e}")


# ── Create reminder records in database ───────────────────────────────────
def create_reminders_for_timetable(timetable):
    Reminder.objects.filter(timetable_event__timetable=timetable).delete()

    weekdays = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    now = timezone.localtime()

    for event in timetable.events.all():
        try:
            class_time = None
            for fmt in ['%I:%M %p', '%I %p', '%H:%M', '%H:%M:%S', '%I:%M%p', '%I%p']:
                try:
                    class_time = datetime.strptime(event.time.strip(), fmt)
                    break
                except:
                    continue

            if class_time is None:
                print(f"⚠️ Could not parse time: {event.time}")
                continue

            target_day = weekdays.get(event.day)
            if target_day is None:
                print(f"⚠️ Unknown day: {event.day}")
                continue

            days_ahead = target_day - now.weekday()
            if days_ahead < 0:
                days_ahead += 7

            class_date = now.date() + timedelta(days=days_ahead)
            class_dt = datetime.combine(class_date, class_time.time())
            class_dt = timezone.make_aware(class_dt)

            remind_at = class_dt - timedelta(minutes=30)
            if remind_at <= now:
                remind_at += timedelta(days=7)

            Reminder.objects.create(
                timetable_event=event,
                student=timetable.student,
                remind_at=remind_at
            )
            print(f"✅ Reminder created: {event.subject} → {remind_at}")

        except Exception as e:
            print(f"❌ Reminder create error for {event.subject}: {e}")


# ── Schedule all reminders using APScheduler ──────────────────────────────
def schedule_db_reminders_for_timetable(timetable):
    sched = app_scheduler.get_scheduler()
    if not sched.running:
        sched.start()

    reminders = Reminder.objects.filter(
        timetable_event__timetable=timetable,
        email_sent=False
    )

    scheduled = 0
    for r in reminders:
        sched.add_job(
            send_scheduled_reminder_email,
            trigger="date",
            run_date=r.remind_at,
            args=[r.id],
            id=f"reminder_{r.id}",
            replace_existing=True
        )
        scheduled += 1

    print(f"⏰ {scheduled} reminders scheduled in APScheduler!")


# ── Main API: parse timetable ─────────────────────────────────────────────
@csrf_exempt
def parse_schedule(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    email = request.POST.get("email", "").strip()
    title = request.POST.get("title", "").strip()
    file = request.FILES.get("file")

    if not email:
        return JsonResponse({"error": "Email required"}, status=400)
    if not file:
        return JsonResponse({"error": "File required"}, status=400)

    student, _ = Student.objects.get_or_create(email=email)

    file_name = file.name.lower()
    file_data = file.read()

    # ── Convert file to base64 image for vision AI ────────────────────
    try:
        if file_name.endswith(".pdf"):
            pdf = fitz.open(stream=file_data, filetype="pdf")
            # ✅ Render at high DPI to capture all text clearly
            pix = pdf[0].get_pixmap(dpi=250)
            image_data = pix.tobytes("jpeg")
            mime_type = "image/jpeg"
        elif file_name.endswith(".png"):
            image_data = file_data
            mime_type = "image/png"
        else:
            image_data = file_data
            mime_type = "image/jpeg"

        base64_image = base64.b64encode(image_data).decode("utf-8")

    except Exception as e:
        return JsonResponse({"error": f"File reading failed: {str(e)}"}, status=400)

    # ── Vision AI: read image directly — no OCR needed ────────────────
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                    },
                    {
                        "type": "text",
                        "text": """You are a timetable reader. Look at this timetable image very carefully.
Your job is to extract EVERY SINGLE class without missing any.
Go through every row and column systematically.

Return ONLY a valid JSON array — no explanation, no markdown, no extra text.

Example format:
[
  {"subject": "Math", "day": "Monday", "time": "9:00 AM", "room": "101", "teacher": "Dr. Smith"},
  {"subject": "Physics", "day": "Tuesday", "time": "11:00 AM", "room": "202", "teacher": "Dr. Ali"}
]

If room or teacher is not visible, use empty string "".
Do NOT skip any class. Extract all of them."""
                    }
                ]
            }],
            max_tokens=4000
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        start = raw.find("[")
        end = raw.rfind("]") + 1

        if start == -1:
            return JsonResponse({"error": "AI could not read timetable"}, status=500)

        events = json.loads(raw[start:end])
        print(f"✅ Vision AI extracted {len(events)} classes")

    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}\nRAW: {raw}")
        return JsonResponse({"error": "AI returned invalid format"}, status=500)
    except Exception as e:
        print(f"❌ AI error: {e}")
        return JsonResponse({"error": str(e)}, status=500)

    # ── Save to database ───────────────────────────────────────────────
    Timetable.objects.filter(student=student, is_active=True).update(is_active=False)

    tt = Timetable.objects.create(
        student=student,
        title=title or "My Timetable",
        uploaded_file_name=file.name,
        is_active=True
    )

    for e in events:
        TimetableEvent.objects.create(
            timetable=tt,
            subject=e.get("subject", ""),
            day=e.get("day", ""),
            time=e.get("time", ""),
            room=e.get("room", ""),
            teacher=e.get("teacher", "")
        )

    print(f"✅ Saved {len(events)} events for {email}")

    create_reminders_for_timetable(tt)
    schedule_db_reminders_for_timetable(tt)

    return JsonResponse({"events": events, "count": len(events)})


# ── List all timetables for a student ─────────────────────────────────────
@csrf_exempt
def list_timetables(request):
    email = request.GET.get("email", "").strip()
    if not email:
        return JsonResponse({"error": "Email required"}, status=400)

    try:
        student = Student.objects.get(email=email)
        timetables = Timetable.objects.filter(student=student).order_by("-created_at")

        result = []
        for tt in timetables:
            events = list(tt.events.values("subject", "day", "time", "room", "teacher"))
            result.append({
                "id": tt.id,
                "title": tt.title,
                "is_active": tt.is_active,
                "created_at": tt.created_at.strftime("%Y-%m-%d %H:%M"),
                "events": events,
                "event_count": len(events)
            })

        return JsonResponse({"timetables": result, "count": len(result)})

    except Student.DoesNotExist:
        return JsonResponse({"timetables": [], "count": 0})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ── Set a timetable as active ──────────────────────────────────────────────
@csrf_exempt
def set_active_timetable(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        data = json.loads(request.body)
        timetable_id = data.get("timetable_id")
        email = data.get("email", "").strip()

        student = Student.objects.get(email=email)
        Timetable.objects.filter(student=student).update(is_active=False)

        tt = Timetable.objects.get(id=timetable_id, student=student)
        tt.is_active = True
        tt.save()

        create_reminders_for_timetable(tt)
        schedule_db_reminders_for_timetable(tt)

        return JsonResponse({"success": True, "active_timetable": tt.title})

    except Student.DoesNotExist:
        return JsonResponse({"error": "Student not found"}, status=404)
    except Timetable.DoesNotExist:
        return JsonResponse({"error": "Timetable not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ── Delete a timetable ────────────────────────────────────────────────────
@csrf_exempt
def delete_timetable(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        data = json.loads(request.body)
        timetable_id = data.get("timetable_id")
        email = data.get("email", "").strip()

        student = Student.objects.get(email=email)
        tt = Timetable.objects.get(id=timetable_id, student=student)
        tt.delete()

        return JsonResponse({"success": True, "message": "Timetable deleted"})

    except Student.DoesNotExist:
        return JsonResponse({"error": "Student not found"}, status=404)
    except Timetable.DoesNotExist:
        return JsonResponse({"error": "Timetable not found"}, status=404)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ── Popup reminders endpoint ───────────────────────────────────────────────
@csrf_exempt
def due_reminders(request):
    email = request.GET.get("email", "").strip()
    if not email:
        return JsonResponse({"reminders": []})

    reminders = Reminder.objects.filter(
        student__email=email,
        popup_sent=False,
        remind_at__lte=timezone.localtime()
    ).select_related("timetable_event")

    result = []
    for r in reminders:
        e = r.timetable_event
        result.append({
            "subject": e.subject,
            "day": e.day,
            "time": e.time,
            "message": f"{e.subject} class starts in 30 minutes. Get ready!"
        })
        r.popup_sent = True
        r.save()

    return JsonResponse({"reminders": result})


# ── AI study tips endpoint ─────────────────────────────────────────────────
@csrf_exempt
def get_reminders(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        data = json.loads(request.body)
        subject = data.get("subject", "")
        day = data.get("day", "")
        time = data.get("time", "")

        # ✅ FIX: actual AI call instead of hardcoded response
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful study assistant. Always respond with valid JSON only. No extra text."
                },
                {
                    "role": "user",
                    "content": f"""Student has {subject} class on {day} at {time}.
Give a motivating reminder and 3 specific study tips for this subject.
Return ONLY this JSON:
{{
  "reminder": "one motivating sentence about attending {subject}",
  "study_tips": [
    "specific tip 1 for {subject}",
    "specific tip 2 for {subject}",
    "specific tip 3 for {subject}"
  ]
}}"""
                }
            ],
            max_tokens=400
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
        result = json.loads(raw)
        return JsonResponse(result)

    except Exception as e:
        print(f"AI reminders fallback: {e}")
        return JsonResponse({
            "reminder": f"Don't forget your {data.get('subject','')} class on {data.get('day','')} at {data.get('time','')}. Be prepared!",
            "study_tips": [
                f"Review your {data.get('subject','')} notes and key concepts",
                "Complete any pending assignments before class",
                "Prepare questions to ask your teacher"
            ]
        })
@csrf_exempt
def test_email(request):
    try:
        send_mail(
            "Test Email",
            "Your email setup is working.",
            settings.EMAIL_HOST_USER,
            ["bashair.nasir24@gmail.com"],
            fail_silently=False,
        )
        return JsonResponse({"message": "sent"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# ── Health check ───────────────────────────────────────────────────────────
def health(request):
    return JsonResponse({"status": "ok", "module": "Schedule Assistant"})