from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()

def start():
    if not scheduler.running:
        scheduler.start()

def get_scheduler():
    return scheduler