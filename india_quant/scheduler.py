"""APScheduler-based scheduler for all pipeline jobs. Timezone: Asia/Kolkata."""
from datetime import datetime

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from flask import Flask, jsonify
from loguru import logger

IST = pytz.timezone("Asia/Kolkata")

_last_run: dict[str, str] = {}


def _track(name: str):
    _last_run[name] = datetime.now(tz=IST).isoformat()


def job_pre_market():
    _track("pre_market")
    from india_quant.data.pipeline import DataPipeline
    DataPipeline.run_pre_market()


def job_intraday():
    _track("intraday")
    from india_quant.data.pipeline import DataPipeline
    DataPipeline.run_intraday()


def job_post_market():
    _track("post_market")
    from india_quant.data.pipeline import DataPipeline
    DataPipeline.run_post_market()


def job_weekly():
    _track("weekly_maintenance")
    from india_quant.data.pipeline import DataPipeline
    DataPipeline.run_weekly_maintenance()


def job_daily_report():
    _track("daily_report")
    from india_quant.reports.daily_report import generate_daily_report
    generate_daily_report()


def create_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(
        jobstores={"default": MemoryJobStore()},
        timezone=IST,
    )

    # Pre-market: 08:00 IST, Mon-Fri
    scheduler.add_job(job_pre_market, "cron", hour=8, minute=0,
                      day_of_week="mon-fri", id="pre_market",
                      misfire_grace_time=300, coalesce=True,
                      max_instances=1)

    # Intraday: every 5 min, Mon-Fri 09:15-15:30
    scheduler.add_job(job_intraday, "cron", minute="*/5",
                      hour="9-15", day_of_week="mon-fri",
                      id="intraday", misfire_grace_time=60,
                      coalesce=True, max_instances=1)

    # Post-market: 16:00 IST, Mon-Fri
    scheduler.add_job(job_post_market, "cron", hour=16, minute=0,
                      day_of_week="mon-fri", id="post_market",
                      misfire_grace_time=600, coalesce=True,
                      max_instances=1)

    # Daily report: 18:00 IST, Mon-Fri
    scheduler.add_job(job_daily_report, "cron", hour=18, minute=0,
                      day_of_week="mon-fri", id="daily_report",
                      misfire_grace_time=600, coalesce=True,
                      max_instances=1)

    # Weekly maintenance: Sunday 22:00 IST
    scheduler.add_job(job_weekly, "cron", hour=22, minute=0,
                      day_of_week="sun", id="weekly_maintenance",
                      misfire_grace_time=900, coalesce=True,
                      max_instances=1)

    return scheduler


def create_health_app(scheduler: BackgroundScheduler) -> Flask:
    app = Flask("india_quant_health")

    @app.route("/health")
    def health():
        jobs = [
            {
                "id": job.id,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            for job in scheduler.get_jobs()
        ]
        return jsonify({
            "status": "ok",
            "scheduler_running": scheduler.running,
            "last_runs": _last_run,
            "jobs": jobs,
        })

    return app


def start_scheduler():
    scheduler = create_scheduler()
    scheduler.start()

    # Run missed pre-market job if today's data is not present
    from datetime import date
    from india_quant.data.pipeline import DataPipeline
    today = date.today()
    logger.info(f"Scheduler started. Running pre-market catchup for {today}...")
    job_pre_market()

    app = create_health_app(scheduler)
    logger.info("Health endpoint at http://localhost:5001/health")
    app.run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    start_scheduler()
