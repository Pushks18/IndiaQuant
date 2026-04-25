"""
India Quant Trading Assistant — entry point.

Usage:
  python main.py                  # show system status
  python main.py --pipeline       # run today's data pipeline
  python main.py --report daily   # generate daily report
  python main.py --scheduler      # start the live scheduler
  python main.py --dashboard      # launch web dashboard (http://localhost:5050)
  python main.py --debate TICKER  # run a debate + trade proposal for one ticker
"""
import argparse
from loguru import logger

from india_quant.config import cfg


def show_status():
    logger.info("India Quant Trading Assistant")
    logger.info("Config loaded.")
    logger.info("System ready. Use --help for options.")


def main():
    parser = argparse.ArgumentParser(description="India Quant Trading Assistant")
    parser.add_argument("--pipeline", action="store_true", help="Run today's data pipeline")
    parser.add_argument("--report", choices=["daily", "weekly", "monthly"], help="Generate a report")
    parser.add_argument("--scheduler", action="store_true", help="Start the live scheduler")
    parser.add_argument("--dashboard", action="store_true", help="Launch web dashboard")
    parser.add_argument("--port", type=int, default=5050, help="Dashboard port")
    parser.add_argument("--debate", metavar="TICKER", help="Run debate + trade proposal for a ticker (e.g. RELIANCE)")
    args = parser.parse_args()

    if args.pipeline:
        from india_quant.data.pipeline import DataPipeline
        from datetime import date
        DataPipeline.run_pre_market(date.today().isoformat())
        DataPipeline.run_post_market(date.today().isoformat())
    elif args.report:
        logger.info(f"Generating {args.report} report...")
        if args.report == "daily":
            from india_quant.reports.daily_report import generate_daily_report
            generate_daily_report()
        elif args.report == "weekly":
            from india_quant.reports.weekly_report import generate_weekly_report
            generate_weekly_report()
        elif args.report == "monthly":
            from india_quant.reports.monthly_report import generate_monthly_report
            generate_monthly_report()
    elif args.scheduler:
        from india_quant.scheduler import start_scheduler
        start_scheduler()
    elif args.dashboard:
        from india_quant.dashboard.app import main as run_dashboard
        run_dashboard(port=args.port)
    elif args.debate:
        from india_quant.agents.judge import run_debate
        from india_quant.agents.trader import TraderAgent
        ticker = args.debate.upper()
        if not (ticker.endswith(".NS") or ticker.endswith(".BO")):
            ticker = ticker + ".NS"
        debate = run_debate(ticker)
        proposal = TraderAgent().propose_trade(debate)
        logger.info(f"Verdict: {debate['judge'].get('verdict')} (conviction {debate['judge'].get('conviction')})")
        logger.info(f"Proposal: {proposal}")
    else:
        show_status()


if __name__ == "__main__":
    main()
