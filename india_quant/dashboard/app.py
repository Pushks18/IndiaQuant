"""Flask dashboard for the India Quant trading system.

Run:
  python -m india_quant.dashboard.app          # serve on http://localhost:5050
  python main.py --dashboard                   # same, via main entrypoint
"""
import threading
from datetime import date

from flask import Flask, jsonify, render_template, request, redirect, url_for
from loguru import logger

from india_quant.dashboard import data as ddata


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # ── Pages ────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            macro=ddata.macro_snapshot(),
            movers=ddata.top_movers(8),
            health=ddata.data_health(),
            tickers=ddata.all_tickers(),
            today=ddata.latest_trading_date().isoformat(),
        )

    @app.route("/signals")
    def signals():
        return render_template(
            "signals.html",
            rows=ddata.signal_summary(),
            today=ddata.latest_trading_date().isoformat(),
        )

    @app.route("/proposals")
    def proposals():
        rows = ddata.latest_proposals(limit=100)
        return render_template("proposals.html", rows=rows)

    @app.route("/intraday")
    def intraday():
        from india_quant.signals.intraday import todays_intraday_plan
        try:
            capital = float(request.args.get("capital", 100_000))
        except ValueError:
            capital = 100_000
        try:
            risk_pct = float(request.args.get("risk", 1.0)) / 100.0
        except ValueError:
            risk_pct = 0.01
        try:
            tgt_mult = float(request.args.get("target", 1.5))
        except ValueError:
            tgt_mult = 1.5
        plans = todays_intraday_plan(
            capital_inr=capital,
            risk_per_trade_pct=risk_pct,
            top_n=15,
            target_multiple=tgt_mult,
        )
        return render_template(
            "intraday.html",
            plans=plans,
            capital=capital,
            risk_pct=risk_pct * 100,
            target_multiple=tgt_mult,
            today=ddata.latest_trading_date().isoformat(),
        )

    @app.route("/debates")
    def debates():
        rows = ddata.latest_debate(limit=50)
        return render_template("debates.html", rows=rows)

    @app.route("/ticker/<ticker>")
    def ticker_detail(ticker):
        ticker = ticker.upper()
        if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
            ticker = ticker + ".NS"
        history = ddata.price_history(ticker, days=180)
        reports = ddata.latest_analyst_reports(ticker, limit=8)
        debates = ddata.latest_debate(ticker, limit=5)
        proposals_ = ddata.latest_proposals(ticker, limit=5)
        return render_template(
            "ticker.html",
            ticker=ticker,
            history=history,
            reports=reports,
            debates=debates,
            proposals=proposals_,
        )

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "data": ddata.data_health()})

    # ── Actions ──────────────────────────────────────────────────────────

    @app.route("/run/debate", methods=["POST"])
    def run_debate():
        ticker = (request.form.get("ticker") or "").strip().upper()
        if not ticker:
            return redirect(url_for("index"))
        if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
            ticker = ticker + ".NS"
        from india_quant.agents.judge import run_debate as _run_debate
        from india_quant.agents.trader import TraderAgent

        try:
            debate = _run_debate(ticker)
            TraderAgent().propose_trade(debate)
        except Exception as e:
            logger.error(f"run_debate failed: {e}")
        return redirect(url_for("ticker_detail", ticker=ticker.replace(".NS", "")))

    @app.route("/run/factors", methods=["POST"])
    def run_factors():
        from india_quant.signals.factors import FactorEngine
        try:
            FactorEngine().compute_all(date.today().isoformat())
        except Exception as e:
            logger.error(f"compute_all failed: {e}")
        return redirect(url_for("signals"))

    @app.route("/run/pipeline", methods=["POST"])
    def run_pipeline():
        """Run pre-market data pipeline async so the request returns fast."""
        from india_quant.data.pipeline import DataPipeline

        def _bg():
            try:
                DataPipeline.run_pre_market(date.today().isoformat())
                DataPipeline.run_post_market(date.today().isoformat())
            except Exception as e:
                logger.error(f"Pipeline failed: {e}")

        threading.Thread(target=_bg, daemon=True).start()
        return redirect(url_for("index"))

    return app


def main(host: str = "0.0.0.0", port: int = 5050):
    app = create_app()
    logger.info(f"India Quant dashboard → http://localhost:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
