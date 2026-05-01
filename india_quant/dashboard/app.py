"""Flask dashboard for the India Quant trading system.

Run:
  python -m india_quant.dashboard.app          # serve on http://localhost:5050
  python main.py --dashboard                   # same, via main entrypoint
"""
import threading
from datetime import date, datetime

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
        from india_quant.signals.screener import run_screener, TICKER_SECTOR
        try:    capital = float(request.args.get("capital", 100_000))
        except ValueError: capital = 100_000
        try:    risk_pct = float(request.args.get("risk", 1.0)) / 100.0
        except ValueError: risk_pct = 0.01
        try:    tgt_mult = float(request.args.get("target", 1.5))
        except ValueError: tgt_mult = 1.5
        try:    top_n = int(request.args.get("top", 20))
        except ValueError: top_n = 20

        plans = run_screener(
            capital_inr=capital,
            risk_per_trade_pct=risk_pct,
            top_n=top_n,
            t1_mult=tgt_mult,
            t2_mult=tgt_mult + 1.3,
        )
        actionable = [p for p in plans if p.get("bias") in ("LONG", "SHORT")]
        return render_template(
            "intraday.html",
            plans=actionable,
            universe_size=len(TICKER_SECTOR),
            capital=capital,
            risk_pct=risk_pct * 100,
            target_multiple=tgt_mult,
            top_n=top_n,
            today=ddata.latest_trading_date().isoformat(),
        )

    @app.route("/live")
    def live():
        from india_quant.signals import live_tracker
        return render_template(
            "live.html",
            today=datetime.now(live_tracker.IST).date().isoformat(),
            preds=live_tracker.get_predictions(),
        )

    @app.route("/api/live/data")
    def api_live_data():
        from india_quant.signals import live_tracker
        try:
            return jsonify(live_tracker.live_state(include_bars=True))
        except Exception as e:
            logger.error(f"live_state failed: {e}")
            return jsonify({"error": str(e), "items": [], "summary": {"n": 0}}), 500

    @app.route("/live/accuracy")
    def live_accuracy():
        from india_quant.signals import live_tracker
        from datetime import date as _date, timedelta as _td
        try:
            days = int(request.args.get("days", 30))
        except ValueError:
            days = 30
        end = _date.today()
        start = end - _td(days=days)
        return render_template(
            "live_accuracy.html",
            data=live_tracker.accuracy_summary(start=start, end=end),
            days=days,
        )

    @app.route("/run/screener-live", methods=["POST"])
    def run_screener_live():
        """Run the v4 screener and persist picks for today's tracker."""
        from india_quant.signals.screener import run_screener
        from india_quant.signals import live_tracker
        try:
            capital   = float(request.form.get("capital", 200_000))
            risk      = float(request.form.get("risk", 1.0)) / 100.0
            top_n     = int(request.form.get("top", 8))
            live_mode = request.form.get("live") == "on"
        except ValueError:
            capital, risk, top_n, live_mode = 200_000, 0.01, 8, False
        try:
            plans = run_screener(
                capital_inr=capital,
                risk_per_trade_pct=risk,
                top_n=top_n,
                live_mode=live_mode,
            )
            live_tracker.persist_today_predictions(plans)
        except Exception as e:
            logger.error(f"run_screener_live failed: {e}")
        return redirect(url_for("live"))

    @app.route("/global")
    def global_context_page():
        from india_quant.signals.global_context import get_global_context, instrument_levels
        from india_quant.signals.screener import run_screener

        try:    capital  = float(request.args.get("capital", 200_000))
        except ValueError: capital = 200_000
        try:    risk_pct = float(request.args.get("risk", 1.0)) / 100.0
        except ValueError: risk_pct = 0.01
        try:    top_n    = int(request.args.get("top", 10))
        except ValueError: top_n = 10

        ctx = get_global_context()

        signal_levels = {}
        for sig in ctx.signals:
            lvl = instrument_levels(sig, usdinr=ctx.usdinr, capital=capital, risk_pct=risk_pct)
            if lvl:
                signal_levels[sig.ticker] = lvl

        plans = run_screener(capital_inr=capital, risk_per_trade_pct=risk_pct, top_n=top_n)
        actionable = []
        for p in plans:
            if p.get("bias") not in ("LONG", "SHORT"):
                continue
            if   ctx.regime == "RISK_ON"  and p["bias"] == "LONG":  p["global_aligned"] = True
            elif ctx.regime == "RISK_OFF" and p["bias"] == "SHORT": p["global_aligned"] = True
            elif ctx.regime == "NEUTRAL":                           p["global_aligned"] = True
            else:                                                   p["global_aligned"] = False
            actionable.append(p)

        return render_template(
            "global_context.html",
            ctx=ctx,
            signal_levels=signal_levels,
            plans=actionable,
            capital=capital,
            risk_pct=risk_pct * 100,
            top_n=top_n,
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
