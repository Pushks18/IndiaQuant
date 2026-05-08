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


def _build_default_artifact():
    """Construct the production LightGBMArtifact, or a StubArtifact fallback.

    Module-scope at app-init: if `models/global_tab/` is missing or any
    NIFTY pickle is unreadable, we log a warning and serve via Stub. The
    /global route then renders an "artifact: stub" banner so the user
    knows they're seeing a degraded forecast.
    """
    from pathlib import Path
    from india_quant.global_tab.forecaster import StubArtifact
    from india_quant.global_tab.lightgbm_artifact import LightGBMArtifact

    models_dir = Path("models/global_tab")
    required = ["NIFTY_direction.pkl", "NIFTY_magnitude_q10.pkl",
                "NIFTY_magnitude_q50.pkl", "NIFTY_magnitude_q90.pkl"]
    if not all((models_dir / f).exists() for f in required):
        logger.warning(
            "global_tab: LightGBM pickles not found under {} — falling back to StubArtifact",
            models_dir,
        )
        return StubArtifact()
    artifact = LightGBMArtifact(models_dir=models_dir)
    # Eager-validate (n_features_in_ guard) so a stale pickle from before a
    # FEATURE_COLUMNS bump falls back to Stub at boot rather than at first request.
    try:
        artifact._load_index("NIFTY")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "global_tab: LightGBM pickle validation failed ({}) — falling back to StubArtifact",
            exc,
        )
        return StubArtifact()
    return artifact


def _build_default_analog_index():
    """Load AnalogIndex from a cached pickle, or build it lazily from the DB.

    On first boot after retraining, the build can take a few seconds (assembles
    the same training_features frame the LightGBM trainer uses). We cache the
    result alongside the model pickles so subsequent boots are instant.
    """
    from datetime import date
    from pathlib import Path
    from india_quant.global_tab.analog_index import AnalogIndex

    cache_path = Path("models/global_tab/analog_index.pkl")
    if cache_path.exists():
        try:
            idx = AnalogIndex.load(cache_path)
            logger.info("AnalogIndex: loaded {} ({} rows)", cache_path, idx.n_samples)
            return idx
        except Exception as exc:  # noqa: BLE001
            logger.warning("AnalogIndex: cache load failed ({}); rebuilding", exc)

    try:
        from india_quant.data.db import get_session_factory
        idx = AnalogIndex.build_from_db(
            index="NIFTY",
            start=date(2021, 1, 1),
            end=date.today(),
            session_factory=get_session_factory(),
        )
        idx.save(cache_path)
        return idx
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "AnalogIndex: build failed ({}); /global will render with zero analog stats",
            exc,
        )
        return None


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # Build the forecaster artifact once at app-init. The first /global
    # request will lazily warm the LightGBM pickle cache.
    app.config["GLOBAL_TAB_ARTIFACT"] = _build_default_artifact()
    app.config["GLOBAL_TAB_ANALOG_INDEX"] = _build_default_analog_index()

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
        from datetime import datetime, date as date_cls
        import pandas as pd
        from flask import request
        from india_quant.signals.global_context import get_global_context
        from india_quant.data.fetchers.gift_nifty_fetcher import fetch_gift_nifty_quote
        from india_quant.data.db import get_session_factory
        from india_quant.global_tab.heatmap_view import render_heatmap_html
        from india_quant.global_tab.options_chain import load_chain_snapshot
        from india_quant.global_tab.orchestrator import build_global_view
        from india_quant.global_tab.types import Mode

        # ── Validate query params ─────────────────────────────────────────
        try:
            capital = float(request.args.get("capital", "100000"))
            if capital <= 0:
                raise ValueError
        except (TypeError, ValueError):
            return ("Bad capital — must be a positive number.", 400)

        mode_str = request.args.get("mode", "balanced").lower()
        try:
            mode = Mode(mode_str)
        except ValueError:
            return (f"Bad mode {mode_str!r} — expected aggressive | balanced | conservative.", 400)

        as_of = datetime.now()
        warnings: list[str] = []

        # ── Build providers ──────────────────────────────────────────────
        def _ctx_provider():
            try:
                return get_global_context()
            except Exception as exc:  # noqa: BLE001
                logger.warning("global_context fetch failed: {}", exc)
                warnings.append("Live global signals unavailable; tiles fall back to —")
                return type("EmptyCtx", (), {"signals": []})()

        def _gift_provider():
            q = fetch_gift_nifty_quote()
            if q is None:
                warnings.append("GIFT Nifty source unreachable; tile shows —")
            return q

        def _history_provider():
            try:
                return ddata.load_global_history(lookback_days=120)
            except Exception as exc:  # noqa: BLE001
                logger.warning("heatmap history load failed: {}", exc)
                warnings.append("Correlation heatmap unavailable; check DB connectivity")
                return pd.DataFrame()

        def _chain_loader(index, when, expiry):
            try:
                return load_chain_snapshot(
                    index, when, expiry, session_factory=get_session_factory(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("chain load failed for {}: {}", index, exc)
                return None

        view = build_global_view(
            as_of=as_of,
            mode=mode,
            capital=capital,
            context_provider=_ctx_provider,
            gift_provider=_gift_provider,
            history_provider=_history_provider,
            chain_loader=_chain_loader,
            model_artifact=app.config.get("GLOBAL_TAB_ARTIFACT"),
            analog_index=app.config.get("GLOBAL_TAB_ANALOG_INDEX"),
        )

        try:
            heatmap_html = render_heatmap_html(view.heatmap)
        except Exception as exc:  # noqa: BLE001
            logger.warning("heatmap render failed: {}", exc)
            heatmap_html = '<div class="heatmap-empty">Heatmap unavailable today.</div>'

        if not view.cards or all(c.direction.value == "no_trade" for c in view.cards):
            warnings.append("All cards NO_TRADE — check GIFT Nifty premium and chain coverage.")

        return render_template(
            "global_v2.html",
            as_of=as_of,
            briefing=view.briefing,
            heatmap_html=heatmap_html,
            cards=view.cards,
            mode=view.mode,
            capital=view.capital,
            data_warnings=warnings,
            artifact_name=view.artifact_paths.get("name", "stub"),
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
