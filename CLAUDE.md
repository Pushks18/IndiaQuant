# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

Graph auto-updates on file changes via hooks. Use `detect_changes` → `get_affected_flows` → `query_graph(tests_for)` as the standard review workflow.

---

## Commands

```bash
# Setup
cp .env.example .env          # fill in credentials
pip install -r requirements.txt

# Run
python main.py                  # show system status
python main.py --pipeline       # run today's pre-market data pipeline
python main.py --report daily   # generate daily report
python main.py --report weekly
python main.py --report monthly
python main.py --scheduler      # start live APScheduler + Flask health endpoint

# Database (requires TimescaleDB via Docker)
docker run -d --name timescale -p 5432:5432 \
  -e POSTGRES_PASSWORD=yourpassword \
  timescale/timescaledb:latest-pg15

# Tests
pytest tests/                          # full suite
pytest tests/ -k test_config_loads     # single test
```

### Test categories

Tests in `tests/test_integration.py` fall into two groups:

- **No infra needed** (run offline): `test_config_loads`, `test_all_models_import`, `test_yfinance_fetcher_imports`, `test_factor_engine_imports`, `test_volatility_har_rv`, `test_backtest_cost_model`, `test_harvey_liu_zhu_gate`, `test_risk_agent_kelly`, `test_risk_agent_reject_low_rr`, `test_scheduler_creates_all_jobs`, `test_telegram_notifier_disabled`, `test_report_modules_import`
- **Requires live DB / credentials / Docker**: Angel SmartAPI login, FinBERT model load (~400 MB first run), TimescaleDB hypertable creation, ML training, walk-forward backtest

> **Important:** `india_quant/config.py` calls `load_config()` at module import time and calls `sys.exit(1)` if any required env var is missing. This means **all** tests — even the "no infra needed" ones — require a valid `.env` file. Dummy values (e.g. `ANGEL_API_KEY=x`) satisfy the import; real values are only needed for live-data tests.

## Architecture

This is a **multi-agent quantitative trading system** for Indian equities (NSE/BSE). All scheduling runs in IST (Asia/Kolkata).

### Data flow

```
External sources → Fetchers → PostgreSQL/TimescaleDB
                                    ↓
                              Factor Engine → ML Models → Signal Labels
                                    ↓
                              Analyst Agents (LLM ReAct loop)
                                    ↓
                         Bull/Bear Debate → Judge → Trader → Risk Agent
                                    ↓
                              Reports (HTML/PDF) → Telegram
```

### Key layers

**Layer 1 — Data (`india_quant/data/`)**
- `fetchers/yfinance_fetcher.py` — EOD/intraday prices for all Nifty-50 tickers (`.NS` suffix)
- `fetchers/angel_fetcher.py` — live prices via Angel One SmartAPI (WebSocket)
- `fetchers/nse_options_fetcher.py` — NSE option chain snapshots
- `fetchers/news_fetcher.py` — news from Google RSS/Finnhub/NewsAPI, scored with FinBERT
- `pipeline.py` — orchestrates all fetchers; `DataPipeline.run_pre_market/intraday/post_market/weekly_maintenance`
- `db.py` — SQLAlchemy engine + `get_session()` context manager
- `models.py` — ORM models: `PriceData`, `OptionChain`, `NewsArticle`, `FactorScores`, `SignalLabels`, `AnalystReport`, `DebateResult`, `TradeProposal`, `VolatilityData`
- `quality_monitor.py` — `run_daily_quality_checks()`: validates price completeness, anomaly detection, options freshness, factor score coverage, sentiment scores; returns list of alert strings (empty = pass)

**Layer 2 — Signals (`india_quant/signals/`)**
- `factors.py` — `FactorEngine`: momentum, value, quality, volatility, liquidity, options factors → `factor_scores` table
- `ml_models.py` — `ReturnPredictor`: XGBoost + LightGBM trained on factor scores; walk-forward validation; SHAP explanations; retrained weekly
- `volatility.py` — `VolatilityEngine`: realized vol, HAR-RV, GARCH, Heston calibration
- `options_signals.py` — `OptionsSignalEngine`: IV spread/skew, VRP, OI flow, PCR, max pain

**Layer 3 — Agents (`india_quant/agents/`)**
- `base.py` — `BaseAnalystAgent`: ReAct loop (max 5 tool calls), FinCon-style memory (lessons → principles), stores reports to `analyst_report` table. All agents use `claude-sonnet-4-20250514`. Also houses all shared Pydantic report schemas (`TechnicalReport`, `FundamentalReport`, `SentimentReport`, `MacroReport`, `AnalystSummary`) — specialist agents import their output schema from here.
- `technical_analyst.py`, `fundamental_analyst.py`, `sentiment_analyst.py`, `macro_analyst.py` — specialised analysts extending `BaseAnalystAgent`; each registers its own tool functions via `_tool_registry()`
- `researcher.py` — `BullAgent` / `BearAgent`: one-shot debate participants
- `judge.py` — `JudgeAgent` synthesises debate into structured verdict; `run_debate()` orchestrates the full Bull→Bear→Judge flow and stores `DebateResult`
- `trader.py` — `TraderAgent`: converts judge verdict into a concrete `TradeProposal`
- `risk_agent.py` — `RiskAgent`: Kelly sizing, R:R gate (minimum 1.5), F&O ban check, circuit proximity, SEBI blackout window, expiry risk, portfolio heat, VaR. `review_trade()` returns a `RiskReview` Pydantic object (`.status`, `.reason`)
- `memory_manager.py` — `MemoryManager`: persists trade outcomes and distilled principles per agent as JSON files in `india_quant/agents/memory/` (one file per agent, keyed by agent name)

**Layer 4 — Backtest (`india_quant/backtest/`)**
- `engine.py` — `IndiaBacktestEngine`: India-specific cost model (STT, brokerage, SEBI charges), walk-forward backtest, Sharpe/Sortino/Calmar/max-drawdown metrics
- `validation.py` — Harvey-Liu-Zhu t-stat gate (threshold 3.0), McLean-Pontiff decay monitor

**Layer 5 — Reports (`india_quant/reports/`)**
- `daily_report.py`, `weekly_report.py`, `monthly_report.py` — Jinja2 HTML → WeasyPrint PDF
- `telegram_bot.py` — `TelegramNotifier`: send reports/alerts; gracefully no-ops if `TELEGRAM_BOT_TOKEN` is unset

**Scheduler (`india_quant/scheduler.py`)**
APScheduler (BackgroundScheduler, IST) + Flask health endpoint at `localhost:5001/health`:
- `08:00` pre-market (Mon–Fri)
- `09:15–15:30` intraday every 5 min (Mon–Fri)
- `16:00` post-market (Mon–Fri)
- `18:00` daily report (Mon–Fri)
- `22:00 Sun` weekly maintenance (ML retrain + data quality)

## Required environment variables

See `.env.example`. All are required except `TELEGRAM_BOT_TOKEN`:

| Variable | Source |
|---|---|
| `ANGEL_API_KEY`, `ANGEL_CLIENT_ID`, `ANGEL_PASSWORD`, `ANGEL_TOTP_SECRET` | smartapi.angelbroking.com |
| `ANTHROPIC_API_KEY` | console.anthropic.com |
| `FINNHUB_KEY` | finnhub.io |
| `NEWSAPI_KEY` | newsapi.org |
| `DATABASE_URL` | PostgreSQL+TimescaleDB connection string |

`india_quant/config.py` calls `load_config()` at import time and exits immediately if any required variable is missing.

## India-specific constraints to always respect

- NSE trading hours: 09:15–15:30 IST; pre-open 09:00; post-close 16:00
- F&O ban list check before any leveraged trade
- SEBI insider blackout windows checked before new positions
- Weekly expiry = Thursday; monthly expiry within 3 days → elevated vol flag
- Circuit limits (5%/10%/20%) affect position sizing
- RBI policy dates (hardcoded in `Config.rbi_policy_dates`) trigger macro regime re-evaluation
- Backtest cost model must include STT, brokerage, SEBI charges — delivery vs intraday rates differ

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
