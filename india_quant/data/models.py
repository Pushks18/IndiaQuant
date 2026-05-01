import uuid
from datetime import datetime, date

from sqlalchemy import (
    Column, String, Float, Integer, Date, DateTime, Text, Boolean,
    UniqueConstraint, Index, func,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class PriceData(Base):
    __tablename__ = "price_data"

    ticker = Column(String(20), primary_key=True)
    datetime = Column(DateTime(timezone=True), primary_key=True)
    interval = Column(String(5), primary_key=True)  # 1d, 1h, 1m, live
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    source = Column(String(30))

    __table_args__ = (
        Index("ix_price_ticker_dt", "ticker", "datetime"),
    )


class OptionChain(Base):
    __tablename__ = "option_chain"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    underlying = Column(String(20), nullable=False)
    trade_date = Column(Date, nullable=False)
    expiry = Column(Date, nullable=False)
    strike = Column(Float, nullable=False)
    option_type = Column(String(2), nullable=False)  # CE / PE
    last_price = Column(Float)
    bid = Column(Float)
    ask = Column(Float)
    iv = Column(Float)
    open_interest = Column(Float)
    oi_change = Column(Float)
    volume = Column(Float)
    timestamp = Column(DateTime(timezone=True), default=func.now())

    __table_args__ = (
        UniqueConstraint("underlying", "trade_date", "expiry", "strike", "option_type",
                         name="uq_option_chain"),
        Index("ix_option_underlying_date", "underlying", "trade_date"),
    )


class NewsArticle(Base):
    __tablename__ = "news_article"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    source = Column(String(50))
    tickers = Column(ARRAY(String))
    headline = Column(Text)
    content = Column(Text)
    url = Column(String(2000), unique=True)
    sentiment_score = Column(Float)  # -1 to +1


class SentimentAggregate(Base):
    __tablename__ = "sentiment_aggregate"

    ticker = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True)
    avg_score = Column(Float)
    article_count = Column(Integer)
    updated_at = Column(DateTime(timezone=True), default=func.now())


class FactorScores(Base):
    __tablename__ = "factor_scores"

    ticker = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True)

    # Momentum
    momentum_12_1 = Column(Float)
    momentum_1 = Column(Float)
    momentum_3 = Column(Float)

    # Value
    size = Column(Float)
    value_bm = Column(Float)
    earnings_yield = Column(Float)

    # Quality
    profitability_roe = Column(Float)
    gross_profitability = Column(Float)
    investment_ag = Column(Float)

    # Volatility
    idiosyncratic_vol = Column(Float)
    realized_vol = Column(Float)
    vol_of_vol = Column(Float)

    # Liquidity
    liquidity_amihud = Column(Float)
    turnover = Column(Float)

    # Options
    iv_skew = Column(Float)
    iv_spread = Column(Float)
    vrp = Column(Float)
    oi_flow = Column(Float)


class SignalLabels(Base):
    __tablename__ = "signal_labels"

    ticker = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True)
    horizon = Column(String(5), primary_key=True)  # 1d, 5d, 21d
    future_return = Column(Float)            # Realized forward return (null until time passes)
    predicted_return = Column(Float)         # ML prediction at this date
    signal_rank = Column(Integer)            # Cross-sectional rank of predicted_return (1 = highest)
    class_label = Column(Integer)            # 1 = up, 0 = down (from realized return)
    created_at = Column(DateTime(timezone=True), default=func.now())


class AnalystReport(Base):
    __tablename__ = "analyst_report"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(20))
    date = Column(Date)
    agent_name = Column(String(50))
    report_json = Column(Text)
    created_at = Column(DateTime(timezone=True), default=func.now())

    __table_args__ = (
        Index("ix_analyst_ticker_date", "ticker", "date"),
    )


class DebateResult(Base):
    __tablename__ = "debate_result"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(20))
    date = Column(Date)
    bull_report = Column(Text)
    bear_report = Column(Text)
    judge_verdict = Column(Text)
    created_at = Column(DateTime(timezone=True), default=func.now())


class TradeProposal(Base):
    __tablename__ = "trade_proposal"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String(20))
    date = Column(Date)
    instrument = Column(String(20))
    direction = Column(String(5))
    entry_price = Column(Float)
    stop_loss = Column(Float)
    target_1 = Column(Float)
    target_2 = Column(Float)
    risk_reward = Column(Float)
    position_size_pct = Column(Float)
    rationale = Column(Text)
    risk_status = Column(String(10))  # APPROVED / REJECTED / MODIFIED
    created_at = Column(DateTime(timezone=True), default=func.now())


class IntradayPrediction(Base):
    """Persisted screener pick + intraday outcome tracking.

    One row per (date, ticker, bias). Status progresses through the day:
    PENDING → TRIGGERED → TARGET1 / TARGET2 / STOPPED.
    """
    __tablename__ = "intraday_prediction"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(Date, nullable=False, index=True)
    ticker = Column(String(20), nullable=False, index=True)
    bias = Column(String(8), nullable=False)               # LONG / SHORT
    score = Column(Float)
    score_long = Column(Float)
    score_short = Column(Float)
    prev_close = Column(Float)
    atr = Column(Float)
    atr_pct = Column(Float)
    trigger = Column(Float, nullable=False)
    stop = Column(Float, nullable=False)
    target1 = Column(Float, nullable=False)
    target2 = Column(Float, nullable=False)
    qty = Column(Integer)
    max_loss_inr = Column(Float)
    profit1_inr = Column(Float)
    profit2_inr = Column(Float)
    verdict = Column(String(16))
    conviction = Column(Integer)

    status = Column(String(12), default="PENDING")          # PENDING/TRIGGERED/TARGET1/TARGET2/STOPPED
    triggered_at = Column(DateTime(timezone=True))
    exit_at = Column(DateTime(timezone=True))
    exit_price = Column(Float)
    exit_reason = Column(String(16))                        # TARGET1/TARGET2/STOPPED/EOD
    max_favorable_pct = Column(Float)                       # peak unrealized gain (post-trigger)
    max_adverse_pct = Column(Float)                         # worst unrealized loss (post-trigger)
    realized_pnl_inr = Column(Float)
    meta = Column(Text)                                     # JSON dump of full plan

    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("date", "ticker", "bias", name="uq_intraday_pred_day_tkr_side"),
        Index("ix_intraday_pred_date", "date"),
    )


class VolatilityData(Base):
    __tablename__ = "volatility_data"

    ticker = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True)
    realized_vol_1d = Column(Float)
    har_forecast_1d = Column(Float)
    garch_forecast_1d = Column(Float)
    heston_v0 = Column(Float)
    heston_kappa = Column(Float)
    heston_theta = Column(Float)
    heston_sigma = Column(Float)
    heston_rho = Column(Float)


class GlobalSignal(Base):
    __tablename__ = "global_signals"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date       = Column(Date, nullable=False)
    ticker     = Column(String(20), nullable=False)
    label      = Column(String(50))
    group      = Column(String(20))
    pct_1d     = Column(Float)
    pct_5d     = Column(Float)
    corr_30d   = Column(Float)
    corr_90d   = Column(Float)
    regime     = Column(String(10))
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("date", "ticker", name="uq_global_signal_date_ticker"),
        Index("ix_global_signal_date", "date"),
    )
