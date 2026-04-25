import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Required env var '{key}' is missing. Add it to .env")
    return val


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)


@dataclass
class Config:
    shoonya_api_key: str
    shoonya_user_id: str
    shoonya_password: str
    shoonya_totp_secret: str
    shoonya_vendor_code: str
    shoonya_imei: str
    anthropic_api_key: str
    finnhub_key: str
    newsapi_key: str
    database_url: str
    telegram_bot_token: str
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-oss-20b:free"
    openrouter_model_fallback: str = "openai/gpt-oss-120b:free"

    # Market calendar — RBI policy dates 2025 (approximate)
    rbi_policy_dates: tuple = (
        "2025-02-07", "2025-04-09", "2025-06-06",
        "2025-08-08", "2025-10-08", "2025-12-05",
    )

    # NSE trading hours (IST)
    market_open: str = "09:15"
    market_close: str = "15:30"
    pre_open_start: str = "09:00"
    post_close_end: str = "16:00"

    def __repr__(self) -> str:
        return (
            f"Config(angel_client_id={self.angel_client_id}, "
            f"db={self.database_url[:30]}..., "
            f"anthropic_key={'***' if self.anthropic_api_key else 'MISSING'})"
        )


def load_config() -> Config:
    return Config(
        shoonya_api_key=_require("SHOONYA_API_KEY"),
        shoonya_user_id=_require("SHOONYA_USER_ID"),
        shoonya_password=_require("SHOONYA_PASSWORD"),
        shoonya_totp_secret=_require("SHOONYA_TOTP_SECRET"),
        shoonya_vendor_code=_require("SHOONYA_VENDOR_CODE"),
        shoonya_imei=_optional("SHOONYA_IMEI", "mac"),
        anthropic_api_key=_optional("ANTHROPIC_API_KEY"),
        finnhub_key=_require("FINNHUB_KEY"),
        newsapi_key=_require("NEWSAPI_KEY"),
        database_url=_require("DATABASE_URL"),
        telegram_bot_token=_optional("TELEGRAM_BOT_TOKEN"),
        openrouter_api_key=_optional("OPENROUTER_API_KEY"),
        openrouter_model=_optional("OPENROUTER_MODEL", "openai/gpt-oss-20b:free"),
        openrouter_model_fallback=_optional("OPENROUTER_MODEL_FALLBACK", "openai/gpt-oss-120b:free"),
    )


try:
    cfg = load_config()
except EnvironmentError as e:
    import sys
    print(f"[CONFIG ERROR] {e}")
    print("Copy .env.example to .env and fill in your credentials.")
    sys.exit(1)
