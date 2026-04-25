"""Telegram notification bot for daily picks and alerts."""
import json
from pathlib import Path

from loguru import logger

from india_quant.config import cfg


class TelegramNotifier:
    BASE_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, chat_id: str = None):
        self.token = cfg.telegram_bot_token
        self.chat_id = chat_id
        self._enabled = bool(self.token and self.token != "")

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self._enabled:
            logger.info(f"[Telegram] (disabled) Would send: {text[:80]}")
            return False
        import requests
        url = self.BASE_URL.format(token=self.token, method="sendMessage")
        try:
            resp = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }, timeout=10)
            return resp.json().get("ok", False)
        except Exception as e:
            logger.error(f"[Telegram] Send failed: {e}")
            return False

    def send_file(self, file_path: str, caption: str = "") -> bool:
        if not self._enabled:
            logger.info(f"[Telegram] (disabled) Would send file: {file_path}")
            return False
        import requests
        url = self.BASE_URL.format(token=self.token, method="sendDocument")
        try:
            with open(file_path, "rb") as f:
                resp = requests.post(url, data={
                    "chat_id": self.chat_id,
                    "caption": caption,
                }, files={"document": f}, timeout=30)
            return resp.json().get("ok", False)
        except Exception as e:
            logger.error(f"[Telegram] File send failed: {e}")
            return False

    def send_daily_picks(self, report_json_path: str):
        """Send top 3 picks from daily report JSON."""
        try:
            data = json.loads(Path(report_json_path).read_text())
            trades = data.get("top_trades", [])[:3]
            macro = data.get("macro", {})

            lines = [
                f"🇮🇳 <b>India Quant Daily — {data.get('date')}</b>",
                f"Regime: <b>{macro.get('regime_label', '--')}</b> | VIX: {macro.get('india_vix', '--')}",
                "",
            ]
            for i, t in enumerate(trades, 1):
                emoji = "🟢" if t.get("direction") == "long" else "🔴"
                lines.append(
                    f"{emoji} <b>{i}. {t.get('ticker')}</b> {t.get('direction', '').upper()}\n"
                    f"   Entry: {t.get('entry_zone')} | SL: {t.get('stop_loss')} | T1: {t.get('target_1')} | R:R: {t.get('risk_reward')}\n"
                    f"   {t.get('rationale', '')[:100]}"
                )
            self.send_message("\n".join(lines))
        except Exception as e:
            logger.error(f"[Telegram] Daily picks failed: {e}")

    def alert_drawdown(self, drawdown_pct: float):
        """Alert if portfolio drawdown hits 10%."""
        if drawdown_pct <= -0.10:
            self.send_message(
                f"⚠️ <b>DRAWDOWN ALERT</b>\nPortfolio drawdown: {drawdown_pct:.1%}\nConsider reducing exposure."
            )

    def alert_pipeline_failure(self, component: str, error: str):
        """Alert on data pipeline failure."""
        self.send_message(
            f"🔴 <b>Pipeline Failure: {component}</b>\n<code>{error[:200]}</code>"
        )

    def alert_ic_decay(self, factor_name: str, decay_pct: float):
        """Alert if model IC decays significantly."""
        self.send_message(
            f"📉 <b>IC Decay Alert: {factor_name}</b>\nIC has decayed {decay_pct:.1f}% — review model"
        )
