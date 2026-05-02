"""F&O eligible universe for Indian equities (NSE).

~180 tickers approved for futures and options trading by SEBI/NSE.
Source: NSE F&O segment list (last quarterly review).

Public exports:
    FO_TICKERS         — list[str], with .NS suffix
    TICKER_SECTOR      — dict[str, str], maps ticker → sector code
    SECTOR_INDEX_TICKER — dict[str, str], maps sector → tradable index ticker
"""
from __future__ import annotations

# ─── Sector → Nifty index mapping ─────────────────────────────────────────────

SECTOR_INDEX_TICKER: dict[str, str] = {
    "BANK":     "^NSEBANK",
    "IT":       "^CNXIT",
    "INFRA":    "^CNXINFRA",
    "PHARMA":   "^CNXPHARMA",
    "REALTY":   "^CNXREALTY",
    "ENERGY":   "^CNXENERGY",
    "AUTO":     "^CNXAUTO",
    "METAL":    "^CNXMETAL",
    "FMCG":     "^CNXFMCG",
    "MEDIA":    "^CNXMEDIA",
    "PSUBANK":  "^CNXPSUBANK",
    "FINSERV":  "NIFTYFINSER.NS",
}

# ─── F&O ticker → sector ──────────────────────────────────────────────────────

TICKER_SECTOR: dict[str, str] = {
    # Banks
    "HDFCBANK.NS": "BANK", "ICICIBANK.NS": "BANK", "KOTAKBANK.NS": "BANK",
    "AXISBANK.NS": "BANK", "INDUSINDBK.NS": "BANK", "BANDHANBNK.NS": "BANK",
    "FEDERALBNK.NS": "BANK", "IDFCFIRSTB.NS": "BANK", "RBLBANK.NS": "BANK",
    "AUBANK.NS": "BANK",
    # PSU Banks
    "SBIN.NS": "PSUBANK", "PNB.NS": "PSUBANK", "BANKBARODA.NS": "PSUBANK",
    "CANBK.NS": "PSUBANK", "UNIONBANK.NS": "PSUBANK", "BANKINDIA.NS": "PSUBANK",
    # Financial services (non-bank)
    "BAJFINANCE.NS": "FINSERV", "BAJAJFINSV.NS": "FINSERV",
    "HDFCLIFE.NS": "FINSERV", "SBILIFE.NS": "FINSERV",
    "ICICIPRULI.NS": "FINSERV", "ICICIGI.NS": "FINSERV",
    "CHOLAFIN.NS": "FINSERV", "MUTHOOTFIN.NS": "FINSERV",
    "LICHSGFIN.NS": "FINSERV", "MFSL.NS": "FINSERV",
    "PFC.NS": "FINSERV", "RECLTD.NS": "FINSERV",
    "IRFC.NS": "FINSERV", "MANAPPURAM.NS": "FINSERV",
    "POLICYBZR.NS": "FINSERV", "PAYTM.NS": "FINSERV",
    # IT
    "TCS.NS": "IT", "INFY.NS": "IT", "WIPRO.NS": "IT", "HCLTECH.NS": "IT",
    "TECHM.NS": "IT", "LTIM.NS": "IT", "PERSISTENT.NS": "IT",
    "MPHASIS.NS": "IT", "COFORGE.NS": "IT", "OFSS.NS": "IT",
    # Pharma
    "SUNPHARMA.NS": "PHARMA", "DRREDDY.NS": "PHARMA", "CIPLA.NS": "PHARMA",
    "DIVISLAB.NS": "PHARMA", "LUPIN.NS": "PHARMA", "AUROPHARMA.NS": "PHARMA",
    "TORNTPHARM.NS": "PHARMA", "ZYDUSLIFE.NS": "PHARMA",
    "BIOCON.NS": "PHARMA", "GLENMARK.NS": "PHARMA", "ALKEM.NS": "PHARMA",
    "LAURUSLABS.NS": "PHARMA", "ABBOTINDIA.NS": "PHARMA", "IPCALAB.NS": "PHARMA",
    # Auto
    "MARUTI.NS": "AUTO", "TATAMOTORS.NS": "AUTO", "M&M.NS": "AUTO",
    "BAJAJ-AUTO.NS": "AUTO", "EICHERMOT.NS": "AUTO", "HEROMOTOCO.NS": "AUTO",
    "TVSMOTOR.NS": "AUTO", "ASHOKLEY.NS": "AUTO", "BOSCHLTD.NS": "AUTO",
    "MOTHERSON.NS": "AUTO", "BALKRISIND.NS": "AUTO", "MRF.NS": "AUTO",
    "EXIDEIND.NS": "AUTO", "ESCORTS.NS": "AUTO",
    # FMCG
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG", "DABUR.NS": "FMCG", "MARICO.NS": "FMCG",
    "GODREJCP.NS": "FMCG", "COLPAL.NS": "FMCG", "TATACONSUM.NS": "FMCG",
    "VBL.NS": "FMCG", "UBL.NS": "FMCG", "MCDOWELL-N.NS": "FMCG",
    # Energy / Oil & Gas
    "RELIANCE.NS": "ENERGY", "ONGC.NS": "ENERGY", "BPCL.NS": "ENERGY",
    "IOC.NS": "ENERGY", "GAIL.NS": "ENERGY", "HINDPETRO.NS": "ENERGY",
    "PETRONET.NS": "ENERGY", "OIL.NS": "ENERGY", "MGL.NS": "ENERGY",
    "IGL.NS": "ENERGY", "GUJGASLTD.NS": "ENERGY",
    # Metals
    "TATASTEEL.NS": "METAL", "JSWSTEEL.NS": "METAL", "HINDALCO.NS": "METAL",
    "VEDL.NS": "METAL", "COALINDIA.NS": "METAL", "JINDALSTEL.NS": "METAL",
    "NMDC.NS": "METAL", "SAIL.NS": "METAL", "NATIONALUM.NS": "METAL",
    "APLAPOLLO.NS": "METAL", "HINDZINC.NS": "METAL", "RATNAMANI.NS": "METAL",
    # Realty
    "DLF.NS": "REALTY", "GODREJPROP.NS": "REALTY", "OBEROIRLTY.NS": "REALTY",
    "PRESTIGE.NS": "REALTY", "LODHA.NS": "REALTY", "BRIGADE.NS": "REALTY",
    # Infra / Construction / Cement
    "LT.NS": "INFRA", "ULTRACEMCO.NS": "INFRA", "GRASIM.NS": "INFRA",
    "AMBUJACEM.NS": "INFRA", "ACC.NS": "INFRA", "SHREECEM.NS": "INFRA",
    "DALBHARAT.NS": "INFRA", "RAMCOCEM.NS": "INFRA",
    "ADANIPORTS.NS": "INFRA", "GMRINFRA.NS": "INFRA",
    "SIEMENS.NS": "INFRA", "ABB.NS": "INFRA", "BHEL.NS": "INFRA",
    "CUMMINSIND.NS": "INFRA", "POLYCAB.NS": "INFRA",
    # Power / Utilities
    "NTPC.NS": "ENERGY", "POWERGRID.NS": "ENERGY",
    "ADANIENT.NS": "ENERGY", "ADANIGREEN.NS": "ENERGY",
    "TATAPOWER.NS": "ENERGY", "TORNTPOWER.NS": "ENERGY", "JSWENERGY.NS": "ENERGY",
    # Media / Telecom
    "BHARTIARTL.NS": "TELECOM", "IDEA.NS": "TELECOM", "INDIAMART.NS": "MEDIA",
    "ZEEL.NS": "MEDIA", "SUNTV.NS": "MEDIA", "PVRINOX.NS": "MEDIA",
    # Consumer durables / Retail
    "TITAN.NS": "FMCG", "ASIANPAINT.NS": "FMCG", "BERGEPAINT.NS": "FMCG",
    "PIDILITIND.NS": "FMCG", "HAVELLS.NS": "FMCG", "VOLTAS.NS": "FMCG",
    "DIXON.NS": "IT", "WHIRLPOOL.NS": "FMCG",
    "TRENT.NS": "FMCG", "ABFRL.NS": "FMCG", "PAGEIND.NS": "FMCG",
    "JUBLFOOD.NS": "FMCG", "DMART.NS": "FMCG",
    # Chemicals / specialty
    "PIIND.NS": "PHARMA", "DEEPAKNTR.NS": "PHARMA", "SRF.NS": "PHARMA",
    "ATUL.NS": "PHARMA", "COROMANDEL.NS": "PHARMA", "UPL.NS": "PHARMA",
    # Misc / capital goods / others
    "ICICIBANK.NS": "BANK",  # duplicate-safe
    "INDIGO.NS": "INFRA",
    "CONCOR.NS": "INFRA",
    "CROMPTON.NS": "FMCG",
    "ASTRAL.NS": "INFRA",
    "BATAINDIA.NS": "FMCG",
    "BHARATFORG.NS": "AUTO",
    "TIINDIA.NS": "AUTO",
    "ABCAPITAL.NS": "FINSERV",
    "ANGELONE.NS": "FINSERV",
    "CDSL.NS": "FINSERV",
    "BSE.NS": "FINSERV",
    "POONAWALLA.NS": "FINSERV",
    "SHRIRAMFIN.NS": "FINSERV",
    "M&MFIN.NS": "FINSERV",
    "MCX.NS": "FINSERV",
    "SBICARD.NS": "FINSERV",
    "JIOFIN.NS": "FINSERV",
}

FO_TICKERS: list[str] = sorted(TICKER_SECTOR.keys())


def sector_for(ticker: str) -> str | None:
    """Return sector code for a ticker, or None if not in F&O universe."""
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker = ticker + ".NS"
    return TICKER_SECTOR.get(ticker)


def tickers_in_sector(sector: str) -> list[str]:
    """Return all F&O tickers mapped to the given sector code."""
    return sorted([t for t, s in TICKER_SECTOR.items() if s == sector])
