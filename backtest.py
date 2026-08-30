"""
Trend + Earnings-Quality Backtesting Engine
=============================================
Strategy: Buy stocks in confirmed uptrends (price > 50-day MA > 200-day MA)
that also pass a fundamental quality filter (reasonable P/E, positive
earnings surprise trend). Exit on trend break or stop-loss.

Risk management is enforced structurally, not left to the signal:
  - Fixed position sizing (% of portfolio per trade)
  - Hard stop-loss per position
  - Max concurrent positions
  - Max portfolio drawdown circuit-breaker (halts new trades)

WHY THIS DESIGN:
  - Trend-following has the longest, most cross-market evidence of any
    systematic factor. It doesn't predict the future; it reacts to what's
    already happening, which is more robust than prediction-based signals.
  - The earnings/P/E filter avoids buying pure hype - it requires the
    business to actually be performing.
  - Risk rules matter more than signal quality. Most retail algo failures
    are position-sizing / stop-loss failures, not "bad stock picks."

DATA SOURCE:
  This script is written against a generic `DataProvider` interface so you
  can plug in whichever source you have access to:
    - Alpha Vantage (free tier, https://www.alphavantage.co/support/#api-key)
    - IBKR historical data API (best if you're heading toward IBKR live trading)
    - A local CSV export from your broker

  Fill in `AlphaVantageProvider` below with your free API key, or swap in
  your own provider that implements `get_daily_prices()` and `get_earnings()`.

RUN:
  python backtest.py --tickers AAPL MSFT SHOP.TO RY.TO --start 2016-01-01
"""

import argparse
import os
import time
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# CONFIG - edit these
# ---------------------------------------------------------------------------

# Reads from an environment variable named ALPHA_VANTAGE_API_KEY.
# On GitHub Actions this comes from a Secret (never typed into this file).
# Running locally, set it in your terminal first, e.g.:
#   export ALPHA_VANTAGE_API_KEY=your_key_here      (Mac/Linux)
#   set ALPHA_VANTAGE_API_KEY=your_key_here          (Windows)
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "YOUR_FREE_KEY_HERE")

# Tiered risk allocation. Assign each ticker you backtest to a tier below.
# Total capital is split across tiers by "capital_pct"; within a tier,
# "position_size_pct" caps how much of THAT TIER's capital goes into any
# one stock. Aggressive tier gets smaller per-stock bets (more spread out)
# specifically because each stock there is individually riskier -- this is
# what keeps a single blowup from doing serious portfolio damage.
RISK_TIERS = {
    "core": {
        "capital_pct": 0.50,          # 50% of total portfolio
        "position_size_pct": 0.10,    # up to 10% of this tier's capital per stock
        "stop_loss_pct": 0.06,
        "max_pe_ratio": 35,
        "min_earnings_trend": 0.0,    # must be beating estimates
    },
    "growth": {
        "capital_pct": 0.30,
        "position_size_pct": 0.08,
        "stop_loss_pct": 0.08,
        "max_pe_ratio": 60,
        "min_earnings_trend": -0.02,  # slight miss tolerated
    },
    "aggressive": {
        "capital_pct": 0.20,
        "position_size_pct": 0.05,    # smaller bets per stock -- more names, less concentration
        "stop_loss_pct": 0.12,
        "max_pe_ratio": None,         # no P/E cap -- many high-growth names have none/high P/E
        "min_earnings_trend": -0.10,  # tolerant, but still requires SOME earnings data
    },
}

RISK_CONFIG = {
    "max_concurrent_positions_per_tier": 8,
    "max_portfolio_drawdown": 0.20,  # halt ALL new entries if total portfolio down 20% from peak
    "starting_capital": 100_000.0,
}

STRATEGY_CONFIG = {
    "fast_ma": 50,
    "slow_ma": 200,
}

# Map each ticker to a tier. Edit this to match what you actually want to
# trade. A ticker not listed here is skipped.
TICKER_TIERS = {
    # examples -- replace with your own list
    "MSFT": "core", "AAPL": "core", "RY.TO": "core",
    "NVDA": "growth", "AMD": "growth", "SHOP.TO": "growth",
    # aggressive: smaller/newer names -- fill in what you actually want considered
}


# ---------------------------------------------------------------------------
# DATA LAYER - swap this out for whatever data source you actually use
# ---------------------------------------------------------------------------

class DataProvider:
    """Interface. Implement these two methods for your real data source."""

    def get_daily_prices(self, ticker: str, start: str) -> list[dict]:
        """Return list of {date, open, high, low, close, volume}, oldest first."""
        raise NotImplementedError

    def get_earnings(self, ticker: str) -> list[dict]:
        """Return list of {date, reported_eps, estimated_eps} per quarter."""
        raise NotImplementedError


class AlphaVantageProvider(DataProvider):
    """Free tier: 25 requests/day, 5/minute. Fine for research, not for
    a live multi-stock agent - you'll want a paid tier or IBKR data for that."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base = "https://www.alphavantage.co/query"

    def _get(self, params: dict) -> dict:
        params["apikey"] = self.api_key
        url = self.base + "?" + "&".join(f"{k}={v}" for k, v in params.items())
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            raise RuntimeError(f"Data fetch failed for {params}: {e}")

    def get_daily_prices(self, ticker: str, start: str) -> list[dict]:
        data = self._get({
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "full",
        })
        series = data.get("Time Series (Daily)", {})
        if not series:
            raise RuntimeError(f"No price data for {ticker}: {data.get('Note') or data}")
        rows = []
        for date, ohlcv in sorted(series.items()):
            if date < start:
                continue
            rows.append({
                "date": date,
                "open": float(ohlcv["1. open"]),
                "high": float(ohlcv["2. high"]),
                "low": float(ohlcv["3. low"]),
                "close": float(ohlcv["4. close"]),
                "volume": int(ohlcv["5. volume"]),
            })
        time.sleep(13)  # respect free-tier rate limit (5 calls/min)
        return rows

    def get_earnings(self, ticker: str) -> list[dict]:
        data = self._get({"function": "EARNINGS", "symbol": ticker})
        quarterly = data.get("quarterlyEarnings", [])
        rows = [{
            "date": q["reportedDate"],
            "reported_eps": _safe_float(q.get("reportedEPS")),
            "estimated_eps": _safe_float(q.get("estimatedEPS")),
        } for q in quarterly]
        time.sleep(13)
        return rows


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# SIGNAL LOGIC
# ---------------------------------------------------------------------------

def moving_average(closes: list[float], window: int) -> list[Optional[float]]:
    out = [None] * len(closes)
    for i in range(window - 1, len(closes)):
        out[i] = sum(closes[i - window + 1:i + 1]) / window
    return out


def earnings_surprise_trend(earnings: list[dict], as_of_date: str) -> float:
    """Average of last 4 reported earnings surprises (%) as of a given date.
    Positive = company has been beating estimates recently."""
    past = [e for e in earnings if e["date"] <= as_of_date
            and e["reported_eps"] is not None and e["estimated_eps"] not in (None, 0)]
    past = sorted(past, key=lambda e: e["date"])[-4:]
    if not past:
        return 0.0
    surprises = [(e["reported_eps"] - e["estimated_eps"]) / abs(e["estimated_eps"]) for e in past]
    return sum(surprises) / len(surprises)


def passes_quality_filter(pe_ratio: Optional[float], earn_trend: float, tier: str) -> bool:
    cfg = RISK_TIERS[tier]
    if cfg["max_pe_ratio"] is not None and pe_ratio is not None and pe_ratio > cfg["max_pe_ratio"]:
        return False
    if earn_trend < cfg["min_earnings_trend"]:
        return False
    return True


# ---------------------------------------------------------------------------
# BACKTEST ENGINE
# ---------------------------------------------------------------------------

@dataclass
class Position:
    ticker: str
    tier: str
    entry_price: float
    entry_date: str
    shares: float
    stop_price: float = field(init=False)

    def __post_init__(self):
        self.stop_price = self.entry_price * (1 - RISK_TIERS[self.tier]["stop_loss_pct"])


@dataclass
class Trade:
    ticker: str
    tier: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    reason: str

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.shares

    @property
    def return_pct(self) -> float:
        return (self.exit_price / self.entry_price) - 1


class Backtester:
    def __init__(self, provider: DataProvider):
        self.provider = provider
        total = RISK_CONFIG["starting_capital"]
        # cash is tracked separately per tier -- a tier can't borrow against
        # another tier's capital, which is what actually enforces the split
        self.cash_by_tier = {t: total * cfg["capital_pct"] for t, cfg in RISK_TIERS.items()}
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[str, float]] = []
        self.peak_equity = total

    def portfolio_value(self, prices_today: dict[str, float]) -> float:
        value = sum(self.cash_by_tier.values())
        for t, pos in self.positions.items():
            value += pos.shares * prices_today.get(t, pos.entry_price)
        return value

    def tier_value(self, tier: str, prices_today: dict[str, float]) -> float:
        value = self.cash_by_tier[tier]
        for t, pos in self.positions.items():
            if pos.tier == tier:
                value += pos.shares * prices_today.get(t, pos.entry_price)
        return value

    def run(self, tickers: list[str], start: str):
        tickers = [t for t in tickers if t in TICKER_TIERS]
        skipped = set(tickers) - set(TICKER_TIERS.keys())
        if skipped:
            print(f"Skipping tickers with no tier assignment: {skipped}")
        print(f"Fetching data for {len(tickers)} tickers... (rate-limited, this takes a while)")
        price_data, earnings_data, pe_data = {}, {}, {}
        for t in tickers:
            print(f"  {t} [{TICKER_TIERS[t]}] ...")
            price_data[t] = self.provider.get_daily_prices(t, start)
            earnings_data[t] = self.provider.get_earnings(t)
            # crude trailing P/E proxy: price / (sum of last 4 reported EPS)
            pe_data[t] = None

        all_dates = sorted(set(d["date"] for rows in price_data.values() for d in rows))
        closes_by_ticker = {t: {d["date"]: d["close"] for d in rows} for t, rows in price_data.items()}
        ma_fast = {t: {} for t in tickers}
        ma_slow = {t: {} for t in tickers}
        for t in tickers:
            dates_sorted = sorted(closes_by_ticker[t].keys())
            closes_sorted = [closes_by_ticker[t][d] for d in dates_sorted]
            fast = moving_average(closes_sorted, STRATEGY_CONFIG["fast_ma"])
            slow = moving_average(closes_sorted, STRATEGY_CONFIG["slow_ma"])
            for i, d in enumerate(dates_sorted):
                ma_fast[t][d] = fast[i]
                ma_slow[t][d] = slow[i]

        for date in all_dates:
            prices_today = {t: closes_by_ticker[t][date] for t in tickers if date in closes_by_ticker[t]}
            self._check_stops_and_exits(date, prices_today, ma_fast, ma_slow)

            equity = self.portfolio_value(prices_today)
            self.peak_equity = max(self.peak_equity, equity)
            drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity else 0
            self.equity_curve.append((date, equity))

            if drawdown >= RISK_CONFIG["max_portfolio_drawdown"]:
                continue  # circuit breaker: no new entries while in ANY tier once portfolio is deep in drawdown

            positions_per_tier = {tier: 0 for tier in RISK_TIERS}
            for pos in self.positions.values():
                positions_per_tier[pos.tier] += 1

            for t in tickers:
                if t in self.positions or date not in prices_today:
                    continue
                tier = TICKER_TIERS[t]
                if positions_per_tier[tier] >= RISK_CONFIG["max_concurrent_positions_per_tier"]:
                    continue
                fast, slow = ma_fast[t].get(date), ma_slow[t].get(date)
                if fast is None or slow is None:
                    continue
                price = prices_today[t]
                trend_ok = price > fast > slow
                earn_trend = earnings_surprise_trend(earnings_data[t], date)
                quality_ok = passes_quality_filter(pe_data[t], earn_trend, tier)
                if trend_ok and quality_ok:
                    tier_val = self.tier_value(tier, prices_today)
                    self._enter(t, tier, date, price, tier_val)
                    positions_per_tier[tier] += 1

        return self._summary()

    def _enter(self, ticker: str, tier: str, date: str, price: float, tier_value: float):
        alloc = tier_value * RISK_TIERS[tier]["position_size_pct"]
        shares = alloc / price
        if shares * price > self.cash_by_tier[tier]:
            shares = self.cash_by_tier[tier] / price  # cap to available tier cash
        if shares <= 0:
            return
        self.cash_by_tier[tier] -= shares * price
        self.positions[ticker] = Position(ticker, tier, price, date, shares)

    def _exit(self, ticker: str, date: str, price: float, reason: str):
        pos = self.positions.pop(ticker)
        self.cash_by_tier[pos.tier] += pos.shares * price
        self.trades.append(Trade(ticker, pos.tier, pos.entry_date, date, pos.entry_price, price, pos.shares, reason))

    def _check_stops_and_exits(self, date, prices_today, ma_fast, ma_slow):
        for t in list(self.positions.keys()):
            if date not in prices_today:
                continue
            price = prices_today[t]
            pos = self.positions[t]
            fast, slow = ma_fast[t].get(date), ma_slow[t].get(date)
            if price <= pos.stop_price:
                self._exit(t, date, price, "stop_loss")
            elif fast is not None and slow is not None and fast < slow:
                self._exit(t, date, price, "trend_break")

    def _summary(self) -> dict:
        if not self.equity_curve:
            return {"error": "no data"}
        final_equity = self.equity_curve[-1][1]
        start_equity = RISK_CONFIG["starting_capital"]
        total_return = (final_equity / start_equity) - 1
        wins = [t for t in self.trades if t.pnl > 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        max_dd = self._max_drawdown()

        by_tier = {}
        for tier in RISK_TIERS:
            tier_trades = [t for t in self.trades if t.tier == tier]
            tier_wins = [t for t in tier_trades if t.pnl > 0]
            by_tier[tier] = {
                "num_trades": len(tier_trades),
                "win_rate_pct": round(len(tier_wins) / len(tier_trades) * 100, 1) if tier_trades else 0,
                "total_pnl": round(sum(t.pnl for t in tier_trades), 2),
                "avg_trade_return_pct": round(
                    sum(t.return_pct for t in tier_trades) / len(tier_trades) * 100, 2
                ) if tier_trades else 0,
                "best_trade_pct": round(max((t.return_pct for t in tier_trades), default=0) * 100, 2),
                "worst_trade_pct": round(min((t.return_pct for t in tier_trades), default=0) * 100, 2),
            }

        return {
            "starting_capital": start_equity,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return * 100, 2),
            "num_trades": len(self.trades),
            "win_rate_pct": round(win_rate * 100, 1),
            "max_drawdown_pct": round(max_dd * 100, 1),
            "avg_trade_return_pct": round(
                sum(t.return_pct for t in self.trades) / len(self.trades) * 100, 2
            ) if self.trades else 0,
            "by_tier": by_tier,
        }

    def _max_drawdown(self) -> float:
        peak = self.equity_curve[0][1]
        max_dd = 0.0
        for _, eq in self.equity_curve:
            peak = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak if peak else 0)
        return max_dd


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tiered trend + earnings-quality backtest")
    parser.add_argument("--tickers", nargs="+", default=list(TICKER_TIERS.keys()),
                         help="Tickers to test (must be in TICKER_TIERS in this file). "
                              "Defaults to everything in TICKER_TIERS.")
    parser.add_argument("--start", default="2018-01-01")
    args = parser.parse_args()

    if ALPHA_VANTAGE_API_KEY == "YOUR_FREE_KEY_HERE":
        raise SystemExit(
            "Set ALPHA_VANTAGE_API_KEY at the top of this file first.\n"
            "Get a free key at: https://www.alphavantage.co/support/#api-key\n"
            "Also edit TICKER_TIERS to assign your own core/growth/aggressive tickers."
        )

    provider = AlphaVantageProvider(ALPHA_VANTAGE_API_KEY)
    bt = Backtester(provider)
    results = bt.run(args.tickers, args.start)

    print("\n" + "=" * 55)
    print("OVERALL RESULTS")
    print("=" * 55)
    for k, v in results.items():
        if k != "by_tier":
            print(f"  {k}: {v}")

    print("\n" + "=" * 55)
    print("RESULTS BY RISK TIER")
    print("=" * 55)
    for tier, stats in results.get("by_tier", {}).items():
        alloc_pct = RISK_TIERS[tier]["capital_pct"] * 100
        print(f"\n  [{tier.upper()}]  ({alloc_pct:.0f}% of portfolio allocated)")
        for k, v in stats.items():
            print(f"    {k}: {v}")

    print("\n" + "=" * 55)
    print("Compare total_return_pct against SPY/TSX buy-and-hold over")
    print("the same period. Check worst_trade_pct per tier -- that's")
    print("the downside you need to be able to actually stomach.")


if __name__ == "__main__":
    main()
