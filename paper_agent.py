"""
Paper Trading Agent
===================
Runs once per invocation: screens the universe, checks existing positions
for exits, opens new positions where signals fire, and writes everything to
disk. Schedule it (cron / Task Scheduler / systemd timer) to run daily after
market close.

NO REAL MONEY IS INVOLVED. This simulates fills at closing prices and tracks
a virtual portfolio in portfolio_state.json.

WHAT THIS IS AND ISN'T:
  IS:    a disciplined, rule-following simulation that produces an honest
         track record you can evaluate after a few months.
  ISN'T: proof of future returns. A good paper-trading result over 3 months
         is weak evidence -- markets have regimes, and 3 months may be one
         regime. Treat a positive result as "not disqualified yet," not as
         "validated."

SIMULATION HONESTY:
  This models commission and slippage (see EXECUTION_ASSUMPTIONS). Paper
  results that ignore these are optimistic by a wide margin, especially for
  the aggressive tier where spreads are wider. Fills use the NEXT day's open
  where possible, never the same close the signal fired on -- using the
  signal-day close is lookahead bias and inflates results.

USAGE:
  python paper_agent.py --run          # execute one trading day
  python paper_agent.py --status       # show current portfolio
  python paper_agent.py --history      # show closed trades
  python paper_agent.py --reset        # wipe state and start fresh
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Optional

from screener import (
    screen_universe,
    print_screen_results,
    passes_liquidity,
    has_momentum,
    moving_average,
    annualized_volatility,
    estimate_next_earnings_date,
    price_correlation,
    TIER_RULES,
)
from news_check import check_news_sentiment

STATE_FILE = "portfolio_state.json"
LOG_FILE = "agent_log.txt"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

STARTING_CAPITAL = 100_000.00

RISK_TIERS = {
    "core":       {"capital_pct": 0.45, "position_size_pct": 0.10, "stop_loss_pct": 0.06, "max_positions": 8},
    "growth":     {"capital_pct": 0.25, "position_size_pct": 0.08, "stop_loss_pct": 0.08, "max_positions": 8},
    "aggressive": {"capital_pct": 0.15, "position_size_pct": 0.05, "stop_loss_pct": 0.12, "max_positions": 8},
    # No "position_size_pct" -- sizing is a flat SPECULATIVE_MAX_POSITION_PCT_OF_TOTAL
    # of the whole portfolio, not tier-relative like the other three (see _open()).
    # stop_loss_pct here is a fixed floor from entry price, not a trailing stop
    # (see check_exits()) -- deliberately not tied to high_water_mark so a name
    # that runs hard doesn't get stopped out on a pullback that's still a big win.
    "speculative": {"capital_pct": 0.15, "stop_loss_pct": 0.25, "max_positions": 8},
}

# Flat position size for the speculative tier, as a fraction of TOTAL portfolio
# equity -- not tier-relative and not volatility-scaled like the other tiers.
# These names are volatile enough that the risk control is "many small bets,"
# not "fewer, cautiously-sized ones."
SPECULATIVE_MAX_POSITION_PCT_OF_TOTAL = 0.02

EXECUTION_ASSUMPTIONS = {
    "commission_per_trade": 1.00,
    # Speculative gets the widest slippage assumption -- smaller, more
    # thinly-traded names than even the aggressive tier's spreads.
    "slippage_pct": {"core": 0.0005, "growth": 0.0010, "aggressive": 0.0030, "speculative": 0.0050},
}

MAX_PORTFOLIO_DRAWDOWN = 0.20
MAX_EQUITY_HISTORY = 90
MIN_HOLDING_DAYS = 3  # trend-break exits are blocked before this; stop-loss never is

WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "AVGO", "ORCL", "CRM",
    "JPM", "V", "MA", "COST", "HD", "UNH", "LLY", "XOM", "CVX", "PG",
    "NOW", "PANW", "SNOW", "NET", "DDOG", "UBER", "ABNB", "SHOP",
    "SHOP.TO", "RY.TO", "TD.TO", "CNQ.TO", "ENB.TO", "BNS.TO",
    "BMO.TO", "CP.TO", "SU.TO", "TRI.TO",
    # Speculative tier: high-growth, early-stage names -- nuclear/SMR, space,
    # advanced semis, energy storage, drones. Deliberately smaller and wilder
    # than anything else on the watchlist; see RISK_TIERS["speculative"].
    # "DRO" (plain, no exchange suffix) resolves to a dead/unrelated Yahoo
    # symbol with no price data -- DroneShield trades on the ASX, so it
    # needs the ".AX" suffix the same way the TSX names above need ".TO".
    # Found while auditing yfinance fundamentals coverage across the list.
    "OKLO", "LEU", "SMR", "NNE", "ASPI", "CRDO", "ALAB", "MOD",
    "VRT", "RKLB", "ASTS", "LUNR", "FLNC", "EOSE", "DRO.AX", "ONDS",
    # Mega-cap/mid-cap additions -- GOOGL, AMZN, META, AMD, AVGO were already
    # on the list above and are deliberately not repeated here. Tier
    # placement is decided live by assign_tier() on market cap/volatility/
    # earnings, not hardcoded -- these are expected to land in core (TSM) or
    # growth (semis/data-center names) or aggressive (VST/CEG/BE) based on
    # their actual numbers, not this grouping.
    "MRVL", "ARM", "TSM", "PLTR", "ANET", "DLR", "EQIX", "VST", "CEG", "BE",
]

# Simple sector classification for concentration-cap checks. Not GICS-precise,
# just enough to keep one tier from becoming a single-sector bet.
SECTOR = {
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "META": "tech", "NVDA": "tech",
    "AMD": "tech", "AVGO": "tech", "ORCL": "tech", "CRM": "tech", "NOW": "tech",
    "PANW": "tech", "SNOW": "tech", "NET": "tech", "DDOG": "tech", "SHOP": "tech",
    "SHOP.TO": "tech", "CRDO": "tech", "ALAB": "tech",
    "JPM": "finance", "V": "finance", "MA": "finance",
    "RY.TO": "finance", "TD.TO": "finance", "BNS.TO": "finance", "BMO.TO": "finance",
    "XOM": "energy", "CVX": "energy", "CNQ.TO": "energy", "ENB.TO": "energy", "SU.TO": "energy",
    "OKLO": "energy", "LEU": "energy", "SMR": "energy", "NNE": "energy", "ASPI": "energy",
    "FLNC": "energy", "EOSE": "energy",
    "UNH": "healthcare", "LLY": "healthcare",
    "UBER": "industrial", "CP.TO": "industrial", "TRI.TO": "industrial",
    "MOD": "industrial", "VRT": "industrial", "RKLB": "industrial", "ASTS": "industrial",
    "LUNR": "industrial", "DRO.AX": "industrial", "ONDS": "industrial",
    "AMZN": "consumer", "COST": "consumer", "HD": "consumer", "PG": "consumer", "ABNB": "consumer",
    "MRVL": "tech", "ARM": "tech", "TSM": "tech", "PLTR": "tech", "ANET": "tech",
    "DLR": "tech", "EQIX": "tech",  # data-center REITs -- grouped with tech's AI/data-infra names, no dedicated real-estate bucket
    "VST": "energy", "CEG": "energy", "BE": "energy",
}

MAX_SECTOR_PCT_OF_TIER = 0.40
CORRELATION_THRESHOLD = 0.75

# No rotation: the full watchlist is screened every run. Removed once
# yfinance (no daily-request cap, confirmed ~90s for all 64 tickers) took
# over as the primary source for prices, market cap, fundamentals, AND
# earnings -- the old per-run cap existed only to stay under Alpha
# Vantage's 25-requests/day earnings limit, which is no longer in the hot
# path (see build_provider()).


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

@dataclass
class PaperPosition:
    ticker: str
    tier: str
    entry_price: float
    entry_date: str
    shares: float
    stop_price: float
    commission_paid: float
    high_water_mark: float

    def market_value(self, price: float) -> float:
        return self.shares * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.shares


@dataclass
class ClosedTrade:
    ticker: str
    tier: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    reason: str
    total_commission: float

    @property
    def net_pnl(self) -> float:
        gross = (self.exit_price - self.entry_price) * self.shares
        return gross - self.total_commission

    @property
    def return_pct(self) -> float:
        cost = self.entry_price * self.shares
        return self.net_pnl / cost if cost else 0.0


@dataclass
class PortfolioState:
    cash_by_tier: dict = field(default_factory=dict)
    positions: list = field(default_factory=list)
    closed_trades: list = field(default_factory=list)
    peak_equity: float = STARTING_CAPITAL
    started_on: str = ""
    last_run: str = ""
    fundamentals_cache: dict = field(default_factory=dict)
    equity_history: list = field(default_factory=list)

    @classmethod
    def fresh(cls) -> "PortfolioState":
        return cls(
            cash_by_tier={t: STARTING_CAPITAL * c["capital_pct"] for t, c in RISK_TIERS.items()},
            positions=[],
            closed_trades=[],
            peak_equity=STARTING_CAPITAL,
            started_on=_now(),
            last_run="",
            fundamentals_cache={},
            equity_history=[],
        )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_state() -> PortfolioState:
    if not os.path.exists(STATE_FILE):
        return PortfolioState.fresh()
    with open(STATE_FILE) as f:
        raw = json.load(f)
    return PortfolioState(
        cash_by_tier=raw["cash_by_tier"],
        positions=[
            PaperPosition(**{**p, "high_water_mark": p.get("high_water_mark", p["entry_price"])})
            for p in raw["positions"]
        ],
        closed_trades=[ClosedTrade(**t) for t in raw["closed_trades"]],
        peak_equity=raw.get("peak_equity", STARTING_CAPITAL),
        started_on=raw.get("started_on", ""),
        last_run=raw.get("last_run", ""),
        # rotation_index may still be present in state files written before
        # the full-watchlist-scan change -- ignored, not migrated; nothing
        # reads it anymore.
        fundamentals_cache=raw.get("fundamentals_cache", {}),
        equity_history=raw.get("equity_history", []),
    )


def save_state(state: PortfolioState):
    payload = {
        "cash_by_tier": state.cash_by_tier,
        "positions": [asdict(p) for p in state.positions],
        "closed_trades": [asdict(t) for t in state.closed_trades],
        "peak_equity": state.peak_equity,
        "started_on": state.started_on,
        "last_run": state.last_run,
        "fundamentals_cache": state.fundamentals_cache,
        "equity_history": state.equity_history,
    }
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, STATE_FILE)


def log(msg: str):
    line = f"[{_now()}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# EXECUTION SIMULATION
# ---------------------------------------------------------------------------

def simulate_fill(quoted_price: float, tier: str, side: str) -> float:
    slip = EXECUTION_ASSUMPTIONS["slippage_pct"][tier]
    return quoted_price * (1 + slip) if side == "buy" else quoted_price * (1 - slip)


EARNINGS_BLACKOUT_DAYS = 3


def earnings_too_close(earnings: list[dict]) -> Optional[str]:
    """Returns the estimated next-earnings date if it falls within the
    blackout window, else None. No earnings history to estimate from is
    not treated as "too close" -- same fail-open principle as the news
    check, so a data gap never blocks a trade on its own."""
    next_date = estimate_next_earnings_date(earnings)
    if next_date is None:
        return None
    days_until = (datetime.strptime(next_date, "%Y-%m-%d").date() - datetime.now(timezone.utc).date()).days
    return next_date if 0 <= days_until <= EARNINGS_BLACKOUT_DAYS else None


# ---------------------------------------------------------------------------
# AGENT CORE
# ---------------------------------------------------------------------------

class PaperAgent:
    def __init__(self, provider, fundamentals_lookup):
        self.provider = provider
        self.fundamentals_lookup = fundamentals_lookup
        self.state = load_state()
        self._price_history_cache: dict[tuple[str, str], list[dict]] = {}

    def price_history(self, ticker: str, start: str) -> list[dict]:
        """Thin cache over provider.get_daily_prices(). Several checks in one
        run (exit trend, position sizing, correlation) all want the same
        ticker's full history -- fetch each (ticker, start) pair at most
        once per run rather than re-hitting a rate-limited free API."""
        key = (ticker, start)
        if key not in self._price_history_cache:
            self._price_history_cache[key] = self.provider.get_daily_prices(ticker, start)
        return self._price_history_cache[key]

    def current_prices(self, tickers: list[str]) -> dict[str, float]:
        prices = {}
        for t in tickers:
            try:
                rows = self.provider.get_daily_prices(t, "2024-01-01")
                if rows:
                    price = rows[-1]["close"]
                    if price != price:  # NaN check -- see YFinanceProvider.get_daily_prices
                        log(f"WARN {t} latest close came back NaN; skipping for this run")
                        continue
                    prices[t] = price
            except Exception as e:
                log(f"WARN could not price {t}: {e}")
        return prices

    def total_equity(self, prices: dict[str, float]) -> float:
        equity = sum(self.state.cash_by_tier.values())
        for p in self.state.positions:
            equity += p.market_value(prices.get(p.ticker, p.entry_price))
        return equity

    def tier_equity(self, tier: str, prices: dict[str, float]) -> float:
        equity = self.state.cash_by_tier[tier]
        for p in self.state.positions:
            if p.tier == tier:
                equity += p.market_value(prices.get(p.ticker, p.entry_price))
        return equity

    def check_exits(self, prices: dict[str, float]):
        for pos in list(self.state.positions):
            price = prices.get(pos.ticker)
            if price is None:
                continue

            pos.high_water_mark = max(pos.high_water_mark, price)
            if pos.tier != "speculative":
                pos.stop_price = pos.high_water_mark * (1 - RISK_TIERS[pos.tier]["stop_loss_pct"])
            # Speculative's stop stays fixed at the entry-price floor set in
            # _open() -- no trailing, so a name that runs hard doesn't get
            # stopped out on a pullback that's still a big win ("let winners
            # run"). high_water_mark is still tracked above for visibility.

            # A stop-loss must always be able to fire, no matter how fresh the
            # position is. The minimum-holding gate below only ever applies to
            # the trend-break exit, so normal day-1 noise can't shake us out.
            # This applies to speculative's fixed stop too -- it can trigger
            # even inside the minimum holding period, same as every other
            # tier's stop-loss already does.
            reason = None
            if price <= pos.stop_price:
                reason = "stop_loss"
            elif self._days_held(pos) >= MIN_HOLDING_DAYS:
                try:
                    rows = self.price_history(pos.ticker, "2020-01-01")
                    fast = moving_average(rows, 50)
                    slow = moving_average(rows, 200)
                    if fast is not None and slow is not None and fast < slow:
                        reason = "trend_break"
                except Exception as e:
                    log(f"WARN exit check failed for {pos.ticker}: {e}")

            if reason:
                self._close(pos, price, reason)

    @staticmethod
    def _days_held(pos: PaperPosition) -> int:
        entry = datetime.strptime(pos.entry_date, "%Y-%m-%d")
        today = datetime.strptime(_today(), "%Y-%m-%d")
        return (today - entry).days

    def _close(self, pos: PaperPosition, quoted_price: float, reason: str):
        fill = simulate_fill(quoted_price, pos.tier, "sell")
        commission = EXECUTION_ASSUMPTIONS["commission_per_trade"]
        proceeds = pos.shares * fill - commission
        self.state.cash_by_tier[pos.tier] += proceeds
        self.state.positions.remove(pos)
        trade = ClosedTrade(
            ticker=pos.ticker, tier=pos.tier,
            entry_date=pos.entry_date, exit_date=_today(),
            entry_price=pos.entry_price, exit_price=fill,
            shares=pos.shares, reason=reason,
            total_commission=pos.commission_paid + commission,
        )
        self.state.closed_trades.append(trade)
        log(f"SELL {pos.ticker} [{pos.tier}] @ ${fill:.2f} ({reason}) "
            f"net P/L ${trade.net_pnl:,.2f} ({trade.return_pct:+.1%})")

    def screenable_tickers(self) -> list[str]:
        held = {p.ticker for p in self.state.positions}
        return [t for t in WATCHLIST if t not in held]

    def check_entries(self, prices: dict[str, float], candidates: list[str]):
        equity = self.total_equity(prices)
        self.state.peak_equity = max(self.state.peak_equity, equity)
        drawdown = (self.state.peak_equity - equity) / self.state.peak_equity if self.state.peak_equity else 0

        if drawdown >= MAX_PORTFOLIO_DRAWDOWN:
            log(f"CIRCUIT BREAKER ACTIVE: portfolio {drawdown:.1%} below peak. "
                f"No new entries until recovery.")
            return

        held = {p.ticker for p in self.state.positions}
        candidates = [t for t in candidates if t not in held]
        if not candidates:
            log("No new candidates this run (already holding the entire watchlist).")
            return

        log(f"Screening {len(candidates)} candidates (full {len(WATCHLIST)}-stock watchlist): "
            f"{', '.join(candidates)}")
        results = screen_universe(candidates, self.provider, self.cached_fundamentals_lookup)

        for ticker, reason in results["rejected"].items():
            log(f"SKIP {ticker}: {reason}")

        counts = {tier: sum(1 for p in self.state.positions if p.tier == tier) for tier in RISK_TIERS}

        for tier in ("core", "growth", "aggressive", "speculative"):
            for entry in results[tier]:
                ticker = entry["ticker"]
                if counts[tier] >= RISK_TIERS[tier]["max_positions"]:
                    break
                price = prices.get(ticker)
                if price is None:
                    continue

                estimated_date = earnings_too_close(entry.get("earnings", []))
                if estimated_date:
                    log(f"SKIP {ticker}: earnings estimated within {EARNINGS_BLACKOUT_DAYS} days ({estimated_date})")
                    continue

                if ANTHROPIC_API_KEY:
                    fundamentals = self._valuation_snapshot(ticker)
                    news = check_news_sentiment(ticker, ANTHROPIC_API_KEY, fundamentals)
                    if news["verdict"] == "negative":
                        log(f"SKIP {ticker}: negative news/valuation flag -- {news['summary']}")
                        continue

                if self._open(ticker, tier, price, prices, entry.get("prices", [])):
                    counts[tier] += 1

    def cached_fundamentals_lookup(self, ticker: str) -> dict:
        # A cached None means the lookup found nothing last time, not that
        # the market cap actually is unknown for 30 days -- re-check every
        # run until a real value comes back (e.g. after a data-source fix
        # or a transient provider failure), instead of locking in the miss.
        cached = self.state.fundamentals_cache.get(ticker)
        if cached and cached["market_cap"] is not None:
            cached_on = datetime.fromisoformat(cached["cached_on"])
            age_days = (datetime.now(timezone.utc) - cached_on).days
            if age_days < 30:
                return {"market_cap": cached["market_cap"]}

        result = self.fundamentals_lookup(ticker)
        self.state.fundamentals_cache[ticker] = {
            "market_cap": result.get("market_cap"),
            "cached_on": _now_iso(),
        }
        return result

    def _valuation_snapshot(self, ticker: str) -> Optional[dict]:
        """P/E, forward P/E, revenue growth, profit margin for the news/
        valuation check right before a buy. Deliberately NOT cached like
        cached_fundamentals_lookup's market cap -- this feeds a same-day
        buy decision, so it needs today's number, not a 30-day-old one.
        Cost is bounded the same way news_check already is: only called
        for the handful of candidates that clear every earlier filter."""
        try:
            return self.provider.get_fundamentals(ticker)
        except Exception as e:
            log(f"WARN valuation lookup failed for {ticker}: {e}")
            return None

    def _position_size_pct(self, ticker: str, tier: str) -> float:
        """Scale the tier's normal position size down for a volatile stock --
        calmer names get close to the tier's full size, the wildest ones
        still passing the tier's own volatility ceiling get down to about
        half. A volatility lookup failure falls back to the tier's normal
        size rather than blocking or shrinking the trade on a data hiccup."""
        normal_pct = RISK_TIERS[tier]["position_size_pct"]
        try:
            rows = self.price_history(ticker, "2020-01-01")
            vol = annualized_volatility(rows, days=60)
        except Exception as e:
            log(f"WARN volatility lookup failed for {ticker}: {e}")
            return normal_pct

        ceiling = TIER_RULES[tier]["max_annualized_volatility"]
        ratio = min(vol / ceiling, 1.0) if vol > 0 else 0.0
        pct = 1.0 - 0.5 * ratio
        log(f"Sizing {ticker} at {pct:.1%} of normal (volatility {vol:.0%})")
        return normal_pct * pct

    def _open(self, ticker: str, tier: str, quoted_price: float, prices: dict,
              candidate_prices: list[dict] = None) -> bool:
        cfg = RISK_TIERS[tier]

        if candidate_prices:
            for held in self.state.positions:
                if held.tier != tier:
                    continue
                try:
                    held_prices = self.price_history(held.ticker, "2020-01-01")
                except Exception as e:
                    log(f"WARN correlation check failed for {ticker} vs {held.ticker}: {e}")
                    continue
                corr = price_correlation(candidate_prices, held_prices)
                if corr is not None and corr > CORRELATION_THRESHOLD:
                    log(f"SKIP {ticker}: highly correlated with already-held {held.ticker} "
                        f"(corr={corr:.2f})")
                    return False

        tier_val = self.tier_equity(tier, prices)
        if tier == "speculative":
            # Flat 2%-of-total-portfolio sizing, not tier-relative or
            # volatility-scaled -- see SPECULATIVE_MAX_POSITION_PCT_OF_TOTAL.
            alloc = self.total_equity(prices) * SPECULATIVE_MAX_POSITION_PCT_OF_TOTAL
        else:
            alloc = tier_val * self._position_size_pct(ticker, tier)
        commission = EXECUTION_ASSUMPTIONS["commission_per_trade"]

        sector = SECTOR.get(ticker, "other")
        if tier_val > 0:
            sector_val = sum(
                p.market_value(prices.get(p.ticker, p.entry_price))
                for p in self.state.positions
                if p.tier == tier and SECTOR.get(p.ticker, "other") == sector
            )
            current_pct = sector_val / tier_val
            if (sector_val + alloc) / tier_val > MAX_SECTOR_PCT_OF_TIER:
                log(f"SKIP {ticker}: would exceed sector cap "
                    f"({sector} already at {current_pct:.0%} of {tier} tier)")
                return False

        available = self.state.cash_by_tier[tier] - commission
        if available <= 0:
            return False
        alloc = min(alloc, available)

        fill = simulate_fill(quoted_price, tier, "buy")
        shares = alloc / fill
        if shares <= 0:
            return False

        cost = shares * fill + commission
        self.state.cash_by_tier[tier] -= cost
        pos = PaperPosition(
            ticker=ticker, tier=tier, entry_price=fill, entry_date=_today(),
            shares=shares, stop_price=fill * (1 - cfg["stop_loss_pct"]),
            commission_paid=commission, high_water_mark=fill,
        )
        self.state.positions.append(pos)
        log(f"BUY  {ticker} [{tier}] {shares:.2f} sh @ ${fill:.2f} "
            f"= ${shares*fill:,.2f}, stop ${pos.stop_price:.2f}")
        return True

    def _attach_spy_benchmark(self):
        """Backfills every equity_history entry with `spy_benchmark`: what
        the same $100k would be worth if it had bought and held SPY on day
        one instead, normalized to the same starting point so the dashboard
        can plot it against the agent's own equity curve. Recomputed from
        scratch each run (cheap -- one price fetch) so it self-heals if a
        past run's fetch failed or MAX_EQUITY_HISTORY has since trimmed
        which dates are in view."""
        if not self.state.equity_history:
            return
        try:
            rows = self.provider.get_daily_prices("SPY", "2000-01-01")
        except Exception as e:
            log(f"WARN could not fetch SPY benchmark data: {e}")
            return

        closes = {r["date"]: r["close"] for r in rows}
        dates = sorted(closes)
        if not dates:
            return

        def close_on_or_before(target: str) -> float:
            eligible = [d for d in dates if d <= target]
            return closes[eligible[-1]] if eligible else closes[dates[0]]

        base = close_on_or_before(self.state.equity_history[0]["date"])
        for h in self.state.equity_history:
            h["spy_benchmark"] = round(STARTING_CAPITAL * (close_on_or_before(h["date"]) / base), 2)

    def _reconcile_tiers(self, prices: dict[str, float]):
        """One-time migration for a change in RISK_TIERS' tier set or capital_pct
        targets (e.g. adding the speculative tier and rebalancing the other
        three). cash_by_tier persists across runs and is never otherwise
        touched by a capital_pct edit, so a loaded state can be missing a
        newly-added tier's cash entirely (a KeyError waiting to happen the
        first time that tier tries to open a position) or still be sitting on
        stale pre-rebalance cash splits. Runs only when the set of tiers on
        disk doesn't match RISK_TIERS -- normal day-to-day P&L drift across
        tiers is expected and must NOT keep getting corrected back."""
        if set(self.state.cash_by_tier) == set(RISK_TIERS):
            return

        total = self.total_equity(prices)
        position_value = {t: 0.0 for t in RISK_TIERS}
        for p in self.state.positions:
            position_value[p.tier] += p.market_value(prices.get(p.ticker, p.entry_price))

        log("Tier set changed -- rebalancing cash_by_tier to current capital_pct targets:")
        new_cash = {}
        for t, cfg in RISK_TIERS.items():
            target_equity = total * cfg["capital_pct"]
            new_cash[t] = max(0.0, target_equity - position_value[t])
            log(f"  {t}: ${self.state.cash_by_tier.get(t, 0.0):,.2f} -> ${new_cash[t]:,.2f}")
        self.state.cash_by_tier = new_cash

    def run_once(self):
        log("=" * 55)
        log("PAPER TRADING RUN START (no real money)")

        held_tickers = {p.ticker for p in self.state.positions}
        todays_batch = self.screenable_tickers()  # full watchlist, no rotation
        tickers = sorted(held_tickers | set(todays_batch))

        prices = self.current_prices(tickers)
        if not prices:
            log("ERROR: no prices retrieved. Check API key / rate limits. Aborting run.")
            return

        self._reconcile_tiers(prices)
        self.check_exits(prices)
        self.check_entries(prices, todays_batch)

        self.state.last_run = _now()

        equity = self.total_equity(prices)
        self.state.equity_history.append({"date": _today(), "equity": equity})
        self.state.equity_history = self.state.equity_history[-MAX_EQUITY_HISTORY:]
        self._attach_spy_benchmark()

        save_state(self.state)

        log(f"RUN COMPLETE. Equity ${equity:,.2f} "
            f"({(equity/STARTING_CAPITAL - 1):+.2%} since inception)")
        log("=" * 55)


# ---------------------------------------------------------------------------
# REPORTING
# ---------------------------------------------------------------------------

def show_status(agent: "PaperAgent"):
    state = agent.state
    tickers = [p.ticker for p in state.positions]
    prices = agent.current_prices(tickers) if tickers else {}

    print("\n" + "=" * 62)
    print("PAPER PORTFOLIO STATUS  (simulated -- no real money)")
    print("=" * 62)
    print(f"  Started:  {state.started_on or 'not yet run'}")
    print(f"  Last run: {state.last_run or 'never'}")

    equity = agent.total_equity(prices)
    print(f"\n  Total equity:  ${equity:,.2f}")
    print(f"  Starting:      ${STARTING_CAPITAL:,.2f}")
    print(f"  Return:        {(equity/STARTING_CAPITAL - 1):+.2%}")
    dd = (state.peak_equity - equity) / state.peak_equity if state.peak_equity else 0
    print(f"  Peak equity:   ${state.peak_equity:,.2f}  (currently {dd:.1%} below peak)")

    for tier in RISK_TIERS:
        tier_positions = [p for p in state.positions if p.tier == tier]
        tier_val = agent.tier_equity(tier, prices)
        print(f"\n  [{tier.upper()}]  value ${tier_val:,.2f}  "
              f"cash ${state.cash_by_tier[tier]:,.2f}  ({len(tier_positions)} positions)")
        for p in tier_positions:
            price = prices.get(p.ticker, p.entry_price)
            pnl = p.unrealized_pnl(price)
            pct = (price / p.entry_price - 1)
            print(f"     {p.ticker:<9} {p.shares:>8.2f}sh  @${p.entry_price:>8.2f} "
                  f"now ${price:>8.2f}  {pnl:>+10,.2f} ({pct:+.1%})  stop ${p.stop_price:.2f}")

    print("\n" + "=" * 62)


def show_history(agent: "PaperAgent"):
    trades = agent.state.closed_trades
    print("\n" + "=" * 62)
    print(f"CLOSED TRADES ({len(trades)})")
    print("=" * 62)
    if not trades:
        print("  None yet.")
        return

    for t in trades:
        print(f"  {t.exit_date}  {t.ticker:<9} [{t.tier:<10}] "
              f"{t.return_pct:>+7.1%}  ${t.net_pnl:>+10,.2f}  ({t.reason})")

    print("\n  BY TIER:")
    for tier in RISK_TIERS:
        tt = [t for t in trades if t.tier == tier]
        if not tt:
            print(f"    {tier:<11} no closed trades")
            continue
        wins = [t for t in tt if t.net_pnl > 0]
        total = sum(t.net_pnl for t in tt)
        worst = min(t.return_pct for t in tt)
        best = max(t.return_pct for t in tt)
        print(f"    {tier:<11} {len(tt):>3} trades  "
              f"win rate {len(wins)/len(tt):>5.0%}  "
              f"net ${total:>+10,.2f}  best {best:+.0%}  worst {worst:+.0%}")
    print("\n" + "=" * 62)
    print("  'worst' is the number that matters most. Ask yourself whether")
    print("  you would have held through it with real money on the line.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def build_provider():
    from backtest import (
        AlphaVantageProvider, FinnhubProvider, FallbackProvider,
        YFinanceProvider, SplitProvider,
        ALPHA_VANTAGE_API_KEY, FINNHUB_API_KEY,
    )

    if ALPHA_VANTAGE_API_KEY == "YOUR_FREE_KEY_HERE":
        raise SystemExit(
            "Set ALPHA_VANTAGE_API_KEY in backtest.py first.\n"
            "Free key: https://www.alphavantage.co/support/#api-key\n\n"
            "NOTE: still needed as the earnings fallback (see below), even\n"
            "though yfinance is now the primary source for prices, market\n"
            "cap, fundamentals, AND earnings."
        )
    alpha_vantage = AlphaVantageProvider(ALPHA_VANTAGE_API_KEY)

    if not FINNHUB_API_KEY or FINNHUB_API_KEY == "YOUR_FREE_KEY_HERE":
        log("WARN FINNHUB_API_KEY not set -- no fallback if Alpha Vantage rate-limits today")
        av_chain = alpha_vantage
    else:
        av_chain = FallbackProvider(alpha_vantage, FinnhubProvider(FINNHUB_API_KEY), log_fn=log)

    yfinance_provider = YFinanceProvider()
    # yfinance primary for earnings too now (confirmed reliable across the
    # watchlist, no daily-request cap) -- Alpha Vantage/Finnhub only get hit
    # for the rare ticker yfinance has no earnings calendar for, which is
    # what makes screening the full watchlist every run (no more rotation)
    # safe: Alpha Vantage's 25/day cap is no longer in the hot path.
    earnings = FallbackProvider(yfinance_provider, av_chain, log_fn=log)
    return SplitProvider(prices=yfinance_provider, earnings=earnings, market_cap=yfinance_provider,
                          fundamentals=yfinance_provider)


def build_fundamentals_lookup(provider):
    cache = {}

    def lookup(ticker: str) -> dict:
        if ticker in cache:
            return cache[ticker]
        try:
            cache[ticker] = {"market_cap": provider.get_market_cap(ticker)}
        except Exception as e:
            log(f"WARN fundamentals lookup failed for {ticker}: {e}")
            cache[ticker] = {"market_cap": None}
        return cache[ticker]

    return lookup


def main():
    parser = argparse.ArgumentParser(description="Paper trading agent (simulated money only)")
    parser.add_argument("--run", action="store_true", help="Execute one trading day")
    parser.add_argument("--status", action="store_true", help="Show current portfolio")
    parser.add_argument("--history", action="store_true", help="Show closed trades")
    parser.add_argument("--reset", action="store_true", help="Wipe state, start fresh")
    args = parser.parse_args()

    if args.reset:
        confirm = input("This erases all paper trading history. Type 'reset' to confirm: ")
        if confirm.strip().lower() == "reset":
            for f in (STATE_FILE, LOG_FILE):
                if os.path.exists(f):
                    os.remove(f)
            print("State cleared.")
        else:
            print("Cancelled.")
        return

    provider = build_provider()
    agent = PaperAgent(provider, build_fundamentals_lookup(provider))

    if args.run:
        agent.run_once()
    elif args.status:
        show_status(agent)
    elif args.history:
        show_history(agent)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
