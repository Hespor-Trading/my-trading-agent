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
)

STATE_FILE = "portfolio_state.json"
LOG_FILE = "agent_log.txt"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

STARTING_CAPITAL = 100_000.00

RISK_TIERS = {
    "core":       {"capital_pct": 0.50, "position_size_pct": 0.10, "stop_loss_pct": 0.06, "max_positions": 8},
    "growth":     {"capital_pct": 0.30, "position_size_pct": 0.08, "stop_loss_pct": 0.08, "max_positions": 8},
    "aggressive": {"capital_pct": 0.20, "position_size_pct": 0.05, "stop_loss_pct": 0.12, "max_positions": 8},
}

EXECUTION_ASSUMPTIONS = {
    "commission_per_trade": 1.00,
    "slippage_pct": {"core": 0.0005, "growth": 0.0010, "aggressive": 0.0030},
}

MAX_PORTFOLIO_DRAWDOWN = 0.20

# WATCHLIST: everything the agent has "on its radar," across sectors and
# both US and Canadian markets. The free API can only afford to check
# ~7-8 NEW stocks per day, so the agent doesn't check this whole list every
# day -- it ROTATES through it, checking a different slice each run (see
# ROTATION_BATCH_SIZE below). Over roughly a week, every stock here gets
# checked. Stocks you already hold are always re-checked daily regardless
# (needed for stop-loss/exit logic) -- rotation only applies to picking
# NEW candidates.
WATCHLIST = [
    # US tech / large-cap
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "AVGO", "ORCL", "CRM",
    # US finance / industrial / consumer
    "JPM", "V", "MA", "COST", "HD", "UNH", "LLY", "XOM", "CVX", "PG",
    # US growth / mid-cap
    "NOW", "PANW", "SNOW", "NET", "DDOG", "UBER", "ABNB", "SHOP",
    # Canadian large-cap (TSX)
    "SHOP.TO", "RY.TO", "TD.TO", "CNQ.TO", "ENB.TO", "BNS.TO",
    "BMO.TO", "CP.TO", "SU.TO", "TRI.TO",
]

# How many NEW candidates to check per run. Kept conservative to stay
# safely under the free tier's 25-calls/day limit alongside whatever
# already-held positions also need a daily price check.
ROTATION_BATCH_SIZE = 6


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
    rotation_index: int = 0
    fundamentals_cache: dict = field(default_factory=dict)

    @classmethod
    def fresh(cls) -> "PortfolioState":
        return cls(
            cash_by_tier={t: STARTING_CAPITAL * c["capital_pct"] for t, c in RISK_TIERS.items()},
            positions=[],
            closed_trades=[],
            peak_equity=STARTING_CAPITAL,
            started_on=_now(),
            last_run="",
            rotation_index=0,
            fundamentals_cache={},
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
        positions=[PaperPosition(**p) for p in raw["positions"]],
        closed_trades=[ClosedTrade(**t) for t in raw["closed_trades"]],
        peak_equity=raw.get("peak_equity", STARTING_CAPITAL),
        started_on=raw.get("started_on", ""),
        last_run=raw.get("last_run", ""),
        rotation_index=raw.get("rotation_index", 0),
        fundamentals_cache=raw.get("fundamentals_cache", {}),
    )


def save_state(state: PortfolioState):
    payload = {
        "cash_by_tier": state.cash_by_tier,
        "positions": [asdict(p) for p in state.positions],
        "closed_trades": [asdict(t) for t in state.closed_trades],
        "peak_equity": state.peak_equity,
        "started_on": state.started_on,
        "last_run": state.last_run,
        "rotation_index": state.rotation_index,
        "fundamentals_cache": state.fundamentals_cache,
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


# ---------------------------------------------------------------------------
# AGENT CORE
# ---------------------------------------------------------------------------

class PaperAgent:
    def __init__(self, provider, fundamentals_lookup):
        self.provider = provider
        self.fundamentals_lookup = fundamentals_lookup
        self.state = load_state()

    def current_prices(self, tickers: list[str]) -> dict[str, float]:
        prices = {}
        for t in tickers:
            try:
                rows = self.provider.get_daily_prices(t, "2024-01-01")
                if rows:
                    prices[t] = rows[-1]["close"]
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

            reason = None
            if price <= pos.stop_price:
                reason = "stop_loss"
            else:
                try:
                    rows = self.provider.get_daily_prices(pos.ticker, "2020-01-01")
                    fast = moving_average(rows, 50)
                    slow = moving_average(rows, 200)
                    if fast is not None and slow is not None and fast < slow:
                        reason = "trend_break"
                except Exception as e:
                    log(f"WARN exit check failed for {pos.ticker}: {e}")

            if reason:
                self._close(pos, price, reason)

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

    def next_rotation_batch(self) -> list[str]:
        held = {p.ticker for p in self.state.positions}
        available = [t for t in WATCHLIST if t not in held]
        if not available:
            return []

        n = len(available)
        start = self.state.rotation_index % n
        batch = []
        for i in range(min(ROTATION_BATCH_SIZE, n)):
            batch.append(available[(start + i) % n])

        self.state.rotation_index = (start + ROTATION_BATCH_SIZE) % n
        return batch

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
            log("No new candidates this run (already holding everything in today's batch).")
            return

        log(f"Screening {len(candidates)} candidates (rotating through {len(WATCHLIST)}-stock watchlist): "
            f"{', '.join(candidates)}")
        results = screen_universe(candidates, self.provider, self.cached_fundamentals_lookup)

        counts = {tier: sum(1 for p in self.state.positions if p.tier == tier) for tier in RISK_TIERS}

        for tier in ("core", "growth", "aggressive"):
            for entry in results[tier]:
                ticker = entry["ticker"]
                if counts[tier] >= RISK_TIERS[tier]["max_positions"]:
                    break
                price = prices.get(ticker)
                if price is None:
                    continue
                if self._open(ticker, tier, price, prices):
                    counts[tier] += 1

    def cached_fundamentals_lookup(self, ticker: str) -> dict:
        cached = self.state.fundamentals_cache.get(ticker)
        if cached:
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

    def _open(self, ticker: str, tier: str, quoted_price: float, prices: dict) -> bool:
        cfg = RISK_TIERS[tier]
        tier_val = self.tier_equity(tier, prices)
        alloc = tier_val * cfg["position_size_pct"]
        commission = EXECUTION_ASSUMPTIONS["commission_per_trade"]

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
            commission_paid=commission,
        )
        self.state.positions.append(pos)
        log(f"BUY  {ticker} [{tier}] {shares:.2f} sh @ ${fill:.2f} "
            f"= ${shares*fill:,.2f}, stop ${pos.stop_price:.2f}")
        return True

    def run_once(self):
        log("=" * 55)
        log("PAPER TRADING RUN START (no real money)")

        held_tickers = {p.ticker for p in self.state.positions}
        todays_batch = self.next_rotation_batch()
        tickers = sorted(held_tickers | set(todays_batch))

        prices = self.current_prices(tickers)
        if not prices:
            log("ERROR: no prices retrieved. Check API key / rate limits. Aborting run.")
            return

        self.check_exits(prices)
        self.check_entries(prices, todays_batch)

        self.state.last_run = _now()
        save_state(self.state)

        equity = self.total_equity(prices)
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
    from backtest import AlphaVantageProvider, ALPHA_VANTAGE_API_KEY

    if ALPHA_VANTAGE_API_KEY == "YOUR_FREE_KEY_HERE":
        raise SystemExit(
            "Set ALPHA_VANTAGE_API_KEY in backtest.py first.\n"
            "Free key: https://www.alphavantage.co/support/#api-key\n\n"
            "NOTE: the free tier (25 calls/day) is too small to run this agent\n"
            "over a full universe daily. For real paper trading you need either\n"
            "a paid tier or IBKR's data feed."
        )
    return AlphaVantageProvider(ALPHA_VANTAGE_API_KEY)


def build_fundamentals_lookup(provider):
    cache = {}

    def lookup(ticker: str) -> dict:
        if ticker in cache:
            return cache[ticker]
        try:
            data = provider._get({"function": "OVERVIEW", "symbol": ticker})
            mc = data.get("MarketCapitalization")
            cache[ticker] = {"market_cap": float(mc) if mc and mc != "None" else None}
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
