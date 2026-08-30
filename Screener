"""
Rule-Based Tier Screener
========================
Instead of a human (or an AI) hand-picking which stocks go in which risk
tier, this assigns tiers by objective, checkable rules from live data.

WHY RULES INSTEAD OF PICKS:
  A hand-picked list goes stale, reflects whoever picked it, and can't be
  audited. A rule set can be tested against history, explains every decision,
  and updates itself. If a stock enters the aggressive tier, you can point at
  exactly which numbers put it there.

TIER ASSIGNMENT LOGIC:
  core        - large cap, low volatility, established earnings history
  growth      - mid/large cap, strong momentum, some earnings history
  aggressive  - smaller cap OR high volatility, still liquid enough to exit

LIQUIDITY FLOOR (applies to ALL tiers, non-negotiable):
  Every candidate must clear a minimum dollar-volume bar. This is the single
  most important filter in this file. Thin-volume stocks are where retail
  accounts die: you can get in, but when the trend breaks you cannot get out
  at anything near the quoted price, and your stop-loss becomes fiction.
  A stop-loss you cannot execute is not risk management.
"""

import statistics
from typing import Optional


# ---------------------------------------------------------------------------
# SCREENING RULES - these are the knobs worth tuning
# ---------------------------------------------------------------------------

LIQUIDITY_RULES = {
    # Minimum average daily dollar volume (price x shares traded).
    # $5M/day means you can move a few thousand dollars without being a
    # meaningful part of the day's volume. Do not lower this casually.
    "min_avg_dollar_volume": 5_000_000,
    "lookback_days": 60,
    # Minimum price. Sub-$5 stocks have wider spreads, worse fills, and are
    # disproportionately represented in pump-and-dump schemes.
    "min_price": 5.00,
}

TIER_RULES = {
    "core": {
        "min_market_cap": 50_000_000_000,   # $50B+
        "max_annualized_volatility": 0.35,
        "min_earnings_quarters": 8,          # 2+ years of reported earnings
    },
    "growth": {
        "min_market_cap": 5_000_000_000,    # $5B - $50B
        "max_annualized_volatility": 0.60,
        "min_earnings_quarters": 4,
    },
    "aggressive": {
        "min_market_cap": 500_000_000,      # $500M+ -- still a real company
        "max_annualized_volatility": 1.20,   # tolerant, but not unlimited
        "min_earnings_quarters": 2,          # must have SOME reporting history
    },
}

MOMENTUM_RULES = {
    "fast_ma": 50,
    "slow_ma": 200,
    "min_return_6mo": 0.0,  # must be up over 6 months to be a momentum candidate
}


# ---------------------------------------------------------------------------
# METRIC CALCULATION
# ---------------------------------------------------------------------------

def avg_dollar_volume(prices: list[dict], days: int) -> float:
    recent = prices[-days:]
    if not recent:
        return 0.0
    return sum(p["close"] * p["volume"] for p in recent) / len(recent)


def annualized_volatility(prices: list[dict], days: int = 252) -> float:
    """Standard deviation of daily returns, annualized. Higher = wilder swings."""
    recent = prices[-days:]
    if len(recent) < 20:
        return float("inf")
    returns = []
    for i in range(1, len(recent)):
        prev, curr = recent[i - 1]["close"], recent[i]["close"]
        if prev > 0:
            returns.append((curr / prev) - 1)
    if len(returns) < 2:
        return float("inf")
    return statistics.stdev(returns) * (252 ** 0.5)


def return_over(prices: list[dict], days: int) -> Optional[float]:
    if len(prices) < days + 1:
        return None
    start, end = prices[-days - 1]["close"], prices[-1]["close"]
    return (end / start) - 1 if start > 0 else None


def moving_average(prices: list[dict], window: int) -> Optional[float]:
    if len(prices) < window:
        return None
    return sum(p["close"] for p in prices[-window:]) / window


# ---------------------------------------------------------------------------
# SCREENING
# ---------------------------------------------------------------------------

def passes_liquidity(prices: list[dict]) -> tuple[bool, str]:
    """Returns (passed, reason_if_failed). Applied to every tier."""
    if not prices:
        return False, "no price data"
    last_price = prices[-1]["close"]
    if last_price < LIQUIDITY_RULES["min_price"]:
        return False, f"price ${last_price:.2f} below ${LIQUIDITY_RULES['min_price']} floor"
    adv = avg_dollar_volume(prices, LIQUIDITY_RULES["lookback_days"])
    if adv < LIQUIDITY_RULES["min_avg_dollar_volume"]:
        return False, f"avg daily volume ${adv:,.0f} below ${LIQUIDITY_RULES['min_avg_dollar_volume']:,} floor"
    return True, ""


def has_momentum(prices: list[dict]) -> tuple[bool, str]:
    fast = moving_average(prices, MOMENTUM_RULES["fast_ma"])
    slow = moving_average(prices, MOMENTUM_RULES["slow_ma"])
    if fast is None or slow is None:
        return False, "insufficient price history for moving averages"
    price = prices[-1]["close"]
    if not (price > fast > slow):
        return False, "not in confirmed uptrend (need price > 50MA > 200MA)"
    r6 = return_over(prices, 126)
    if r6 is None or r6 < MOMENTUM_RULES["min_return_6mo"]:
        return False, f"6-month return {r6:.1%} below threshold" if r6 is not None else "no 6mo history"
    return True, ""


def assign_tier(
    prices: list[dict],
    market_cap: Optional[float],
    earnings_quarters: int,
) -> tuple[Optional[str], str]:
    """Assign the most conservative tier this stock qualifies for.

    Conservative-first ordering is deliberate: a stock that could belong in
    core should be sized like a core holding, not dropped into aggressive
    just because it also clears the looser bar.
    """
    vol = annualized_volatility(prices)
    if market_cap is None:
        return None, "no market cap data"

    for tier in ("core", "growth", "aggressive"):
        rules = TIER_RULES[tier]
        if market_cap < rules["min_market_cap"]:
            continue
        if vol > rules["max_annualized_volatility"]:
            continue
        if earnings_quarters < rules["min_earnings_quarters"]:
            continue
        return tier, f"cap=${market_cap/1e9:.1f}B vol={vol:.0%} earnings_qtrs={earnings_quarters}"

    return None, (
        f"fails all tiers: cap=${market_cap/1e9:.1f}B vol={vol:.0%} "
        f"earnings_qtrs={earnings_quarters}"
    )


def screen_universe(
    universe: list[str],
    provider,
    fundamentals_lookup,
) -> dict:
    """Screen a list of tickers and return tier assignments + rejection reasons.

    provider: object with get_daily_prices(ticker, start) and get_earnings(ticker)
    fundamentals_lookup: callable(ticker) -> {"market_cap": float or None}
    """
    results = {"core": [], "growth": [], "aggressive": [], "rejected": {}}

    for ticker in universe:
        try:
            prices = provider.get_daily_prices(ticker, "2000-01-01")
        except Exception as e:
            results["rejected"][ticker] = f"data fetch failed: {e}"
            continue

        ok, reason = passes_liquidity(prices)
        if not ok:
            results["rejected"][ticker] = f"LIQUIDITY: {reason}"
            continue

        ok, reason = has_momentum(prices)
        if not ok:
            results["rejected"][ticker] = f"MOMENTUM: {reason}"
            continue

        try:
            earnings = provider.get_earnings(ticker)
            earnings_quarters = len([e for e in earnings if e.get("reported_eps") is not None])
        except Exception:
            earnings_quarters = 0

        fundamentals = fundamentals_lookup(ticker) or {}
        tier, detail = assign_tier(prices, fundamentals.get("market_cap"), earnings_quarters)

        if tier is None:
            results["rejected"][ticker] = f"TIER: {detail}"
        else:
            results[tier].append({"ticker": ticker, "detail": detail})

    return results


def print_screen_results(results: dict):
    print("\n" + "=" * 60)
    print("SCREEN RESULTS")
    print("=" * 60)
    for tier in ("core", "growth", "aggressive"):
        entries = results[tier]
        print(f"\n[{tier.upper()}] -- {len(entries)} qualified")
        for e in entries:
            print(f"   {e['ticker']:<10} {e['detail']}")

    print(f"\n[REJECTED] -- {len(results['rejected'])}")
    for ticker, reason in results["rejected"].items():
        print(f"   {ticker:<10} {reason}")
    print("\n" + "=" * 60)
    print("Every rejection above is a rule doing its job. If a name you")
    print("wanted got rejected on LIQUIDITY, that is the filter working --")
    print("resist the urge to lower that floor to let a favourite through.")
