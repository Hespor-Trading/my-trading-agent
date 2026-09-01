"""
Test harness using synthetic data.
Verifies the agent's logic works without needing an API key or network.

Run: python test_agent.py
"""

import os
import math
import random
from datetime import datetime, timezone, timedelta
import paper_agent
from screener import passes_liquidity, has_momentum, assign_tier, annualized_volatility


class MockProvider:
    """Generates deterministic synthetic price series with known properties."""

    def __init__(self):
        self.profiles = {
            # steady uptrend, huge cap, low vol -> should land in CORE
            "STEADY":   {"start": 100, "drift": 0.0006, "vol": 0.010, "volume": 20_000_000, "cap": 2_000e9, "qtrs": 20},
            # strong uptrend, mid cap, higher vol -> GROWTH
            "MOMO":     {"start": 50,  "drift": 0.0012, "vol": 0.025, "volume": 8_000_000,  "cap": 20e9,   "qtrs": 12},
            # wild uptrend, small cap -> AGGRESSIVE
            "WILD":     {"start": 20,  "drift": 0.0015, "vol": 0.045, "volume": 3_000_000,  "cap": 1.5e9,  "qtrs": 6},
            # downtrend -> should be rejected on MOMENTUM
            "FALLING":  {"start": 80,  "drift": -0.0010, "vol": 0.020, "volume": 5_000_000, "cap": 30e9,   "qtrs": 16},
            # illiquid penny stock -> should be rejected on LIQUIDITY
            "THIN":     {"start": 2,   "drift": 0.0020, "vol": 0.060, "volume": 50_000,     "cap": 200e6,  "qtrs": 3},
        }

    def get_daily_prices(self, ticker, start):
        p = self.profiles.get(ticker)
        if p is None:
            raise RuntimeError(f"unknown mock ticker {ticker}")
        rng = random.Random(hash(ticker) & 0xFFFFFFFF)
        rows, price = [], p["start"]
        for i in range(400):
            price *= math.exp(p["drift"] + rng.gauss(0, p["vol"]))
            rows.append({
                "date": f"2024-{(i//30)%12+1:02d}-{i%28+1:02d}",
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price, "volume": int(p["volume"] / price),
            })
        return rows

    def get_earnings(self, ticker):
        p = self.profiles.get(ticker, {})
        return [{"date": f"2024-{i:02d}-01", "reported_eps": 1.0 + i * 0.1, "estimated_eps": 1.0}
                for i in range(1, p.get("qtrs", 0) + 1)]


def mock_fundamentals(ticker):
    provider = MockProvider()
    p = provider.profiles.get(ticker, {})
    return {"market_cap": p.get("cap")}


def main():
    print("=" * 60)
    print("AGENT LOGIC TEST (synthetic data, no network, no API key)")
    print("=" * 60)

    provider = MockProvider()

    print("\n--- Screener rule checks ---")
    for ticker in provider.profiles:
        prices = provider.get_daily_prices(ticker, "2000-01-01")
        liq_ok, liq_reason = passes_liquidity(prices)
        mom_ok, mom_reason = has_momentum(prices)
        vol = annualized_volatility(prices)

        status = []
        if not liq_ok:
            status.append(f"REJECT liquidity ({liq_reason})")
        elif not mom_ok:
            status.append(f"REJECT momentum ({mom_reason})")
        else:
            fundamentals = mock_fundamentals(ticker)
            qtrs = len(provider.get_earnings(ticker))
            tier, detail = assign_tier(prices, fundamentals["market_cap"], qtrs)
            status.append(f"-> {tier or 'REJECT'}  ({detail})")

        print(f"  {ticker:<9} vol={vol:>5.0%}  {status[0]}")

    print("\n--- Full agent run (simulated) ---")
    paper_agent.WATCHLIST = list(provider.profiles.keys())
    paper_agent.STATE_FILE = "test_state.json"
    paper_agent.LOG_FILE = "test_log.txt"
    for f in ("test_state.json", "test_log.txt"):
        if os.path.exists(f):
            os.remove(f)

    agent = paper_agent.PaperAgent(provider, mock_fundamentals)
    agent.run_once()

    print("\n--- Resulting portfolio ---")
    paper_agent.show_status(agent)

    print("\n--- Verifying volatility-based position sizing ---")
    normal_pct = paper_agent.RISK_TIERS["core"]["position_size_pct"]
    calm_pct = agent._position_size_pct("STEADY", "core")   # low vol -> near full size
    wild_pct = agent._position_size_pct("WILD", "core")     # high vol -> should hit the ~50% floor
    print(f"  STEADY (calm)  -> {calm_pct:.1%} of tier value  (normal {normal_pct:.1%})")
    print(f"  WILD   (wild)  -> {wild_pct:.1%} of tier value  (normal {normal_pct:.1%})")

    sizing_ok = (
        normal_pct * 0.5 - 1e-9 <= wild_pct <= normal_pct + 1e-9
        and normal_pct * 0.5 - 1e-9 <= calm_pct <= normal_pct + 1e-9
        and wild_pct <= calm_pct
    )
    print("RESULT:", "sizing scales down with volatility, within bounds" if sizing_ok
          else "VOLATILITY SIZING BROKEN -- BUG")

    print("\n--- Verifying earnings-blackout skip logic ---")
    today = datetime.now(timezone.utc)
    # last reported 88 days ago -> next estimate lands ~3 days out -> should skip
    near_date = (today - timedelta(days=88)).strftime("%Y-%m-%d")
    # last reported 60 days ago -> next estimate lands ~31 days out -> should not skip
    far_date = (today - timedelta(days=60)).strftime("%Y-%m-%d")
    near_earnings = [{"date": near_date, "reported_eps": 1.0, "estimated_eps": 1.0}]
    far_earnings = [{"date": far_date, "reported_eps": 1.0, "estimated_eps": 1.0}]

    near_result = paper_agent.earnings_too_close(near_earnings)
    far_result = paper_agent.earnings_too_close(far_earnings)
    none_result = paper_agent.earnings_too_close([])
    print(f"  reported {near_date} (~88d ago) -> {'SKIP: ' + near_result if near_result else 'no skip'}")
    print(f"  reported {far_date} (~60d ago) -> {'SKIP: ' + far_result if far_result else 'no skip'}")
    print(f"  no earnings history             -> {'SKIP: ' + none_result if none_result else 'no skip'} (fail-open expected)")

    blackout_ok = near_result is not None and far_result is None and none_result is None
    print("RESULT:", "earnings blackout logic correct" if blackout_ok
          else "EARNINGS BLACKOUT LOGIC BROKEN -- BUG")

    print("\n--- Verifying tier capital limits held ---")
    prices = agent.current_prices(list(provider.profiles.keys()))
    total = agent.total_equity(prices)
    ok = True
    for tier, cfg in paper_agent.RISK_TIERS.items():
        tier_val = agent.tier_equity(tier, prices)
        expected_cap = paper_agent.STARTING_CAPITAL * cfg["capital_pct"]
        pct_of_total = tier_val / total if total else 0
        flag = "OK" if tier_val <= expected_cap * 1.02 else "BREACH"
        if flag == "BREACH":
            ok = False
        print(f"  {tier:<11} ${tier_val:>11,.2f}  ({pct_of_total:>5.1%} of portfolio)  "
              f"limit ${expected_cap:,.0f}  [{flag}]")

    print("\n" + "=" * 60)
    print("RESULT:", "tier isolation holding correctly" if ok else "TIER LIMIT BREACHED -- BUG")
    print("=" * 60)

    for f in ("test_state.json", "test_log.txt"):
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
