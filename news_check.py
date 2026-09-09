"""
News + Valuation Check (optional add-on)
=========================================
Uses Claude with web search to do a quick sanity check on recent news AND
current valuation for a candidate stock, BEFORE the agent buys it. This is
a SECONDARY filter, not the decision-maker -- the trend + earnings rules
already decided the stock is a legitimate candidate; this only asks "is
there a glaring red flag right now that price/earnings data alone wouldn't
show?" -- either a news catalyst or a valuation that has become disconnected
from the company's actual growth/profitability.

Deliberately narrow: it only looks for a genuinely serious negative
catalyst (fraud, major lawsuit, guidance cut, executive scandal, regulatory
action) or an extreme valuation disconnect -- not routine volatility, mixed
opinions, or a high P/E that's well-supported by strong growth and margins.
A stock isn't skipped just because some article is neutral, or its P/E is
merely on the high side.

FAILS OPEN: if this check errors for any reason (bad key, network issue,
unexpected response), it returns "neutral" rather than blocking a trade.
A broken check should never be able to silently stop the whole agent. Same
principle for fundamentals: any missing metric (None) is passed through as
"unknown" in the prompt rather than treated as a red flag on its own --
plenty of legitimate candidates (e.g. pre-earnings growth names) have no
trailing P/E simply because they have no/negative earnings yet.

COST: only called on stocks that already passed the trend/earnings/tier
filters -- typically 0-3 calls per day, not one per watchlist stock.
"""

import json
import urllib.request
import urllib.error

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-5"  # more capable model for better-quality judgment on company news, at a modestly higher per-check cost


def _fmt_pct(x):
    return f"{x:+.1%}" if x is not None else "N/A"


def _fmt_ratio(x):
    return f"{x:.1f}" if x is not None else "N/A (no/negative earnings, or unavailable)"


def check_news_sentiment(ticker: str, api_key: str, fundamentals: dict = None) -> dict:
    """Returns {"verdict": "positive"|"neutral"|"negative", "summary": str}.

    fundamentals, if given, is {"pe_ratio", "forward_pe", "revenue_growth",
    "profit_margin"} (any value may be None -- see FAILS OPEN above)."""
    if not api_key:
        return {"verdict": "neutral", "summary": "no API key configured"}

    fundamentals = fundamentals or {}
    valuation_line = (
        f"Current valuation: trailing P/E {_fmt_ratio(fundamentals.get('pe_ratio'))}, "
        f"forward P/E {_fmt_ratio(fundamentals.get('forward_pe'))}, "
        f"revenue growth {_fmt_pct(fundamentals.get('revenue_growth'))}, "
        f"profit margin {_fmt_pct(fundamentals.get('profit_margin'))}.\n\n"
    )

    prompt = (
        f"Search for the most recent news (last 7 days) about {ticker} stock.\n\n"
        f"{valuation_line}"
        f"Based on the news AND the valuation data above, is there a clear, "
        f"significant red flag a reasonable investor should know about before "
        f"buying today? This means either: (a) a serious negative news catalyst "
        f"(fraud, major lawsuit, guidance cut, executive scandal, regulatory "
        f"action), or (b) an extreme valuation disconnect -- a very high P/E "
        f"combined with weak, negative, or decelerating revenue growth and "
        f"margins, suggesting the price already assumes near-perfect execution "
        f"with no room for error. Respond with ONLY a JSON object, no other text: "
        f'{{"verdict": "positive" or "neutral" or "negative", "summary": "one sentence"}}. '
        f'Use "negative" only for a genuinely serious red flag -- not routine '
        f"volatility, mixed analyst opinions, or a high P/E that's well-supported "
        f"by strong growth and margins. A missing (N/A) metric is not itself a "
        f"red flag."
    )

    body = json.dumps({
        "model": MODEL,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"type": "web_search_20250305", "name": "web_search"}],
    }).encode()

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return {"verdict": "neutral", "summary": f"news check failed: {e}"}

    text_parts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    full_text = " ".join(text_parts).strip()

    try:
        start = full_text.index("{")
        end = full_text.rindex("}") + 1
        parsed = json.loads(full_text[start:end])
        verdict = parsed.get("verdict", "neutral")
        if verdict not in ("positive", "neutral", "negative"):
            verdict = "neutral"
        return {"verdict": verdict, "summary": parsed.get("summary", "")}
    except Exception:
        return {"verdict": "neutral", "summary": "could not parse response"}
