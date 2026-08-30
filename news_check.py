"""
News Sentiment Check (optional add-on)
=======================================
Uses Claude with web search to do a quick sanity check on recent news for a
candidate stock, BEFORE the agent buys it. This is a SECONDARY filter, not
the decision-maker -- the trend + earnings rules already decided the stock
is a legitimate candidate; this only asks "is there a glaring red flag in
the news right now that price/earnings data wouldn't show yet?"

Deliberately narrow: it only looks for a genuinely serious negative
catalyst (fraud, major lawsuit, guidance cut, executive scandal, regulatory
action) -- not routine volatility or mixed opinions. A stock isn't skipped
just because some article somewhere is neutral or mildly cautious.

FAILS OPEN: if this check errors for any reason (bad key, network issue,
unexpected response), it returns "neutral" rather than blocking a trade.
A broken news check should never be able to silently stop the whole agent.

COST: only called on stocks that already passed the trend/earnings/tier
filters -- typically 0-3 calls per day, not one per watchlist stock.
"""

import json
import urllib.request
import urllib.error

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"  # cheapest current model; sufficient for this narrow task


def check_news_sentiment(ticker: str, api_key: str) -> dict:
    """Returns {"verdict": "positive"|"neutral"|"negative", "summary": str}."""
    if not api_key:
        return {"verdict": "neutral", "summary": "no API key configured"}

    prompt = (
        f"Search for the most recent news (last 7 days) about {ticker} stock. "
        f"Based only on what you find, is there a clear, significant negative "
        f"catalyst (fraud, major lawsuit, guidance cut, executive scandal, "
        f"regulatory action) that a reasonable investor should know about "
        f"before buying today? Respond with ONLY a JSON object, no other text: "
        f'{{"verdict": "positive" or "neutral" or "negative", "summary": "one sentence"}}. '
        f'Use "negative" only for a genuinely serious red flag, not routine volatility '
        f"or mixed analyst opinions."
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
