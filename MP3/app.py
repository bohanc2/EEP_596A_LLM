"""
Streamlit chat app for MP3 — Single Agent and Multi-Agent with conversational memory.
Run: streamlit run app.py
Uses .env for OPENAI_API_KEY and ALPHAVANTAGE_API_KEY.
"""
import os
import re
import json
import time
import sqlite3
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI

# ── Env & client ───────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or ""
MODEL_SMALL = "gpt-4o-mini"
MODEL_LARGE = "gpt-4o"
ACTIVE_MODEL = MODEL_SMALL  # set from sidebar below
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
DB_PATH = "stocks.db"


# ── Tool functions ─────────────────────────────────────────────────
def get_price_performance(tickers: list, period: str = "1y") -> dict:
    results = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
            if data.empty:
                results[ticker] = {"error": "No data — possibly delisted"}
                continue
            start = float(data["Close"].iloc[0].item())
            end = float(data["Close"].iloc[-1].item())
            results[ticker] = {
                "start_price": round(start, 2),
                "end_price": round(end, 2),
                "pct_change": round((end - start) / start * 100, 2),
                "period": period,
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}
    return results


def get_market_status() -> dict:
    return requests.get(
        f"https://www.alphavantage.co/query?function=MARKET_STATUS"
        f"&apikey={ALPHAVANTAGE_API_KEY}",
        timeout=10,
    ).json()


def get_top_gainers_losers() -> dict:
    return requests.get(
        f"https://www.alphavantage.co/query?function=TOP_GAINERS_LOSERS"
        f"&apikey={ALPHAVANTAGE_API_KEY}",
        timeout=10,
    ).json()


def get_news_sentiment(ticker: str, limit: int = 5) -> dict:
    data = requests.get(
        f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT"
        f"&tickers={ticker}&limit={limit}&apikey={ALPHAVANTAGE_API_KEY}",
        timeout=10,
    ).json()
    return {
        "ticker": ticker,
        "articles": [
            {
                "title": a.get("title"),
                "source": a.get("source"),
                "sentiment": a.get("overall_sentiment_label"),
                "score": a.get("overall_sentiment_score"),
            }
            for a in data.get("feed", [])[:limit]
        ],
    }


def query_local_db(sql: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}


def get_company_overview(ticker: str) -> dict:
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=OVERVIEW"
        f"&symbol={ticker}"
        f"&apikey={ALPHAVANTAGE_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except Exception:
        return {"error": f"No overview data for {ticker}"}
    if not data.get("Name"):
        return {"error": f"No overview data for {ticker}"}
    return {
        "ticker": ticker,
        "name": data.get("Name", ""),
        "sector": data.get("Sector", ""),
        "pe_ratio": str(data.get("PERatio", "")),
        "eps": str(data.get("EPS", "")),
        "market_cap": str(data.get("MarketCapitalization", "")),
        "52w_high": str(data.get("52WeekHigh", "")),
        "52w_low": str(data.get("52WeekLow", "")),
    }


def get_tickers_by_sector(sector: str) -> dict:
    conn = sqlite3.connect(DB_PATH)
    search = sector.strip()
    df = pd.read_sql_query(
        "SELECT ticker, company, industry FROM stocks WHERE LOWER(TRIM(sector)) = LOWER(TRIM(?))",
        conn,
        params=(search,),
    )
    if len(df) == 0:
        df = pd.read_sql_query(
            "SELECT ticker, company, industry FROM stocks WHERE industry LIKE ?",
            conn,
            params=(f"%{search}%",),
        )
    conn.close()
    return {"sector": sector, "stocks": df.to_dict(orient="records")}


# ── Schemas ───────────────────────────────────────────────────────
def _s(name, desc, props, req):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req},
        },
    }


SCHEMA_TICKERS = _s(
    "get_tickers_by_sector",
    "Return all stocks in a sector or industry from the local database. "
    "Use broad sector names ('Information Technology', 'Energy') or sub-sectors ('semiconductor', 'insurance').",
    {"sector": {"type": "string", "description": "Sector or industry name"}},
    ["sector"],
)
SCHEMA_PRICE = _s(
    "get_price_performance",
    "Get % price change for a list of tickers over a time period. Periods: '1mo','3mo','6mo','ytd','1y'.",
    {
        "tickers": {"type": "array", "items": {"type": "string"}},
        "period": {"type": "string", "default": "1y"},
    },
    ["tickers"],
)
SCHEMA_OVERVIEW = _s(
    "get_company_overview",
    "Get fundamentals for one stock: P/E ratio, EPS, market cap, 52-week high and low.",
    {"ticker": {"type": "string", "description": "Ticker symbol e.g. 'AAPL'"}},
    ["ticker"],
)
SCHEMA_STATUS = _s(
    "get_market_status",
    "Check whether global stock exchanges are currently open or closed.",
    {},
    [],
)
SCHEMA_MOVERS = _s(
    "get_top_gainers_losers",
    "Get today's top gaining, top losing, and most actively traded stocks.",
    {},
    [],
)
SCHEMA_NEWS = _s(
    "get_news_sentiment",
    "Get latest news headlines and Bullish/Bearish/Neutral sentiment scores for a stock.",
    {"ticker": {"type": "string"}, "limit": {"type": "integer", "default": 5}},
    ["ticker"],
)
SCHEMA_SQL = _s(
    "query_local_db",
    "Run a SQL SELECT on stocks.db. Table 'stocks': ticker, company, sector, industry, market_cap (Large/Mid/Small), exchange.",
    {"sql": {"type": "string", "description": "A valid SQL SELECT statement"}},
    ["sql"],
)

ALL_SCHEMAS = [
    SCHEMA_TICKERS,
    SCHEMA_PRICE,
    SCHEMA_OVERVIEW,
    SCHEMA_STATUS,
    SCHEMA_MOVERS,
    SCHEMA_NEWS,
    SCHEMA_SQL,
]

ALL_TOOL_FUNCTIONS = {
    "get_tickers_by_sector": get_tickers_by_sector,
    "get_price_performance": get_price_performance,
    "get_company_overview": get_company_overview,
    "get_market_status": get_market_status,
    "get_top_gainers_losers": get_top_gainers_losers,
    "get_news_sentiment": get_news_sentiment,
    "query_local_db": query_local_db,
}


# ── AgentResult & run_specialist_agent ─────────────────────────────
@dataclass
class AgentResult:
    agent_name: str
    answer: str
    tools_called: list = field(default_factory=list)
    raw_data: dict = field(default_factory=dict)
    confidence: float = 0.0
    issues_found: list = field(default_factory=list)
    reasoning: str = ""


def run_specialist_agent(
    agent_name: str,
    system_prompt: str,
    task: str,
    tool_schemas: list,
    max_iters: int = 8,
    verbose: bool = False,
) -> AgentResult:
    if not client:
        return AgentResult(
            agent_name=agent_name,
            answer="(OpenAI API key not set. Add OPENAI_API_KEY to .env)",
            tools_called=[],
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    tools_called = []
    raw_data = {}
    last_text = ""
    for _ in range(max_iters):
        if tool_schemas:
            resp = client.chat.completions.create(
                model=ACTIVE_MODEL,
                messages=messages,
                tools=tool_schemas,
            )
        else:
            resp = client.chat.completions.create(
                model=ACTIVE_MODEL,
                messages=messages,
            )
        msg = resp.choices[0].message
        if getattr(msg, "content", None):
            last_text = msg.content
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            answer = (msg.content or last_text or "").strip() or "(No answer returned.)"
            return AgentResult(
                agent_name=agent_name,
                answer=answer,
                tools_called=tools_called,
                raw_data=raw_data,
            )
        messages.append(msg)
        for tc in tool_calls:
            tool_name = tc.function.name
            tool_call_id = tc.id
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            if verbose:
                print(f"[{agent_name}] tool_call → {tool_name}({args})")
            if tool_name not in ALL_TOOL_FUNCTIONS:
                tool_result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    tool_result = ALL_TOOL_FUNCTIONS[tool_name](**args)
                except Exception as e:
                    tool_result = {"error": str(e)}
            tools_called.append(tool_name)
            if tool_name in raw_data:
                if isinstance(raw_data[tool_name], list):
                    raw_data[tool_name].append(tool_result)
                else:
                    raw_data[tool_name] = [raw_data[tool_name], tool_result]
            else:
                raw_data[tool_name] = tool_result
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(tool_result),
                }
            )
    fallback = (last_text or "Reached max tool-call iterations without finishing.").strip()
    return AgentResult(
        agent_name=agent_name,
        answer=fallback,
        tools_called=tools_called,
        raw_data=raw_data,
    )


SINGLE_AGENT_PROMPT = """
You are a single-agent financial analyst with tool access.
Goal: Answer the user's question accurately and concisely.
Tool-use rules (critical):
- If the question asks for CURRENT/RECENT data (prices/returns, P/E, EPS, market cap, 52-week high/low, news sentiment, market open/close), you MUST use the appropriate tool(s).
- If a tool returns an error/empty data, say so and do not fabricate numbers.
When to use which tool:
- get_company_overview(ticker): fundamentals (P/E, EPS, market cap, 52w high/low)
- get_news_sentiment(ticker): recent sentiment/headlines
- get_price_performance([tickers], period): returns over 1mo/3mo/6mo/ytd/1y
- get_tickers_by_sector(sector_or_industry): fetch tickers for a sector/industry keyword
- query_local_db(sql): filtering/slicing the local S&P500 dataset
- get_market_status(): exchange open/closed
- get_top_gainers_losers(): today's movers
Planning: For multi-step questions, chain tools (e.g. tickers → price performance → rank → summarize).
CONVERSATION CONTEXT: The user may refer to previous messages (e.g. "that", "the two", "it", "those stocks"). Use the conversation history above to resolve pronouns and references before calling tools.
Answer format: Use short bullets or a compact table. Include tickers and key numbers sourced from tools.
""".strip()

MARKET_TOOLS = [SCHEMA_TICKERS, SCHEMA_PRICE, SCHEMA_STATUS, SCHEMA_MOVERS, SCHEMA_SQL]
FUNDAMENTAL_TOOLS = [SCHEMA_OVERVIEW, SCHEMA_SQL, SCHEMA_TICKERS]
SENTIMENT_TOOLS = [SCHEMA_NEWS, SCHEMA_SQL, SCHEMA_TICKERS]

MARKET_SPECIALIST_PROMPT = """
You are the Market Specialist. You handle: finding relevant tickers (sector/industry/SQL), computing return/performance, market status and top movers.
Rules: Use tools; do not guess tickers or numbers. When selecting many tickers, LIMIT to a manageable number. If the question asks for "top N" by return, you MUST call get_price_performance and compute the ranking.
""".strip()

FUNDAMENTALS_SPECIALIST_PROMPT = """
You are the Fundamentals Specialist. You handle: Company overview fundamentals (P/E, EPS, market cap, 52-week high/low). Use get_company_overview for each requested ticker. Do not invent numbers.
""".strip()

SENTIMENT_SPECIALIST_PROMPT = """
You are the Sentiment Specialist. You handle: Recent news sentiment for tickers. Use get_news_sentiment for each ticker. Summarize sentiment labels briefly.
""".strip()


def _extract_tickers_from_text(q: str) -> list:
    candidates = re.findall(r"\b[A-Z]{1,5}\b", q or "")
    banned = {"P", "E", "EPS", "YTD", "CEO", "CFO", "USD"}
    out = []
    for t in candidates:
        if t in banned or t in out:
            continue
        if len(t) < 2:  # ignore single letters (e.g. "M" from "Microsoft")
            continue
        out.append(t)
    return out


def _resolve_tickers_from_conversation(q: str) -> list:
    """Use LLM to resolve 'that', 'the two', etc. from conversation context. Returns list of ticker symbols."""
    if not client or not q:
        return []
    if "User:" not in q and "Conversation so far" not in q:
        return []
    try:
        resp = client.chat.completions.create(
            model=ACTIVE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a resolver. Given a conversation about stocks, output ONLY a JSON object with one key: \"tickers\", a list of stock ticker symbols (e.g. [\"NVDA\", \"AMD\", \"AAPL\"]). Use the conversation to resolve pronouns: 'that' = the stock just discussed, 'the two' = the two stocks just compared. If the latest question asks about specific tickers mentioned earlier, include them. Use uppercase 2-5 letter symbols. If none or unclear, use []. No other text.",
                },
                {"role": "user", "content": q},
            ],
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        tickers = data.get("tickers", [])
        if isinstance(tickers, list):
            return [str(t).upper() for t in tickers if isinstance(t, str) and 1 <= len(t) <= 5]
        return []
    except Exception:
        return []


def _pick_period_from_question(q: str) -> str:
    s = (q or "").lower()
    if "ytd" in s or "year to date" in s:
        return "ytd"
    if "6-month" in s or "6 month" in s or "six month" in s:
        return "6mo"
    if "3-month" in s or "3 month" in s:
        return "3mo"
    if "this month" in s or "past month" in s or "last month" in s or "1-month" in s or "1 month" in s:
        return "1mo"
    if "1-year" in s or "1 year" in s or "past year" in s:
        return "1y"
    return "1y"


def _tool_has_error(obj) -> bool:
    if isinstance(obj, dict):
        if "error" in obj:
            return True
        return any(_tool_has_error(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_tool_has_error(x) for x in obj)
    return False


def _extract_price_map(agent: AgentResult) -> dict:
    data = agent.raw_data.get("get_price_performance")
    if isinstance(data, list):
        data = data[-1] if data else None
    return data if isinstance(data, dict) else {}


def _extract_sector_rows(agent: AgentResult) -> list:
    data = agent.raw_data.get("get_tickers_by_sector")
    if isinstance(data, list):
        data = data[-1] if data else None
    if isinstance(data, dict):
        return data.get("stocks", []) or []
    return []


def _rank_top_n_by_return(price_map: dict, n: int = 3) -> list:
    rows = []
    for t, v in (price_map or {}).items():
        if not isinstance(v, dict) or "error" in v or "pct_change" not in v:
            continue
        try:
            rows.append((t, float(v["pct_change"])))
        except Exception:
            continue
    rows.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in rows[:n]]


def _sentiment_summary(sent_data: dict) -> str:
    if not isinstance(sent_data, dict):
        return "No sentiment data"
    arts = sent_data.get("articles") or []
    if not arts:
        return "No recent articles"
    counts = {}
    for a in arts:
        lab = a.get("sentiment") or "Unknown"
        counts[lab] = counts.get(lab, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    return f"{top} (n={len(arts)})"


def run_single_agent(question: str, verbose: bool = False) -> AgentResult:
    return run_specialist_agent(
        agent_name="Single Agent",
        system_prompt=SINGLE_AGENT_PROMPT,
        task=question,
        tool_schemas=ALL_SCHEMAS,
        max_iters=10,
        verbose=verbose,
    )


def run_multi_agent(question: str, verbose: bool = False) -> dict:
    t0 = time.time()
    q = question or ""
    period = _pick_period_from_question(q)
    explicit_tickers = _extract_tickers_from_text(q)
    resolved_tickers = _resolve_tickers_from_conversation(q)
    combined_tickers = list(dict.fromkeys((resolved_tickers or []) + (explicit_tickers or [])))
    # Use only the current user question (first line after "New user message:") for need_* to avoid false triggers
    if "New user message:" in q:
        after_label = q.split("New user message:")[-1].strip()
        current_msg = after_label.split("\n")[0].strip().lower() if after_label else q.lower()
    else:
        current_msg = q.lower()
    if not current_msg:
        current_msg = q.lower()
    q_lower = q.lower()
    need_market = any(
        k in q_lower
        for k in [
            "return", "performance", "dropped", "grew", "gainers", "losers", "most active",
            "sector", "industry", "stocks", "market status", "open", "closed",
            "top 3", "top 5", "best", "worst", "better return", "had the better",
        ]
    )
    if (not need_market) and combined_tickers and ("what about" in q_lower or "how about" in q_lower):
        need_market = True
    # Only run Fundamentals when the current message explicitly asks for fundamental data (P/E, valuation, etc.)
    # Do not trigger from explicit_tickers alone — "What about Microsoft?" should not run Fundamentals.
    need_fundamentals = any(
        k in current_msg
        for k in ["p/e", "pe ratio", "eps", "market cap", "52", "fundamental", "valuation", "compare"]
    )
    need_sentiment = any(
        k in current_msg for k in ["sentiment", "news", "headline", "bullish", "bearish"]
    )
    agent_results = []
    market_res = None
    chosen_tickers = combined_tickers if combined_tickers else []
    # When we already have tickers (e.g. from resolver) and need return/performance → direct price task
    if need_market and chosen_tickers:
        market_task = (
            f"Get price performance for these tickers only: {', '.join(chosen_tickers)}. "
            f"Use period='{period}'. Call get_price_performance with these exact tickers and return the comparison."
        )
        market_res = run_specialist_agent(
            agent_name="Market Specialist",
            system_prompt=MARKET_SPECIALIST_PROMPT,
            task=market_task,
            tool_schemas=MARKET_TOOLS,
            max_iters=10,
            verbose=verbose,
        )
        agent_results.append(market_res)
    elif need_market and not explicit_tickers:
        market_task = (
            f"Question: {q}\n\n"
            f"If returns are requested, use period='{period}'. "
            f"If sector/industry keyword is mentioned, use get_tickers_by_sector first; if too many, use query_local_db with LIMIT 50."
        )
        market_res = run_specialist_agent(
            agent_name="Market Specialist",
            system_prompt=MARKET_SPECIALIST_PROMPT,
            task=market_task,
            tool_schemas=MARKET_TOOLS,
            max_iters=10,
            verbose=verbose,
        )
        agent_results.append(market_res)
    if not chosen_tickers and market_res is not None:
        price_map = _extract_price_map(market_res)
        if price_map:
            chosen_tickers = _rank_top_n_by_return(price_map, n=3)
        else:
            sector_rows = _extract_sector_rows(market_res)
            chosen_tickers = [r.get("ticker") for r in sector_rows if r.get("ticker")][:3]
    if not chosen_tickers and not explicit_tickers:
        for kw in ["semiconductor", "energy", "information technology", "technology", "health care", "financial"]:
            if kw in q.lower():
                try:
                    fallback = get_tickers_by_sector(kw)
                    chosen_tickers = [r.get("ticker") for r in (fallback.get("stocks") or []) if r.get("ticker")][:3]
                except Exception:
                    chosen_tickers = []
                break
    fundamentals_res = None
    if need_fundamentals and chosen_tickers:
        fundamentals_task = (
            "Get company overview fundamentals for these tickers: "
            + ", ".join(chosen_tickers)
            + ". Return P/E ratio (pe_ratio), EPS, market cap, 52w high/low."
        )
        fundamentals_res = run_specialist_agent(
            agent_name="Fundamentals Specialist",
            system_prompt=FUNDAMENTALS_SPECIALIST_PROMPT,
            task=fundamentals_task,
            tool_schemas=FUNDAMENTAL_TOOLS,
            max_iters=10,
            verbose=verbose,
        )
        agent_results.append(fundamentals_res)
    sentiment_res = None
    if need_sentiment and chosen_tickers:
        sentiment_task = (
            "Get current news sentiment for these tickers: "
            + ", ".join(chosen_tickers)
            + ". Use limit=3 per ticker and summarize briefly."
        )
        sentiment_res = run_specialist_agent(
            agent_name="Sentiment Specialist",
            system_prompt=SENTIMENT_SPECIALIST_PROMPT,
            task=sentiment_task,
            tool_schemas=SENTIMENT_TOOLS,
            max_iters=10,
            verbose=verbose,
        )
        agent_results.append(sentiment_res)
    all_issues = []
    for r in agent_results:
        if _tool_has_error(r.raw_data):
            r.issues_found.append("tool_error")
            all_issues.append(f"{r.agent_name}: tool_error")
            r.confidence = min(r.confidence or 1.0, 0.5)
        else:
            r.confidence = r.confidence or 0.8
    lines = ["Architecture: pipeline-specialists"]
    if chosen_tickers:
        lines.append(f"Tickers used: {', '.join(chosen_tickers)}")
    returns_by_ticker = {}
    if market_res is not None:
        pm = _extract_price_map(market_res)
        for t in chosen_tickers:
            v = (pm or {}).get(t)
            if isinstance(v, dict) and "pct_change" in v and "error" not in v:
                returns_by_ticker[t] = v.get("pct_change")
    fundamentals_by_ticker = {}
    if fundamentals_res is not None:
        od = fundamentals_res.raw_data.get("get_company_overview")
        if isinstance(od, list):
            for item in od:
                if isinstance(item, dict) and item.get("ticker"):
                    fundamentals_by_ticker[item["ticker"]] = item
        elif isinstance(od, dict) and od.get("ticker"):
            fundamentals_by_ticker[od["ticker"]] = od
    sentiment_by_ticker = {}
    if sentiment_res is not None:
        sd = sentiment_res.raw_data.get("get_news_sentiment")
        if isinstance(sd, list):
            for item in sd:
                if isinstance(item, dict) and item.get("ticker"):
                    sentiment_by_ticker[item["ticker"]] = item
        elif isinstance(sd, dict) and sd.get("ticker"):
            sentiment_by_ticker[sd["ticker"]] = sd
    if chosen_tickers:
        lines.append("\nResults:")
        for t in chosen_tickers:
            parts = [t]
            if t in returns_by_ticker:
                parts.append(f"return({period})={returns_by_ticker[t]}%")
            if t in fundamentals_by_ticker:
                f = fundamentals_by_ticker[t]
                parts.append(f"P/E={f.get('pe_ratio', '')}")
                if f.get("eps"):
                    parts.append(f"EPS={f.get('eps')}")
            if t in sentiment_by_ticker:
                parts.append(f"sentiment={_sentiment_summary(sentiment_by_ticker[t])}")
            lines.append("- " + " | ".join(parts))
        # If the question asked "which had the better return", add an explicit conclusion
        if len(returns_by_ticker) >= 2 and any(
            phrase in current_msg for phrase in ["better return", "had the better", "which of the two", "which had the better"]
        ):
            best_ticker = max(returns_by_ticker, key=lambda t: returns_by_ticker[t])
            best_pct = returns_by_ticker[best_ticker]
            others = [f"{t} ({returns_by_ticker[t]}%)" for t in chosen_tickers if t != best_ticker]
            lines.append(f"\nConclusion: {best_ticker} had the better return ({best_pct}%) compared to {', '.join(others)}.")
    else:
        lines.append("\n(Unable to confidently determine tickers. Specialist outputs below.)")
        for r in agent_results:
            lines.append(f"\n[{r.agent_name}]\n{r.answer.strip()}")
    if all_issues:
        lines.append("\nIssues detected:")
        for iss in all_issues:
            lines.append(f"- {iss}")
    final_answer = "\n".join(lines).strip()
    elapsed = time.time() - t0
    return {
        "final_answer": final_answer,
        "agent_results": agent_results,
        "elapsed_sec": float(elapsed),
        "architecture": "pipeline-specialists",
    }


# ── Streamlit UI ───────────────────────────────────────────────────
st.set_page_config(page_title="MP3 FinTech Agents", layout="wide")

st.sidebar.title("MP3 — Agentic FinTech")
st.sidebar.markdown("---")
agent_choice = st.sidebar.radio(
    "Agent architecture",
    ["Single Agent", "Multi-Agent"],
    index=0,
    help="Single Agent: one LLM with all 7 tools. Multi-Agent: pipeline of Market, Fundamentals, Sentiment specialists.",
)
model_choice = st.sidebar.radio(
    "Model",
    ["gpt-4o-mini", "gpt-4o"],
    index=0,
    help="OpenAI model to use for the selected agent.",
)
st.sidebar.markdown("---")
st.sidebar.caption("Conversation history is passed to the agent each turn so follow-up questions (e.g. 'How does that compare to AMD?') resolve correctly.")

# Set active model for this run (used by run_specialist_agent / run_multi_agent)
ACTIVE_MODEL = model_choice

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Clear conversation", use_container_width=True):
    st.session_state.messages = []
    st.rerun()


def build_context_for_agent(messages: list) -> str:
    if not messages:
        return ""
    parts = []
    for m in messages:
        role = m.get("role", "")
        raw = m.get("content")
        if role == "user":
            content = (raw if isinstance(raw, str) else "").strip()
        elif role == "assistant":
            content = raw.get("answer", str(raw)) if isinstance(raw, dict) else (raw or "").strip()
        else:
            content = ""
        if not content:
            continue
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    return "\n\n".join(parts) if parts else ""


st.title("FinTech QA Chat")
st.caption(f"Architecture: **{agent_choice}**  |  Model: **{model_choice}**  |  Context from previous turns is included so follow-ups work.")

for msg in st.session_state.messages:
    role = msg.get("role")
    content = msg.get("content")
    if role == "user":
        with st.chat_message("user"):
            st.write(content)
    elif role == "assistant":
        with st.chat_message("assistant"):
            if isinstance(content, dict):
                meta = content.get("meta", "")
                if meta:
                    st.caption(meta)
                st.write(content.get("answer", str(content)))
            else:
                st.write(content)

if prompt := st.chat_input("Ask a finance question (e.g. What is NVIDIA's P/E ratio?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    context = build_context_for_agent(st.session_state.messages[:-1])
    if context:
        task = f"""Conversation so far:

{context}

---

New user message: {prompt}

Answer the new user message. Use the conversation history to resolve any references (e.g. "that", "the two", "it", "those stocks"). Call tools as needed with the correct tickers or criteria."""
    else:
        task = prompt

    with st.spinner(f"Running {agent_choice}..."):
        try:
            if agent_choice == "Single Agent":
                result = run_single_agent(task, verbose=False)
                answer = result.answer
                meta = f"**{agent_choice}** · **{model_choice}** · tools: {', '.join(result.tools_called) or 'none'}"
            else:
                out = run_multi_agent(task, verbose=False)
                answer = out["final_answer"]
                agents_ran = [r.agent_name for r in out.get("agent_results", [])]
                meta = f"**{agent_choice}** · **{model_choice}** · architecture: {out.get('architecture', '')} · agents: {', '.join(agents_ran) or 'none'}"
        except Exception as e:
            answer = f"Error: {e}"
            meta = f"**{agent_choice}** · **{model_choice}**"

    st.session_state.messages.append({
        "role": "assistant",
        "content": {"answer": answer, "meta": meta},
    })
    st.rerun()
