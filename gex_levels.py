import os
import math
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf

SYMS = ["GLD", "QQQ"]
TARGET = {"GLD": "ORO (XAUUSD)", "QQQ": "NAS100"}
MAX_DAYS = 45
RANGE = 0.12
R = 0.04
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT = os.getenv("TG_CHAT", "")


def last_price(ticker):
    t = yf.Ticker(ticker)
    h = t.history(period="1d", interval="1m")
    if h.empty:
        h = t.history(period="5d")
    return float(h["Close"].iloc[-1])


def gold_spot():
    try:
        r = requests.get("https://api.gold-api.com/price/XAU", timeout=15)
        return float(r.json()["price"])
    except Exception:
        return last_price("GC=F")


def nas100():
    nq = last_price("NQ=F")
    try:
        a = yf.Ticker("NQ=F").history(period="5d", interval="30m")["Close"]
        b = yf.Ticker("^NDX").history(period="5d", interval="30m")["Close"]
        common = a.index.intersection(b.index)
        t = common[-1]
        return nq - (a[t] - b[t])
    except Exception:
        return nq


def target_price(sym):
    return gold_spot() if sym == "GLD" else nas100()


def chain(sym):
    t = yf.Ticker(sym)
    now = datetime.now(timezone.utc)
    rows = []
    for exp in t.options:
        dt = datetime.strptime(exp, "%Y-%m-%d").replace(hour=20, tzinfo=timezone.utc)
        days = (dt - now).total_seconds() / 86400
        if days < 0 or days > MAX_DAYS:
            continue
        T = max(days, 0.5) / 365
        oc = t.option_chain(exp)
        for df, sign in ((oc.calls, 1), (oc.puts, -1)):
            d = df[["strike", "openInterest", "impliedVolatility"]].copy()
            d["sign"] = sign
            d["T"] = T
            rows.append(d)
    if not rows:
        raise RuntimeError("sin cadena de opciones")
    df = pd.concat(rows)
    return df[(df.openInterest > 0) & (df.impliedVolatility > 0.01) & (df.impliedVolatility < 3)]


def gamma(S, K, iv, T):
    d1 = (np.log(S / K) + (R + 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))
    return np.exp(-0.5 * d1 ** 2) / (np.sqrt(2 * math.pi) * S * iv * np.sqrt(T))


def gex_by_option(S, df):
    K, iv, T = df.strike.values, df.impliedVolatility.values, df["T"].values
    return gamma(S, K, iv, T) * df.openInterest.values * df.sign.values * 100 * S ** 2 * 0.01


def levels(sym):
    spot = last_price(sym)
    df = chain(sym)
    df = df[(df.strike > spot * (1 - RANGE * 2)) & (df.strike < spot * (1 + RANGE * 2))]
    grid = np.linspace(spot * (1 - RANGE), spot * (1 + RANGE), 241)
    prof = np.array([gex_by_option(s, df).sum() for s in grid])
    flips = [grid[i] - prof[i] * (grid[i + 1] - grid[i]) / (prof[i + 1] - prof[i])
             for i in range(len(grid) - 1) if np.sign(prof[i]) != np.sign(prof[i + 1])]
    flip = min(flips, key=lambda x: abs(x - spot)) if flips else 0
    df = df.assign(g=gex_by_option(spot, df))
    by = df.groupby(["strike", "sign"]).g.sum().unstack(fill_value=0)
    cw = float(by[1].idxmax()) if 1 in by else 0
    pw = float(by[-1].idxmin()) if -1 in by else 0
    net = float(df.g.sum())
    return {"sym": sym, "spot": spot, "net": net, "flip": round(flip, 2), "cw": cw, "pw": pw}


def telegram(text):
    if not (TG_TOKEN and TG_CHAT):
        return
    requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                  json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"}, timeout=20)


def main():
    rows = []
    for s in SYMS:
        try:
            rows.append(levels(s))
        except Exception as e:
            print(f"{s}: error {e}")
    code = "|".join(f"{r['sym']}:{r['flip']},{r['cw']},{r['pw']}" for r in rows)
    lines = [f"<b>GEX {datetime.now().strftime('%d/%m')}</b>"]
    for r in rows:
        name = TARGET[r["sym"]]
        try:
            px = target_price(r["sym"])
        except Exception:
            px = r["spot"]
        k = px / r["spot"]
        reg = "POSITIVA (mercado tranquilo)" if r["net"] > 0 else "NEGATIVA (movimientos fuertes)"
        lado = "ARRIBA" if r["spot"] > r["flip"] else "ABAJO"
        lines.append(f"\n<b>{name}</b> precio {px:,.2f}")
        lines.append(f"Gamma {reg}")
        lines.append(f"Precio {lado} del flip")
        lines.append(f"🟡 Gamma Flip: {r['flip'] * k:,.2f}")
        lines.append(f"🟢 Call Wall:  {r['cw'] * k:,.2f}")
        lines.append(f"🔴 Put Wall:   {r['pw'] * k:,.2f}")
    lines.append(f"\nPara TradingView:\n<code>{code}</code>")
    msg = "\n".join(lines)
    print(msg)
    with open("levels.json", "w") as f:
        json.dump({"date": datetime.now(timezone.utc).isoformat(), "code": code, "rows": rows}, f, indent=2)
    telegram(msg)


if __name__ == "__main__":
    main()
