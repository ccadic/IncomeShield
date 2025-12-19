#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backtest GUI (Tkinter) — Portefeuille Rendement US (BDC + Covered Call) + Benchmarks
===================================================================================

✅ Yahoo Finance SANS yfinance (endpoint chart)
✅ Corrige le bug d'apports "fin de série"
✅ Force une série journalière propre (1 ligne / jour)
✅ Ajoute coûts :
   - FX (ex: 0.15% sur conversion EUR<->USD)
   - transaction (commission + slippage, ex: 0.02% + 0.01%)
✅ Benchmarks : SPY (actions), VNQ (foncières US), BND et/ou AGG (obligations)
✅ Sorties :
   - CSV equity_curve.csv, trades.csv
   - CSV monthly_report.csv (perf, drawdown, cashflow)
✅ Option (checkbox) :
   - "Retirer les revenus mensuels" (cashflow mensuel simulé) vs réinvestissement
✅ Résultats "verbose" affichés dans le terminal (et dans un panneau texte)

✅ NOUVEAU : Mode "Risk-Off" (réduction BDC automatique)
   - Signal Crédit : HYG/LQD < SMA(200)
   - Signal Stress : (SPY < SMA(200) & VIX > seuil) OU (Drawdown SPY <= seuil)
   - Anti-whipsaw : entrée si signal vrai N jours ; sortie si faux M jours
   - Action : BDC 60% -> 30% ; Covered Call optionnel (ex 100%/75%/50%)
   - Défensif : CASH ou AGG/BND ou cash-like proxies (SGOV/BIL/SHV)

Remarques importantes (réalisme):
- Les "revenus" sont estimés à partir de la variation d'Adj Close (approximation).
  Les distributions réelles (dividendes/ROC) ne sont pas reconstruites exactement.
- Frais/fiscalité/withholding tax ne sont pas intégrés.

Réglages conseillés (pour ton cas)
Entrée : 7 jours
Sortie : 14 jours
Défensif : AUTO (=> SGOV puis BIL puis SHV)
BDC scale : 0.50 (ton 60% → 30%)
CC scale : commence à 1.00, puis teste 0.75 si tu veux vraiment réduire le bêta en stress.
"""

import os
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import requests
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# -----------------------------
# CONFIG (Yahoo fetch)
# -----------------------------
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

FX_SYMBOL = "EURUSD=X"  # 1 EUR = x USD

DEFAULT_WEIGHTS = {
    "MAIN": 0.20,
    "ARCC": 0.15,
    "BXSL": 0.15,
    "OBDC": 0.10,   # ou CSWC
    "JEPI": 0.15,
    "JEPQ": 0.10,
    "DIVO": 0.10,
    "BST":  0.05,
}

BENCHMARKS_DEFAULT = {
    "SPY": 1.0,   # actions US
    "VNQ": 1.0,   # REITs / foncières US
    "BND": 1.0,   # obligations US broad
    "AGG": 1.0,   # obligations US aggregate
}

# Signaux risk-off (tickers Yahoo)
HYG_SYMBOL = "HYG"
LQD_SYMBOL = "LQD"
VIX_SYMBOL = "^VIX"
SPY_SYMBOL = "SPY"  # utilisé pour signaux (même si benchmark décoché)

# Cash-like proxies
CASHLIKE_CANDIDATES = ["SGOV", "BIL", "SHV"]


# -----------------------------
# Yahoo fetch (sans yfinance)
# -----------------------------
def fetch_yahoo_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    p1 = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
    p2 = int(end_dt.replace(tzinfo=timezone.utc).timestamp())

    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{requests.utils.quote(symbol, safe='')}"
           f"?period1={p1}&period2={p2}&interval=1d&events=div,splits")

    headers = {"User-Agent": UA, "Accept": "*/*"}

    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "chart" not in data or not data["chart"].get("result"):
        raise RuntimeError(f"Yahoo: réponse inattendue pour {symbol}: {data}")

    res = data["chart"]["result"][0]
    ts = res.get("timestamp") or []
    if not ts:
        raise RuntimeError(f"Yahoo: pas de timestamps pour {symbol}")

    ind = pd.to_datetime(pd.Series(ts), unit="s", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)

    indicators = res.get("indicators", {})
    adj = indicators.get("adjclose", [{}])[0].get("adjclose")
    close = indicators.get("quote", [{}])[0].get("close")

    if adj and any(x is not None for x in adj):
        prices = pd.Series(adj, index=ind, name="close")
    else:
        prices = pd.Series(close, index=ind, name="close")

    df = prices.to_frame().dropna()
    df.index.name = "date"
    df.index = pd.to_datetime(df.index.date)  # normalise jour
    df = df[~df.index.duplicated(keep="last")]
    return df


# -----------------------------
# Helpers calendrier / stats
# -----------------------------
def month_start_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    df = pd.DataFrame(index=dates)
    df["ym"] = df.index.to_period("M")
    firsts = df.groupby("ym").apply(lambda x: x.index.min())
    return pd.DatetimeIndex(firsts.values)


def year_anniversary_dates(start_date: pd.Timestamp,
                           dates: pd.DatetimeIndex,
                           years: int) -> List[pd.Timestamp]:
    last = dates[-1]
    out = []
    for k in range(1, years + 1):
        target = start_date + pd.DateOffset(years=k)
        if target > last:
            break
        pos = dates.searchsorted(target)
        if pos < len(dates):
            out.append(dates[pos])
    return sorted(set(out))


def max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    dd = series / peak - 1.0
    return float(dd.min())


def cagr(series: pd.Series, freq=252) -> float:
    rets = series.pct_change().dropna()
    if rets.empty:
        return 0.0
    return float((series.iloc[-1] / series.iloc[0]) ** (freq / len(rets)) - 1)


def vol_annual(series: pd.Series, freq=252) -> float:
    rets = series.pct_change().dropna()
    if rets.empty:
        return 0.0
    return float(rets.std() * math.sqrt(freq))


def sharpe0(series: pd.Series, freq=252) -> float:
    rets = series.pct_change().dropna()
    if rets.empty:
        return 0.0
    denom = rets.std() * math.sqrt(freq) + 1e-12
    return float((rets.mean() * freq) / denom)


def as_pct(x: float) -> str:
    return f"{x*100:,.2f}%"


def money(x: float, ccy="EUR") -> str:
    return f"{x:,.2f} {ccy}"


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=n).mean()


# -----------------------------
# Backtest core
# -----------------------------
@dataclass
class FeesConfig:
    fx_fee_pct: float
    tx_fee_pct: float
    slippage_pct: float


@dataclass
class RiskOffConfig:
    enabled: bool
    use_credit_signal: bool
    use_spy_vix_signal: bool
    sma_len: int
    vix_threshold: float
    spy_drawdown_threshold: float

    # anti-whipsaw
    entry_confirm_days: int          # ex 7
    exit_confirm_days: int           # ex 14

    # défensif
    defensive_mode: str              # "AUTO"|"AGG"|"BND"|"CASH"|"SGOV"|"BIL"|"SHV"

    # plancher / CC scaling
    bdc_scale_in_riskoff: float      # ex 0.5 => 60%->30%
    cc_scale_in_riskoff: float       # ex 1.0 garder; 0.75 réduire de 25%


@dataclass
class BacktestConfig:
    start_eur: float
    annual_add_eur: float
    years: int
    rebalance: str
    withdraw_income: bool
    out_dir: str
    fourth_bdc: str
    fees: FeesConfig
    benchmarks: Dict[str, bool]
    riskoff: RiskOffConfig


def build_weights(fourth_bdc: str) -> Dict[str, float]:
    w = dict(DEFAULT_WEIGHTS)
    if fourth_bdc != "OBDC":
        v = w.pop("OBDC")
        w[fourth_bdc] = v
    s = sum(w.values())
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Poids invalides (somme={s})")
    return w


def compute_monthly_table(equity_eur: pd.Series,
                          cashflow_eur: pd.Series,
                          riskoff_flag: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"equity_eur": equity_eur,
                       "cashflow_eur": cashflow_eur,
                       "riskoff": riskoff_flag}).copy()
    df["month"] = df.index.to_period("M")
    rows = []
    for m, g in df.groupby("month"):
        g = g.sort_index()
        start = float(g["equity_eur"].iloc[0])
        end = float(g["equity_eur"].iloc[-1])
        mret = (end / start - 1.0) if start > 0 else 0.0
        mdd = max_drawdown(g["equity_eur"])
        income = float(g["cashflow_eur"].sum())
        riskoff_pct = float(g["riskoff"].mean()) if len(g) else 0.0
        rows.append({
            "month": str(m),
            "value_start_eur": start,
            "value_end_eur": end,
            "monthly_return": mret,
            "monthly_max_dd": mdd,
            "income_withdrawn_eur": income,
            "riskoff_days_pct": riskoff_pct
        })
    out = pd.DataFrame(rows).set_index("month")
    return out


def choose_defensive_asset_preferred(mode: str) -> List[str]:
    """
    Retourne une liste de tickers à essayer (dans l'ordre).
    """
    m = (mode or "AUTO").upper().strip()
    if m in ("AGG", "BND", "CASH", "SGOV", "BIL", "SHV"):
        return [m]
    # AUTO: d'abord cash-like, sinon bonds, sinon cash
    return ["SGOV", "BIL", "SHV", "AGG", "BND", "CASH"]


def build_riskoff_raw_signal(prices: pd.DataFrame, dates: pd.DatetimeIndex, cfg: BacktestConfig) -> pd.Series:
    """
    Signal brut (sans hystérésis) : True/False daily.
    """
    r = cfg.riskoff
    if not r.enabled:
        return pd.Series(False, index=dates)

    sma_len = int(r.sma_len)
    vix_thr = float(r.vix_threshold)
    dd_thr = float(r.spy_drawdown_threshold)

    # Series
    hyg = prices[HYG_SYMBOL].reindex(dates).ffill()
    lqd = prices[LQD_SYMBOL].reindex(dates).ffill()
    spy = prices[SPY_SYMBOL].reindex(dates).ffill()
    vix = prices[VIX_SYMBOL].reindex(dates).ffill()

    conds = []

    if r.use_credit_signal:
        ratio = (hyg / lqd).replace([np.inf, -np.inf], np.nan).ffill()
        ratio_sma = sma(ratio, sma_len)
        credit_off = (ratio < ratio_sma).fillna(False)
        conds.append(credit_off)

    if r.use_spy_vix_signal:
        spy_sma = sma(spy, sma_len)
        spy_below = (spy < spy_sma)
        vix_high = (vix > vix_thr)

        peak = spy.cummax()
        dd = spy / peak - 1.0
        dd_bad = (dd <= dd_thr)

        stress_off = ((spy_below & vix_high) | dd_bad).fillna(False)
        conds.append(stress_off)

    if not conds:
        return pd.Series(False, index=dates)

    out = conds[0].copy()
    for c in conds[1:]:
        out = out | c
    return out.reindex(dates).fillna(False)


def apply_hysteresis(raw: pd.Series, entry_days: int, exit_days: int) -> pd.Series:
    """
    Anti-whipsaw:
    - Passage à True si raw True pendant entry_days consécutifs
    - Retour à False si raw False pendant exit_days consécutifs
    """
    entry_days = max(1, int(entry_days))
    exit_days = max(1, int(exit_days))

    state = False
    true_streak = 0
    false_streak = 0
    out = []

    for v in raw.astype(bool).values:
        if v:
            true_streak += 1
            false_streak = 0
        else:
            false_streak += 1
            true_streak = 0

        if not state:
            if true_streak >= entry_days:
                state = True
                false_streak = 0  # reset
        else:
            if false_streak >= exit_days:
                state = False
                true_streak = 0  # reset

        out.append(state)

    return pd.Series(out, index=raw.index)


def dynamic_targets(total_usd: float,
                    base_weights: Dict[str, float],
                    tickers: List[str],
                    riskoff: bool,
                    defensive_asset: str,
                    bdc_scale: float,
                    cc_scale: float) -> Tuple[Dict[str, float], float]:
    """
    En risk-off:
    - BDC weights * bdc_scale (ex 0.5 => 60%->30%)
    - Covered call weights * cc_scale (ex 1.0 garder; 0.75 réduire)
    - le "freed weight" va au défensif (ticker ou cash)
    """
    bdc_set = {"MAIN", "ARCC", "BXSL", "OBDC", "CSWC"}
    cc_set = {"JEPI", "JEPQ", "DIVO", "BST"}

    if not riskoff:
        targets = {t: total_usd * base_weights.get(t, 0.0) for t in tickers}
        return targets, 0.0

    bdc_scale = float(bdc_scale)
    cc_scale = float(cc_scale)
    bdc_scale = max(0.0, min(1.0, bdc_scale))
    cc_scale = max(0.0, min(1.0, cc_scale))

    targets = {t: 0.0 for t in tickers}
    freed_w = 0.0

    for t in tickers:
        w = base_weights.get(t, 0.0)
        if t in bdc_set:
            w2 = w * bdc_scale
            freed_w += (w - w2)
            targets[t] = total_usd * w2
        elif t in cc_set:
            w2 = w * cc_scale
            freed_w += (w - w2)
            targets[t] = total_usd * w2
        else:
            # Défensif (si présent dans tickers avec base weight 0) ou autre
            targets[t] = total_usd * w

    # Tout ce qui est libéré va au défensif
    if defensive_asset == "CASH":
        return targets, total_usd * freed_w

    if defensive_asset not in targets:
        # fallback en cash si le ticker n'est pas investissable
        return targets, total_usd * freed_w

    targets[defensive_asset] += total_usd * freed_w
    return targets, 0.0


def run_backtest(cfg: BacktestConfig, log_fn) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_weights = build_weights(cfg.fourth_bdc)

    # tickers investissables = portefeuille + (éventuellement) défensif ETF si sélectionné
    tickers = sorted(base_weights.keys())

    # Déterminer le défensif souhaité (liste à essayer si AUTO)
    defensive_try = choose_defensive_asset_preferred(cfg.riskoff.defensive_mode) if cfg.riskoff.enabled else ["CASH"]
    # On pré-charge ces tickers si défensif ETF (pas CASH)
    # (en AUTO on va essayer dans l'ordre, donc on les ajoute tous aux downloads)
    defensive_etfs = [d for d in defensive_try if d != "CASH"]

    # Si riskoff activé: on autorise le défensif à être un ticker investissable (AGG/BND/SGOV/BIL/SHV)
    if cfg.riskoff.enabled:
        for d in defensive_etfs:
            if d not in tickers:
                tickers.append(d)
                base_weights[d] = 0.0
    tickers = sorted(set(tickers))

    # Benchmarks pour reporting (indépendants)
    bench_syms = [s for s, enabled in cfg.benchmarks.items() if enabled]

    # Symbols obligatoires pour signaux
    signal_syms = [HYG_SYMBOL, LQD_SYMBOL, SPY_SYMBOL, VIX_SYMBOL] if cfg.riskoff.enabled else []
    all_syms = sorted(set(tickers + [FX_SYMBOL] + bench_syms + signal_syms))

    end_dt = datetime.today().date()
    start_dt = (pd.Timestamp(end_dt) - pd.DateOffset(years=cfg.years) - pd.DateOffset(days=15)).date()
    start_str = start_dt.isoformat()
    end_str = (pd.Timestamp(end_dt) + pd.DateOffset(days=2)).date().isoformat()

    log_fn("Téléchargement Yahoo...")
    frames = {}
    for sym in all_syms:
        df = fetch_yahoo_daily(sym, start_str, end_str)
        frames[sym] = df.rename(columns={"close": sym})
        log_fn(f"  {sym:8s}: {len(df):4d} jours (de {df.index.min().date()} à {df.index.max().date()})")

    prices = pd.concat([frames[s] for s in all_syms], axis=1).sort_index()
    prices = prices.ffill().dropna()
    prices.index = pd.to_datetime(prices.index.date)
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()

    fx = prices[FX_SYMBOL].copy()
    px = prices[tickers].copy()
    bench_px = prices[bench_syms].copy() if bench_syms else pd.DataFrame(index=prices.index)

    dates = px.index
    start_trade_date = dates[0]
    last_date = dates[-1]

    # Rebal + apports
    rebal_dates = month_start_dates(dates)
    add_dates = year_anniversary_dates(start_trade_date, dates, cfg.years)
    event_dates = sorted(set(rebal_dates).union(set(add_dates)))
    event_dates = [d for d in event_dates if (d >= start_trade_date and d <= last_date)]

    # --- Risk-off: signal brut + hystérésis
    if cfg.riskoff.enabled:
        raw = build_riskoff_raw_signal(prices, dates, cfg)
        riskoff_daily = apply_hysteresis(raw, cfg.riskoff.entry_confirm_days, cfg.riskoff.exit_confirm_days)
    else:
        riskoff_daily = pd.Series(False, index=dates)

    # Choisir le défensif effectif (en AUTO: premier disponible avec data)
    def pick_defensive_asset() -> str:
        if not cfg.riskoff.enabled:
            return "CASH"
        for d in choose_defensive_asset_preferred(cfg.riskoff.defensive_mode):
            if d == "CASH":
                return "CASH"
            if d in prices.columns and prices[d].notna().any():
                return d
        return "CASH"

    defensive_asset = pick_defensive_asset()

    if cfg.riskoff.enabled:
        log_fn("")
        log_fn("MODE RISK-OFF: ACTIVÉ")
        log_fn(f" - Signal crédit (HYG/LQD<SMA): {'OUI' if cfg.riskoff.use_credit_signal else 'NON'}")
        log_fn(f" - Signal stress (SPY/VIX/DD): {'OUI' if cfg.riskoff.use_spy_vix_signal else 'NON'}")
        log_fn(f" - SMA len                   : {cfg.riskoff.sma_len}")
        log_fn(f" - VIX threshold             : {cfg.riskoff.vix_threshold}")
        log_fn(f" - SPY DD threshold          : {cfg.riskoff.spy_drawdown_threshold}")
        log_fn(f" - Anti-whipsaw entrée       : {cfg.riskoff.entry_confirm_days} j")
        log_fn(f" - Anti-whipsaw sortie       : {cfg.riskoff.exit_confirm_days} j")
        log_fn(f" - Défensif                  : {defensive_asset}")
        log_fn(f" - BDC scale risk-off        : {cfg.riskoff.bdc_scale_in_riskoff:.2f}")
        log_fn(f" - CC  scale risk-off        : {cfg.riskoff.cc_scale_in_riskoff:.2f}")
        log_fn("")

    # Portefeuille
    cash_usd = 0.0
    shares = {t: 0.0 for t in tickers}
    trades = []
    cashflow_eur_daily = pd.Series(0.0, index=dates)

    def portfolio_value_usd(dt: pd.Timestamp) -> float:
        return float(sum(shares[t] * px.loc[dt, t] for t in tickers) + cash_usd)

    def record_trade(dt, ticker, side, qty, price, note="", fee_usd=0.0, slip_usd=0.0):
        trades.append({
            "date": dt,
            "ticker": ticker,
            "side": side,
            "qty": float(qty),
            "price_usd": float(price),
            "notional_usd": float(qty) * float(price),
            "fee_usd": float(fee_usd),
            "slippage_usd": float(slip_usd),
            "note": note
        })

    def apply_tx_costs(notional_usd: float) -> Tuple[float, float]:
        fee = abs(notional_usd) * cfg.fees.tx_fee_pct
        slip = abs(notional_usd) * cfg.fees.slippage_pct
        return fee, slip

    def fx_convert_eur_to_usd(dt: pd.Timestamp, eur_amount: float) -> float:
        eurusd = float(fx.loc[dt])
        gross = eur_amount * eurusd
        cost = gross * cfg.fees.fx_fee_pct
        return gross - cost

    # Funding initial
    cash_usd += fx_convert_eur_to_usd(start_trade_date, cfg.start_eur)
    trades.append({
        "date": start_trade_date, "ticker": "CASH", "side": "ADD_EUR_INIT",
        "qty": float(cfg.start_eur), "price_usd": float(fx.loc[start_trade_date]),
        "notional_usd": float(cfg.start_eur) * float(fx.loc[start_trade_date]),
        "fee_usd": float(cfg.start_eur) * float(fx.loc[start_trade_date]) * cfg.fees.fx_fee_pct,
        "slippage_usd": 0.0,
        "note": "initial_funding"
    })

    def rebalance(dt: pd.Timestamp, note="rebalance"):
        nonlocal cash_usd, defensive_asset

        # actualiser défensif (au cas où AUTO et data change, mais ici stable)
        defensive_asset = pick_defensive_asset()

        total = portfolio_value_usd(dt)
        ro = bool(riskoff_daily.loc[dt]) if cfg.riskoff.enabled else False

        targets, target_cash = dynamic_targets(
            total_usd=total,
            base_weights=base_weights,
            tickers=tickers,
            riskoff=ro,
            defensive_asset=defensive_asset,
            bdc_scale=cfg.riskoff.bdc_scale_in_riskoff if cfg.riskoff.enabled else 1.0,
            cc_scale=cfg.riskoff.cc_scale_in_riskoff if cfg.riskoff.enabled else 1.0
        )

        # 1) Sell overweight
        for t in tickers:
            price = float(px.loc[dt, t])
            cur_val = shares[t] * price
            tgt_val = targets.get(t, 0.0)
            diff = cur_val - tgt_val
            if diff > 1e-6:
                qty = diff / price
                notional = qty * price
                fee, slip = apply_tx_costs(notional)
                shares[t] -= qty
                cash_usd += notional - fee - slip
                record_trade(dt, t, "SELL", qty, price, note + (":RISKOFF" if ro else ""), fee_usd=fee, slip_usd=slip)

        # 2) Buy underweight (en gardant cash cible si mode CASH)
        def cash_available_for_buys() -> float:
            if target_cash <= 0:
                return cash_usd
            return max(0.0, cash_usd - target_cash)

        for t in tickers:
            price = float(px.loc[dt, t])
            cur_val = shares[t] * price
            tgt_val = targets.get(t, 0.0)
            need = tgt_val - cur_val
            if need > 1e-6 and cash_available_for_buys() > 1e-6:
                buy_usd = min(need, cash_available_for_buys())
                qty = buy_usd / price
                notional = qty * price
                fee, slip = apply_tx_costs(notional)
                total_cost = notional + fee + slip

                avail = cash_available_for_buys()
                if total_cost > avail:
                    qty = avail / (price * (1 + cfg.fees.tx_fee_pct + cfg.fees.slippage_pct))
                    notional = qty * price
                    fee, slip = apply_tx_costs(notional)
                    total_cost = notional + fee + slip

                if qty > 1e-8 and total_cost <= cash_available_for_buys() + 1e-6:
                    shares[t] += qty
                    cash_usd -= total_cost
                    record_trade(dt, t, "BUY", qty, price, note + (":RISKOFF" if ro else ""), fee_usd=fee, slip_usd=slip)

        # Log risk flag
        if cfg.riskoff.enabled:
            trades.append({
                "date": dt, "ticker": "RISK", "side": "FLAG",
                "qty": 1.0 if ro else 0.0,
                "price_usd": 0.0,
                "notional_usd": 0.0,
                "fee_usd": 0.0,
                "slippage_usd": 0.0,
                "note": f"riskoff={ro};defensive={defensive_asset};target_cash={target_cash:,.2f}"
            })

    # allocation initiale
    rebalance(start_trade_date, note="initial_alloc")

    equity_rows = []
    prev_total_usd = portfolio_value_usd(start_trade_date)

    # Simulation
    for i, dt in enumerate(dates):
        # Apport annuel
        if dt in add_dates and cfg.annual_add_eur > 0:
            usd_added = fx_convert_eur_to_usd(dt, cfg.annual_add_eur)
            cash_usd += usd_added
            trades.append({
                "date": dt, "ticker": "CASH", "side": "ADD_EUR",
                "qty": float(cfg.annual_add_eur),
                "price_usd": float(fx.loc[dt]),
                "notional_usd": float(cfg.annual_add_eur) * float(fx.loc[dt]),
                "fee_usd": float(cfg.annual_add_eur) * float(fx.loc[dt]) * cfg.fees.fx_fee_pct,
                "slippage_usd": 0.0,
                "note": "annual_contribution"
            })

        # Rebalance mensuel
        if dt in event_dates:
            rebalance(dt, note="monthly_rebalance")

        # Mark-to-market
        total_usd = portfolio_value_usd(dt)
        eurusd = float(fx.loc[dt])
        total_eur = total_usd / eurusd if eurusd > 0 else np.nan

        # Cashflow simulé (optionnel)
        if cfg.withdraw_income and i > 0:
            daily_ret = (total_usd / prev_total_usd - 1.0) if prev_total_usd > 0 else 0.0
            income_usd = max(0.0, prev_total_usd * daily_ret)
            if income_usd > 0 and cash_usd > 0:
                wdraw = min(income_usd, cash_usd)
                cash_usd -= wdraw
                cashflow_eur_daily.loc[dt] = wdraw / eurusd if eurusd > 0 else 0.0
                trades.append({
                    "date": dt, "ticker": "CASH", "side": "WITHDRAW_INCOME",
                    "qty": float(wdraw),
                    "price_usd": 1.0,
                    "notional_usd": float(wdraw),
                    "fee_usd": 0.0,
                    "slippage_usd": 0.0,
                    "note": "income_withdrawal_simulated"
                })

        prev_total_usd = total_usd
        ro_flag = bool(riskoff_daily.loc[dt]) if cfg.riskoff.enabled else False

        row = {
            "date": dt,
            "portfolio_usd": float(total_usd),
            "eurusd": float(eurusd),
            "portfolio_eur": float(total_eur),
            "cash_usd": float(cash_usd),
            "riskoff": int(ro_flag),
        }
        for t in tickers:
            row[f"{t}_shares"] = float(shares[t])
            row[f"{t}_px"] = float(px.loc[dt, t])
            row[f"{t}_val_usd"] = float(shares[t] * px.loc[dt, t])

        if bench_syms:
            for b in bench_syms:
                row[f"BM_{b}_px"] = float(bench_px.loc[dt, b])

        equity_rows.append(row)

    equity = pd.DataFrame(equity_rows).set_index("date")
    trades_df = pd.DataFrame(trades)

    # Benchmarks: courbes normalisées EUR (indice base 100)
    bm_curves = {}
    if bench_syms:
        base_dt = equity.index[0]
        for b in bench_syms:
            b_eur = (bench_px[b] / fx).reindex(equity.index).ffill()
            b0 = float(b_eur.loc[base_dt])
            bm_curves[b] = 100.0 * (b_eur / b0)

    # Monthly table + bench retours mensuels
    monthly = compute_monthly_table(equity["portfolio_eur"], cashflow_eur_daily, equity["riskoff"])
    if bm_curves:
        for b, s in bm_curves.items():
            tmp = pd.DataFrame({"v": s})
            tmp["month"] = tmp.index.to_period("M")
            m_end = tmp.groupby("month")["v"].last()
            m_start = tmp.groupby("month")["v"].first()
            m_ret = (m_end / m_start - 1.0).rename(f"bm_{b}_monthly_return")
            monthly = monthly.join(m_ret, how="left")

    return equity, trades_df, monthly


# -----------------------------
# Verbose reporting
# -----------------------------
def verbose_report(equity: pd.DataFrame, trades: pd.DataFrame, monthly: pd.DataFrame,
                   cfg: BacktestConfig, log_fn):
    s = equity["portfolio_eur"]
    end = float(s.iloc[-1])
    dd = max_drawdown(s)
    cg = cagr(s)
    vol = vol_annual(s)
    sh = sharpe0(s)

    n_add = int((trades["side"] == "ADD_EUR").sum()) if not trades.empty else 0
    injected = cfg.start_eur + cfg.annual_add_eur * n_add

    log_fn("")
    log_fn("="*78)
    log_fn("RÉSULTATS — PORTFEUILLE (EUR)")
    log_fn("="*78)
    log_fn(f"Période réelle (données) : {equity.index.min().date()} -> {equity.index.max().date()} "
           f"({len(equity)} séances)")
    log_fn(f"Capital initial          : {money(cfg.start_eur)}")
    log_fn(f"Apport annuel            : {money(cfg.annual_add_eur)}  | apports réalisés = {n_add}  "
           f"| injecté total ≈ {money(injected)}")
    log_fn(f"Valeur finale            : {money(end)}")
    log_fn(f"Gain vs injecté (≈)       : {money(end - injected)}")
    log_fn(f"CAGR (sur equity)         : {as_pct(cg)}")
    log_fn(f"Volatilité annualisée     : {as_pct(vol)}")
    log_fn(f"Sharpe (rf=0)             : {sh:,.2f}")
    log_fn(f"Max Drawdown              : {as_pct(dd)}")
    log_fn(f"Mode retrait revenus      : {'OUI' if cfg.withdraw_income else 'NON'}")

    if "riskoff" in equity.columns and cfg.riskoff.enabled:
        ro_pct = float(equity["riskoff"].mean())
        log_fn(f"Risk-off (jours)          : {ro_pct*100:,.1f}%")

    log_fn("")
    log_fn("-"*78)
    log_fn("FRAIS")
    log_fn("-"*78)
    log_fn(f"FX fee                    : {as_pct(cfg.fees.fx_fee_pct)}")
    log_fn(f"Transaction fee           : {as_pct(cfg.fees.tx_fee_pct)}")
    log_fn(f"Slippage                  : {as_pct(cfg.fees.slippage_pct)}")

    if not trades.empty:
        total_fees = float(trades.get("fee_usd", pd.Series(dtype=float)).fillna(0).sum())
        total_slip = float(trades.get("slippage_usd", pd.Series(dtype=float)).fillna(0).sum())
        log_fn(f"Total fees USD (simulé)   : {total_fees:,.2f} USD")
        log_fn(f"Total slippage USD        : {total_slip:,.2f} USD")

    if not monthly.empty:
        best = monthly["monthly_return"].idxmax()
        worst = monthly["monthly_return"].idxmin()
        log_fn("")
        log_fn("-"*78)
        log_fn("TABLEAU MENSUEL (aperçu)")
        log_fn("-"*78)
        log_fn(f"Meilleur mois : {best}  | {as_pct(float(monthly.loc[best,'monthly_return']))} "
               f"| DD mois {as_pct(float(monthly.loc[best,'monthly_max_dd']))} "
               f"| revenu retiré {money(float(monthly.loc[best,'income_withdrawn_eur']))} "
               f"| riskoff {float(monthly.loc[best,'riskoff_days_pct'])*100:,.0f}%")
        log_fn(f"Pire mois     : {worst} | {as_pct(float(monthly.loc[worst,'monthly_return']))} "
               f"| DD mois {as_pct(float(monthly.loc[worst,'monthly_max_dd']))} "
               f"| revenu retiré {money(float(monthly.loc[worst,'income_withdrawn_eur']))} "
               f"| riskoff {float(monthly.loc[worst,'riskoff_days_pct'])*100:,.0f}%")

        log_fn("")
        log_fn("Derniers mois:")
        tail = monthly.tail(6).copy()
        for idx, r in tail.iterrows():
            log_fn(f" {idx} | end {money(r['value_end_eur'])} | ret {as_pct(float(r['monthly_return']))}"
                   f" | dd {as_pct(float(r['monthly_max_dd']))}"
                   f" | income {money(float(r['income_withdrawn_eur']))}"
                   f" | riskoff {float(r['riskoff_days_pct'])*100:,.0f}%")

    log_fn("")
    log_fn("="*78)
    log_fn("FIN DU RAPPORT")
    log_fn("="*78)
    log_fn("")


# -----------------------------
# GUI
# -----------------------------
class BacktestGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Backtest — BDC + Covered Call (Yahoo sans yfinance)")
        self.root.geometry("1120x900")
        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill="both", expand=True)

        top = ttk.LabelFrame(frm, text="Paramètres", padding=10)
        top.pack(fill="x")

        self.var_start = tk.StringVar(value="400000")
        self.var_add = tk.StringVar(value="80000")
        self.var_years = tk.StringVar(value="10")
        self.var_fourth = tk.StringVar(value="OBDC")

        self.var_fx_fee = tk.StringVar(value="0.15")  # %
        self.var_tx_fee = tk.StringVar(value="0.02")  # %
        self.var_slip = tk.StringVar(value="0.01")    # %

        self.var_withdraw = tk.BooleanVar(value=False)

        # Benchmarks
        self.var_bm_spy = tk.BooleanVar(value=True)
        self.var_bm_vnq = tk.BooleanVar(value=True)
        self.var_bm_bnd = tk.BooleanVar(value=True)
        self.var_bm_agg = tk.BooleanVar(value=False)

        # Risk-off UI
        self.var_riskon = tk.BooleanVar(value=False)
        self.var_risk_credit = tk.BooleanVar(value=True)
        self.var_risk_stress = tk.BooleanVar(value=True)
        self.var_risk_sma = tk.StringVar(value="200")
        self.var_risk_vix = tk.StringVar(value="20")
        self.var_risk_dd = tk.StringVar(value="-0.10")

        # anti-whipsaw
        self.var_risk_entry = tk.StringVar(value="7")   # jours consécutifs
        self.var_risk_exit = tk.StringVar(value="14")   # jours consécutifs

        # défensif
        self.var_risk_def = tk.StringVar(value="AUTO")  # AUTO/AGG/BND/CASH/SGOV/BIL/SHV

        # plancher / scaling
        self.var_risk_bdc_scale = tk.StringVar(value="0.50")  # 0.50 => 60%->30%
        self.var_risk_cc_scale = tk.StringVar(value="1.00")   # 1.00 garder CC; 0.75 réduire

        row = 0
        ttk.Label(top, text="Capital initial (EUR)").grid(row=row, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.var_start, width=14).grid(row=row, column=1, sticky="w", padx=6)
        ttk.Label(top, text="Apport annuel (EUR)").grid(row=row, column=2, sticky="w", padx=(18, 0))
        ttk.Entry(top, textvariable=self.var_add, width=14).grid(row=row, column=3, sticky="w", padx=6)
        ttk.Label(top, text="Horizon (années)").grid(row=row, column=4, sticky="w", padx=(18, 0))
        ttk.Entry(top, textvariable=self.var_years, width=8).grid(row=row, column=5, sticky="w", padx=6)

        row += 1
        ttk.Label(top, text="4e BDC").grid(row=row, column=0, sticky="w", pady=(8, 0))
        cb = ttk.Combobox(top, textvariable=self.var_fourth, values=["OBDC", "CSWC"], width=10, state="readonly")
        cb.grid(row=row, column=1, sticky="w", padx=6, pady=(8, 0))

        ttk.Checkbutton(top, text="Retirer les revenus (cashflow mensuel simulé)",
                        variable=self.var_withdraw).grid(row=row, column=2, columnspan=4, sticky="w", pady=(8, 0))

        row += 1
        fees = ttk.LabelFrame(top, text="Frais (éditables) — en %", padding=8)
        fees.grid(row=row, column=0, columnspan=6, sticky="we", pady=(10, 0))
        fees.columnconfigure(5, weight=1)

        ttk.Label(fees, text="FX fee (EUR↔USD)").grid(row=0, column=0, sticky="w")
        ttk.Entry(fees, textvariable=self.var_fx_fee, width=10).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(fees, text="Transaction fee").grid(row=0, column=2, sticky="w", padx=(18, 0))
        ttk.Entry(fees, textvariable=self.var_tx_fee, width=10).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(fees, text="Slippage").grid(row=0, column=4, sticky="w", padx=(18, 0))
        ttk.Entry(fees, textvariable=self.var_slip, width=10).grid(row=0, column=5, sticky="w", padx=6)

        row += 1
        bm = ttk.LabelFrame(top, text="Benchmarks", padding=8)
        bm.grid(row=row, column=0, columnspan=6, sticky="we", pady=(10, 0))
        ttk.Checkbutton(bm, text="SPY (actions US)", variable=self.var_bm_spy).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(bm, text="VNQ (foncières/REITs US)", variable=self.var_bm_vnq).grid(row=0, column=1, sticky="w", padx=12)
        ttk.Checkbutton(bm, text="BND (obligations US)", variable=self.var_bm_bnd).grid(row=0, column=2, sticky="w", padx=12)
        ttk.Checkbutton(bm, text="AGG (obligations US)", variable=self.var_bm_agg).grid(row=0, column=3, sticky="w", padx=12)

        # Risk-off frame
        row += 1
        ro = ttk.LabelFrame(top, text="Risk-Off (optionnel) — réduction BDC + anti-whipsaw + défensif cash-like", padding=8)
        ro.grid(row=row, column=0, columnspan=6, sticky="we", pady=(10, 0))

        ttk.Checkbutton(ro, text="Activer Risk-Off", variable=self.var_riskon).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(ro, text="Signal crédit (HYG/LQD < SMA)", variable=self.var_risk_credit).grid(row=0, column=1, sticky="w", padx=12)
        ttk.Checkbutton(ro, text="Signal stress (SPY/VIX/DD)", variable=self.var_risk_stress).grid(row=0, column=2, sticky="w", padx=12)

        ttk.Label(ro, text="SMA len").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(ro, textvariable=self.var_risk_sma, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(ro, text="VIX seuil").grid(row=1, column=2, sticky="w", padx=(18, 0), pady=(6, 0))
        ttk.Entry(ro, textvariable=self.var_risk_vix, width=10).grid(row=1, column=3, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(ro, text="SPY DD seuil (ex -0.10)").grid(row=1, column=4, sticky="w", padx=(18, 0), pady=(6, 0))
        ttk.Entry(ro, textvariable=self.var_risk_dd, width=10).grid(row=1, column=5, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(ro, text="Entrée (jours vrai)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(ro, textvariable=self.var_risk_entry, width=10).grid(row=2, column=1, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(ro, text="Sortie (jours faux)").grid(row=2, column=2, sticky="w", padx=(18, 0), pady=(6, 0))
        ttk.Entry(ro, textvariable=self.var_risk_exit, width=10).grid(row=2, column=3, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(ro, text="Défensif (AUTO/CASH/SGOV/BIL/SHV/AGG/BND)").grid(row=3, column=0, sticky="w", pady=(6, 0))
        cb2 = ttk.Combobox(ro, textvariable=self.var_risk_def,
                           values=["AUTO", "CASH", "SGOV", "BIL", "SHV", "AGG", "BND"],
                           width=10, state="readonly")
        cb2.grid(row=3, column=1, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(ro, text="BDC scale (risk-off)").grid(row=3, column=2, sticky="w", padx=(18, 0), pady=(6, 0))
        ttk.Entry(ro, textvariable=self.var_risk_bdc_scale, width=10).grid(row=3, column=3, sticky="w", padx=6, pady=(6, 0))

        ttk.Label(ro, text="CC scale (risk-off)").grid(row=3, column=4, sticky="w", padx=(18, 0), pady=(6, 0))
        ttk.Entry(ro, textvariable=self.var_risk_cc_scale, width=10).grid(row=3, column=5, sticky="w", padx=6, pady=(6, 0))

        # Buttons
        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=10)
        self.btn_run = ttk.Button(btns, text="▶ Lancer le backtest", command=self.on_run)
        self.btn_run.pack(side="left")
        self.lbl_status = ttk.Label(btns, text="Prêt.")
        self.lbl_status.pack(side="left", padx=12)

        # Output text
        out = ttk.LabelFrame(frm, text="Sortie / Rapport (également envoyé au terminal)", padding=10)
        out.pack(fill="both", expand=True)
        self.txt = scrolledtext.ScrolledText(out, height=30, wrap="word")
        self.txt.pack(fill="both", expand=True)

    def log(self, msg: str):
        print(msg)
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.root.update_idletasks()

    def _read_float(self, var: tk.StringVar, name: str) -> float:
        try:
            return float(var.get().strip().replace(",", "."))
        except Exception:
            raise ValueError(f"Champ invalide: {name}")

    def on_run(self):
        th = threading.Thread(target=self._run_thread, daemon=True)
        th.start()

    def _run_thread(self):
        try:
            self.btn_run.config(state="disabled")
            self.txt.delete("1.0", "end")
            self.lbl_status.config(text="Téléchargement & backtest...")

            start_eur = self._read_float(self.var_start, "Capital initial")
            add_eur = self._read_float(self.var_add, "Apport annuel")
            years = int(self._read_float(self.var_years, "Horizon (années)"))

            fx_fee = self._read_float(self.var_fx_fee, "FX fee") / 100.0
            tx_fee = self._read_float(self.var_tx_fee, "Transaction fee") / 100.0
            slip = self._read_float(self.var_slip, "Slippage") / 100.0

            fourth = self.var_fourth.get().strip()
            withdraw = bool(self.var_withdraw.get())

            benchmarks = {
                "SPY": bool(self.var_bm_spy.get()),
                "VNQ": bool(self.var_bm_vnq.get()),
                "BND": bool(self.var_bm_bnd.get()),
                "AGG": bool(self.var_bm_agg.get()),
            }

            # Risk-off params
            risk_enabled = bool(self.var_riskon.get())
            risk_credit = bool(self.var_risk_credit.get())
            risk_stress = bool(self.var_risk_stress.get())
            risk_sma = int(self._read_float(self.var_risk_sma, "Risk SMA len"))
            risk_vix = float(self._read_float(self.var_risk_vix, "VIX seuil"))
            risk_dd = float(self._read_float(self.var_risk_dd, "SPY DD seuil"))

            entry_days = int(self._read_float(self.var_risk_entry, "Entrée (jours vrai)"))
            exit_days = int(self._read_float(self.var_risk_exit, "Sortie (jours faux)"))

            risk_def = self.var_risk_def.get().strip().upper()

            bdc_scale = float(self._read_float(self.var_risk_bdc_scale, "BDC scale (risk-off)"))
            cc_scale = float(self._read_float(self.var_risk_cc_scale, "CC scale (risk-off)"))

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(os.path.expanduser("~"), "backtest_bdc_cc_gui", ts)
            os.makedirs(out_dir, exist_ok=True)

            cfg = BacktestConfig(
                start_eur=start_eur,
                annual_add_eur=add_eur,
                years=years,
                rebalance="monthly",
                withdraw_income=withdraw,
                out_dir=out_dir,
                fourth_bdc=fourth,
                fees=FeesConfig(fx_fee_pct=fx_fee, tx_fee_pct=tx_fee, slippage_pct=slip),
                benchmarks=benchmarks,
                riskoff=RiskOffConfig(
                    enabled=risk_enabled,
                    use_credit_signal=risk_credit,
                    use_spy_vix_signal=risk_stress,
                    sma_len=risk_sma,
                    vix_threshold=risk_vix,
                    spy_drawdown_threshold=risk_dd,
                    entry_confirm_days=entry_days,
                    exit_confirm_days=exit_days,
                    defensive_mode=risk_def,
                    bdc_scale_in_riskoff=bdc_scale,
                    cc_scale_in_riskoff=cc_scale
                )
            )

            self.log("=== PARAMÈTRES ===")
            self.log(f"Capital initial : {money(cfg.start_eur)}")
            self.log(f"Apport annuel   : {money(cfg.annual_add_eur)}")
            self.log(f"Horizon         : {cfg.years} ans")
            self.log(f"4e BDC          : {cfg.fourth_bdc}")
            self.log(f"Retrait revenus : {'OUI' if cfg.withdraw_income else 'NON'}")
            self.log(f"Frais FX        : {as_pct(cfg.fees.fx_fee_pct)}")
            self.log(f"Frais TX        : {as_pct(cfg.fees.tx_fee_pct)}")
            self.log(f"Slippage        : {as_pct(cfg.fees.slippage_pct)}")
            self.log(f"Benchmarks      : {', '.join([k for k, v in benchmarks.items() if v]) or 'aucun'}")
            self.log(f"Risk-Off        : {'OUI' if cfg.riskoff.enabled else 'NON'}")
            if cfg.riskoff.enabled:
                self.log(f" - crédit HYG/LQD<SMA : {'OUI' if cfg.riskoff.use_credit_signal else 'NON'}")
                self.log(f" - stress SPY/VIX/DD  : {'OUI' if cfg.riskoff.use_spy_vix_signal else 'NON'}")
                self.log(f" - SMA len            : {cfg.riskoff.sma_len}")
                self.log(f" - VIX seuil          : {cfg.riskoff.vix_threshold}")
                self.log(f" - DD seuil           : {cfg.riskoff.spy_drawdown_threshold}")
                self.log(f" - Entrée (jours)     : {cfg.riskoff.entry_confirm_days}")
                self.log(f" - Sortie (jours)     : {cfg.riskoff.exit_confirm_days}")
                self.log(f" - Défensif           : {cfg.riskoff.defensive_mode}")
                self.log(f" - BDC scale          : {cfg.riskoff.bdc_scale_in_riskoff:.2f}")
                self.log(f" - CC scale           : {cfg.riskoff.cc_scale_in_riskoff:.2f}")
            self.log(f"Sortie          : {cfg.out_dir}")
            self.log("")

            equity, trades, monthly = run_backtest(cfg, self.log)

            equity_csv = os.path.join(cfg.out_dir, "equity_curve.csv")
            trades_csv = os.path.join(cfg.out_dir, "trades.csv")
            monthly_csv = os.path.join(cfg.out_dir, "monthly_report.csv")
            equity.to_csv(equity_csv, index=True)
            trades.to_csv(trades_csv, index=False)
            monthly.to_csv(monthly_csv, index=True)

            verbose_report(equity, trades, monthly, cfg, self.log)

            self.log("CSV sauvegardés :")
            self.log(f" - {equity_csv}")
            self.log(f" - {trades_csv}")
            self.log(f" - {monthly_csv}")

            self.lbl_status.config(text="Terminé.")
        except Exception as e:
            self.lbl_status.config(text="Erreur.")
            messagebox.showerror("Erreur", str(e))
            self.log(f"\n[ERREUR] {e}")
        finally:
            self.btn_run.config(state="normal")


def main():
    root = tk.Tk()
    app = BacktestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
