import json
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytz
import shioaji as sj

import shioaji_connector


# Fix for Windows console encoding issues with Traditional Chinese
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# ================ 
# different usage
# ================ 
def signal_enum():
    ma_base = shioaji_connector.api.kbars(
        shioaji_connector.api.Contracts.Futures.TXF.TXFR1,
        start=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
    )
    prev_open = ma_base.Open[-2]
    prev_close = ma_base.Close[-2]
    current_open = ma_base.Open[-1]
    top_10_volumes = sorted(ma_base.Volume[-270:], reverse=True)[:10]

    ma_270 = 0
    if len(ma_base.Close[-270:]) == 270:
        ma_270 = sum(ma_base.Close[-270:]) / 270

    _txf = shioaji_connector.api.snapshots([shioaji_connector.api.Contracts.Futures.TXF.TXFR1])[0]
    _txf_range = _txf.high - _txf.low

    if _txf.close > _txf.high - _txf_range * 0.15:
        signal = "Wait for A-reversal after new high"
    elif _txf.close < _txf.low + _txf_range * 0.15:
        signal = "Wait for V-rebound after new low"
    elif abs(_txf.close - ma_270) < _txf_range * 0.1:
        signal = "Hovering around MA"
    elif max([prev_open, prev_close, current_open]) < ma_270 < _txf.close:
        signal = "Price breaks above MA270"
    elif min([prev_open, prev_close, current_open]) > ma_270 > _txf.close:
        signal = "Price breaks below MA270"
    elif ma_base.Volume[-1] >= sum(top_10_volumes) / 10:
        signal = "Volume spike over top 10 average"
    else:
        signal = "No significant signal"
    log = {
        "t": datetime.now().strftime("%H:%M:%S"),
        "bounce": int(_txf.close - _txf.low),
        "pullback": int(_txf.high - _txf.close),
        "chg": int(_txf.change_price),
        "signal": signal,
    }
    return f"{log.get('t', '')} | Up: {log.get('bounce', 0)} | Down: {log.get('pullback', 0)} | Chg: {log.get('chg', 0):+} | {log.get('signal', '')}"


def _process_kbars_df(kbars_data: dict) -> pd.DataFrame:
    df = pd.DataFrame({**kbars_data})
    df.ts = pd.to_datetime(df.ts)
    df.set_index("ts", inplace=True)
    return df


# --- Core Functions ---
def kbars_resample(sid="2330", interval="15T", start_date=None):
    """Resamples K-bars for a given stock or future."""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    if len(sid) == 4:
        contract = shioaji_connector.api.Contracts.Stocks[sid]
    elif sid.upper() == "TXF":
        contract = shioaji_connector.api.Contracts.Futures.TXF.TXFR1
    else:
        print(f"Sid {sid} not supported.")
        return pd.DataFrame()

    kbars = shioaji_connector.api.kbars(contract, start=start_date)
    if not kbars.ts:
        return pd.DataFrame()

    df = _process_kbars_df(kbars)
    df_ret = df.resample(interval, closed="right", label="right").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    df_ret.dropna(inplace=True)
    return df_ret


def strike_atm(weekth="TX4", yyyymm="202509", strike_range=2, backwardation=0):
    """
    weekth and yyyymm should fetch global variables
    """
    close_price = shioaji_connector.api.kbars(shioaji_connector.api.Contracts.Futures["TXFR1"])
    base_strike = backwardation + int(close_price / 50) * 50

    contracts_strikes = []
    options_contract_group = getattr(shioaji_connector.api.Contracts.Options, weekth)

    for offset in range(-strike_range, strike_range):
        strike = str(base_strike + offset * 50)
        for cp in ["C", "P"]:
            contract = getattr(options_contract_group, f"{weekth}{yyyymm}{strike}{cp}")
            contracts_strikes.append(contract)

    min_abs_diff = float("inf")
    atm_strike_info = {}

    if not contracts_strikes:
        print("No contracts found for the given range.")
        return None, None, None

    snapshots = shioaji_connector.api.snapshots(contracts_strikes)
    snapshot_map = {s.code: s.close for s in snapshots}

    for i in range(0, len(contracts_strikes), 2):
        call_contract = contracts_strikes[i]
        put_contract = contracts_strikes[i + 1]

        call_close = snapshot_map.get(call_contract.code, 0)
        put_close = snapshot_map.get(put_contract.code, 0)

        current_abs_diff = abs(call_close - put_close)
        if current_abs_diff < min_abs_diff:
            min_abs_diff = current_abs_diff
            atm_strike_info = {
                "weekth": weekth,
                "yyyymm": yyyymm,
                "strike": call_contract.code[3:8],
            }
    return atm_strike_info.get("strike")

def determine_direction_by_area():
    """#! 學macd用面積表達強弱，先看前一個大面積慢慢貼近均線，等順勢吃第二個大面積
    2025-09-12 18:30:00   -370.000000
    2025-09-12 20:30:00     61.333333
    2025-09-12 20:45:00     -3.833333
    2025-09-12 21:30:00     36.500000
    2025-09-12 21:45:00     -7.833333
    """
    base = kbars_resample(
        sid="TXF",
        interval="15T",
        start_date=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
    )

    ma = base.Close.rolling(18).mean().iloc[-1]
    last_close = base.Close.iloc[-1]
    if last_close > ma:
        print("做多") if abs(last_close - ma) < 10 else print("不追多")
    if last_close < ma:
        print("做空") if abs(last_close - ma) < 10 else print("不追空")

    base["area"] = base.Close - ma
    base["sign"] = (base["area"] >= 0).astype(int)
    base["group"] = (base["sign"] != base["sign"].shift()).cumsum()
    result = base.groupby("group").agg({"Close": "last", "area": "sum"})
    result.index = base.groupby("group").apply(lambda x: x.index[-1])
    return result[["Close", "area"]]


# --- Contract Helper ---
def get_contract(code: str, yyyymm: str = None, strike: str = None, cp: str = None):
    """
    A helper function to get a contract object.
    - For Futures: get_contract("TXF", "202509")
    - For Options: get_contract("TX1", "202509", "24450", "C")
    """
    try:
        if strike and cp and yyyymm:  # Options
            weekth = code
            contract_name = f"{weekth}{yyyymm}{strike}{cp}"
            options_group = getattr(shioaji_connector.api.Contracts.Options, weekth)
            return getattr(options_group, contract_name)
        elif yyyymm:  # Futures
            future_code = f"{code.upper()}R1"
            future_group = getattr(shioaji_connector.api.Contracts.Futures, code.upper())
            return getattr(future_group, future_code)
        else:  # Stocks
            return getattr(shioaji_connector.api.Contracts.Stocks, code)
    except (AttributeError, KeyError) as e:
        print(f"Contract not found for: {code} {yyyymm} {strike} {cp}. Error: {e}")
        return None


def _get_price_data(contract):
    """Fetches and formats price data for a given contract."""
    if contract is None:
        return pd.DataFrame()
    kbars = shioaji_connector.api.kbars(contract)
    if not kbars.ts:
        return pd.DataFrame()
    df = _process_kbars_df(kbars)
    return (
        df[["Close"]]
        .tail(300)
        .assign(
            time=lambda x: [
                datetime.fromtimestamp(ts.timestamp(), tz=pytz.UTC).strftime(
                    "%m-%d %H:%M:%S"
                )
                for ts in x.index
            ]
        )
        .set_index("time")
    )


def parity_premium_trend(weekth, yyyymm, strike):
    def get_price_data(contract):
        return (
            pd.DataFrame({**shioaji_connector.api.kbars(contract)})[["ts", "Close"]]
            .tail(300)
            .assign(
                time=lambda x: [
                    datetime.fromtimestamp(ts / 1e9, tz=pytz.UTC).strftime(
                        "%m-%d %H:%M:%S"
                    )
                    for ts in x["ts"]
                ]
            )
            .drop(columns=["ts"])
            .set_index("time")
        )

    df = get_price_data(shioaji_connector.api.Contracts.Futures["TXFR1"])
    options_group = getattr(shioaji_connector.api.Contracts.Options, weekth)
    for i in [int(strike) - 100, strike, int(strike) + 100]:
        call = getattr(options_group, f"{weekth}{yyyymm}{i}C")
        put = getattr(options_group, f"{weekth}{yyyymm}{i}P")
        df = df.join(
            [
                get_price_data(call).rename(columns={"Close": f"Close_call_{i}"}),
                get_price_data(put).rename(columns={"Close": f"Close_put_{i}"}),
            ]
        ).ffill()
        df[f"{i}_sum"] = df[f"Close_call_{i}"] + df[f"Close_put_{i}"]
    df.dropna(inplace=True)
    return df.iloc[-270:, [3, 6, 9]].reset_index()


def oversea_future(yyyymm="202509", symbols=None):
    """Fetches and normalizes price data for overseas futures against TXF."""
    if symbols is None:
        symbols = ["UNF", "UDF"]

    df = _get_price_data(shioaji_connector.api.Contracts.Futures.TXF.TXFR1)

    for symbol in symbols:
        try:
            contract = getattr(shioaji_connector.api.Contracts.Futures, symbol)[f"{symbol}{yyyymm}"]
            df = df.join(
                [_get_price_data(contract).rename(columns={"Close": contract.category})]
            ).ffill()
        except (AttributeError, KeyError):
            print(f"Oversea future contract not found for {symbol}{yyyymm}")
            continue

    numeric_cols = df.select_dtypes(include="number").columns
    normalized_df = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        first_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else 1
        if first_value == 0:
            first_value = 1  # Avoid division by zero
        normalized_df[col] = ((df[col] - first_value) / first_value) * 100

    return normalized_df


def strike_spread_sum(weekth, yyyymm, strike):
    """Calculates the bid-ask spread sum for a call and put option."""
    try:
        options_group = getattr(shioaji_connector.api.Contracts.Options, weekth)
        call = options_group[f"{weekth}{yyyymm}{strike}C"]
        put = options_group[f"{weekth}{yyyymm}{strike}P"]
    except (AttributeError, KeyError):
        return f"Contracts for {weekth} {yyyymm} {strike} not found."

    snapshots = shioaji_connector.api.snapshots([call, put])
    if len(snapshots) < 2:
        return "Could not fetch snapshot data for both contracts."

    snapshot_map = {s.code: s for s in snapshots}
    call_snap = snapshot_map.get(call.code)
    put_snap = snapshot_map.get(put.code)

    if not call_snap or not put_snap:
        return "Snapshot data missing for one or both contracts."

    call_spread = call_snap.sell_price - call_snap.buy_price
    put_spread = put_snap.sell_price - put_snap.buy_price

    print(
        f"{strike}, Call: {call_snap.buy_price} ⇄ {call_snap.sell_price}, Put: {put_snap.buy_price} ⇄ {put_snap.sell_price}"
    )
    return call_spread + put_spread


def rsi(plot=False):
    watchlist = [
        "1513", "2363", "2455", "3443", "3078", "2449", "8046", "6515",
        "3376", "6139", "2408", "3037", "3653", "3030", "3076", "8028",
        "2429", "6191", "4985", "3035", "3661", "6805", "6442", "1519",
        "4979",
    ]
    two_week_ago = (date.today() - timedelta(weeks=2)).strftime("%Y-%m-%d")

    def get_contract(stock_id):
        try:
            return getattr(shioaji_connector.api.Contracts.Stocks.TSE, f"TSE{stock_id}")
        except AttributeError:
            return getattr(shioaji_connector.api.Contracts.Stocks.OTC, f"OTC{stock_id}")

    series_list = [
        pd.Series(
            shioaji_connector.api.kbars(shioaji_connector.api.Contracts.Indexs.TSE.TSE001, start=two_week_ago)["Close"],
            name="TSE001",
        )
    ]
    series_list.extend(
        [
            pd.Series(
                shioaji_connector.api.kbars(get_contract(stock_id), start=two_week_ago)["Close"],
                name=stock_id,
            )
            for stock_id in watchlist
        ]
    )

    df = pd.concat(series_list, axis=1).ffill()
    normalized_df = df.apply(
        lambda col: ((col - col.dropna().iloc[0]) / col.dropna().iloc[0]) * 100
    )
    normalized_df = normalized_df.reindex(
        columns=normalized_df.iloc[-1].sort_values(ascending=False).index
    )

    if plot:
        import matplotlib.pyplot as plt

        normalized_df.plot()
        plt.legend(loc=3)
        plt.show()

    return normalized_df


# ================== 
# candle_stick chart
# ================== 
def ohlc_chart(interval: str):
    week_ago = (date.today() - timedelta(weeks=1)).strftime("%Y-%m-%d")
    kbars = shioaji_connector.api.kbars(shioaji_connector.api.Contracts.Futures.TXF.TXFR1, start=week_ago)
    if not kbars.ts:
        return pd.DataFrame()
    df = pd.DataFrame({**kbars})
    df.ts = pd.to_datetime(df.ts)
    df.set_index("ts", inplace=True)
    df_resample = df.resample(interval, closed="right", label="right").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    df_resample.dropna(inplace=True)
    return df_resample.tail(120)


def query_positions_unitshare():
    data = []
    for pos in shioaji_connector.api.list_positions(shioaji_connector.api.stock_account, unit=sj.constant.Unit.Share):
        data.append(
            {
                "code": pos.code,
                "direction": pos.direction.value,
                "quantity": pos.quantity,
                "price": pos.price,
                "last_price": pos.last_price,
                "pnl": f"{pos.pnl:.2f}",
            }
        )
    return data


if __name__ == "__main__":
    print("\n--- strike_atm ---")