import pandas as pd
import numpy as np
import sys, os, json, time, pytz
import shioaji as sj
from datetime import datetime, date, timedelta
from collections import defaultdict

# Fix for Windows console encoding issues with Traditional Chinese
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# --- API Initialization ---
api = None # Initialize api as None

def initialize_shioaji_api():
    global api
    if api is not None:
        print("Shioaji API already initialized.")
        return api
    try:
        with open(r'Sinopac.json', 'r', encoding='utf-8') as f:
            file = json.load(f)
        api = sj.Shioaji()
        accounts = api.login(file.get('API_Key'), file.get('Secret_Key'))
        ########### 只驗API有沒有效，不需試下單的話，這裡面註掉
        api.activate_ca(
            ca_path=r"Sinopac.pfx",
            ca_passwd=file.get('ca_passwd'),
            person_id=file.get('person_id')
        )
        ###########
        print("Shioaji API initialized successfully.")
        return api
    except FileNotFoundError:
        print("Error: Credentials file 'Sinopac.json' or 'Sinopac.pfx' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during API initialization: {e}")
        sys.exit(1)

# Initialize API when the module is imported
initialize_shioaji_api()


# ================
# different usage
# ================ 
def signal_enum():
    ma_base = api.kbars(api.Contracts.Futures.TXF.TXFR1, start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
    ma_前開 = ma_base.Open[-2]
    ma_前收 = ma_base.Close[-2]
    ma_今開 = ma_base.Open[-1]
    ma_前十大量 = sorted(ma_base.Volume[-270:], reverse=True)[:10]

    ma_270 = 0
    if len(ma_base.Close[-270:]) == 270:
        ma_270 = sum(ma_base.Close[-270:]) / 270

    _txf = api.snapshots([api.Contracts.Futures.TXF.TXFR1])[0]
    _txf_range = _txf.high - _txf.low

    if _txf.close > _txf.high - _txf_range * 0.15: signal = "等過高Ａ轉"
    elif _txf.close < _txf.low + _txf_range * 0.15: signal = "等破底Ｖ彈"
    elif abs(_txf.close - ma_270) < _txf_range * 0.1: signal = "徘徊在均線"
    elif max([ma_前開, ma_前收, ma_今開]) < ma_270 < _txf.close: signal = "站上兩百七"
    elif min([ma_前開, ma_前收, ma_今開]) > ma_270 > _txf.close: signal = "跌破兩百七"
    elif ma_base.Volume[-1] >= sum(ma_前十大量) / 10: signal = "爆前十大量"
    else: signal = "不值一提"
    log = {
        "t": datetime.now().strftime('%H:%M:%S'),
        "bounce": int(_txf.close - _txf.low),
        "pullback": int(_txf.high - _txf.close),
        "chg": int(_txf.change_price),
        "signal": signal,
    }
    return f"{log.get('t', '')} ﹍↗{log.get('bounce', 0)}  ﹉↘{log.get('pullback', 0)} {'▲' if (log.get('chg', 0)) >= 0 else '▼' }{(log.get('chg', 0))}　{log.get('signal', '')}"


def _process_kbars_df(kbars_data: dict) -> pd.DataFrame:
    df = pd.DataFrame({**kbars_data})
    df.ts = pd.to_datetime(df.ts)
    df.set_index('ts', inplace=True)
    return df


# --- Core Functions ---
def kbars_resample(sid="2330", interval="15T", start_date=None):
    """Resamples K-bars for a given stock or future."""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

    if len(sid) == 4:
        contract = api.Contracts.Stocks[sid]
    elif sid.upper() == 'TXF':
        contract = api.Contracts.Futures.TXF.TXFR1
    else:
        print(f"Sid {sid} not supported.")
        return pd.DataFrame()

    kbars = api.kbars(contract, start=start_date)
    if not kbars.ts:
        return pd.DataFrame()

    df = _process_kbars_df(kbars)
    df_ret = df.resample(interval, closed='right', label='right').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    df_ret.dropna(inplace=True)
    return df_ret


def strike_atm(weekth="TX4", yyyymm="202509", strike_range = 2, backwardation = 0):
    '''
    weekth and yyyymm should fetch global variables
    '''
    close_price = api.kbars(api.Contracts.Futures["TXFR1"]).Close[-1]
    base_strike = backwardation + int(close_price / 50) * 50

    contracts_strikes = []
    options_contract_group = getattr(api.Contracts.Options, weekth)

    for offset in range(-strike_range, strike_range):
        strike = str(base_strike + offset * 50)
        for cp in ["C", "P"]:
            contract = getattr(api.Contracts.Options, weekth)[f"{weekth}{yyyymm}{strike}{cp}"]
            contracts_strikes.append(contract)

    min_abs_diff = float('inf')
    atm_strike_info = {}

    if not contracts_strikes:
        print("No contracts found for the given range.")
        return None, None, None

    snapshots = api.snapshots(contracts_strikes)
    snapshot_map = {s.code: s.close for s in snapshots}

    for i in range(0, len(contracts_strikes), 2):
        call_contract = contracts_strikes[i]
        put_contract = contracts_strikes[i+1]

        call_close = snapshot_map.get(call_contract.code, 0)
        put_close = snapshot_map.get(put_contract.code, 0)

        current_abs_diff = abs(call_close - put_close)
        if current_abs_diff < min_abs_diff:
            min_abs_diff = current_abs_diff
            atm_strike_info = {
                "weekth": weekth,
                "yyyymm": yyyymm,
                "strike": call_contract.code[3:8],
                # "strike": call_contract.code,
            }
    # return atm_strike_info.get("weekth"), atm_strike_info.get("yyyymm"), atm_strike_info.get("strike")
    return atm_strike_info.get("strike")


def 面積決定方向():
    ''' #! 學macd用面積表達強弱，先看前一個大面積慢慢貼近均線，等順勢吃第二個大面積
    2025-09-12 18:30:00   -370.000000
    2025-09-12 20:30:00     61.333333
    2025-09-12 20:45:00     -3.833333
    2025-09-12 21:30:00     36.500000
    2025-09-12 21:45:00     -7.833333
    '''
    base = kbars_resample(sid = "TXF", interval = "15T", start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
    
    ma = base.Close.rolling(18).mean().iloc[-1]
    last_close = base.Close.iloc[-1]
    if last_close > ma: 
        print("做多") if abs(last_close - ma) < 10 else print("不追多")
    if last_close < ma:
        print("做空") if abs(last_close - ma) < 10 else print("不追空")

    base['area'] = base.Close -  ma
    base['sign'] = (base['area'] >= 0).astype(int)
    base['group'] = (base['sign'] != base['sign'].shift()).cumsum()
    result = base.groupby('group').agg({'Close': 'last', 'area': 'sum'})
    result.index = base.groupby('group').apply(lambda x: x.index[-1])
    return result[['Close', 'area']]


# --- Contract Helper ---
def get_contract(code: str, yyyymm: str = None, strike: str = None, cp: str = None):
    """
    A helper function to get a contract object.
    - For Futures: get_contract("TXF", "202509")
    - For Options: get_contract("TX1", "202509", "24450", "C")
    """
    try:
        if strike and cp and yyyymm: # Options
            weekth = code
            contract_name = f"{weekth}{yyyymm}{strike}{cp}"
            return getattr(getattr(api.Contracts.Options, weekth), contract_name)
        elif yyyymm: # Futures
            # Assuming the format is like 'TXFR1' or 'MXFR1'
            # This part might need adjustment based on actual future codes
            # For now, keeping the original logic for future code construction
            future_code = f"{code.upper()}R1"
            return getattr(getattr(api.Contracts.Futures, code.upper()), future_code)
        else: # Stocks
            return getattr(api.Contracts.Stocks, code)
    except (AttributeError, KeyError) as e:
        print(f"Contract not found for: {code} {yyyymm} {strike} {cp}. Error: {e}")
        return None

def _get_price_data(contract):
    """Fetches and formats price data for a given contract."""
    if contract is None:
        return pd.DataFrame()
    kbars = api.kbars(contract)
    if not kbars.ts:
        return pd.DataFrame()
    df = _process_kbars_df(kbars)
    return (
        df[["Close"]]
        .tail(300)
        .assign(time=lambda x: [
            datetime.fromtimestamp(ts.timestamp(), tz=pytz.UTC).strftime("%m-%d %H:%M:%S")
            for ts in x.index
        ])
        .set_index('time'))


def parity_premium_trend(weekth, yyyymm, strike):

    def get_price_data(contract):
        '''是不同的get_price_data，kbars對台指期取時間戳、對選擇權取Close後再算價平和'''
        return (
            pd.DataFrame({** api.kbars(contract)})[["ts", "Close"]]
            .tail(300)
            .assign(time=lambda x: [
                datetime.fromtimestamp(ts / 1e9, tz=pytz.UTC).strftime("%m-%d %H:%M:%S")
                for ts in x['ts']
            ])
            .drop(columns=['ts'])
            .set_index('time'))

    df = get_price_data(api.Contracts.Futures["TXFR1"])
    for i in [int(strike)-100, strike, int(strike)+100]:
        call = eval(f"api.Contracts.Options.{weekth}.{weekth}{yyyymm}{i}C")
        put = eval(f"api.Contracts.Options.{weekth}.{weekth}{yyyymm}{i}P")
        df = df.join([
            get_price_data(call).rename(columns={'Close': f'Close_call_{i}'}),
            get_price_data(put).rename(columns={'Close': f'Close_put_{i}'})
        ]).ffill()
        df[f'{i}_sum'] = df[ f'Close_call_{i}'] +  df[f'Close_put_{i}']
    df.dropna(inplace=True)
    return df.iloc[-270:, [3, 6, 9]].reset_index()


def oversea_future(yyyymm="202509", symbols=None):
    """Fetches and normalizes price data for overseas futures against TXF."""
    if symbols is None:
        symbols = ["UNF", "UDF"]

    df = _get_price_data(api.Contracts.Futures.TXF.TXFR1)

    for symbol in symbols:
        try:
            # Assuming a standard contract code format, e.g., UNF202509
            contract = getattr(api.Contracts.Futures, symbol)[f"{symbol}{yyyymm}"]
            df = df.join([_get_price_data(contract).rename(columns={'Close': contract.category})]).ffill()
        except (AttributeError, KeyError):
            print(f"Oversea future contract not found for {symbol}{yyyymm}")
            continue

    numeric_cols = df.select_dtypes(include='number').columns
    normalized_df = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        first_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else 1
        if first_value == 0: first_value = 1 # Avoid division by zero
        normalized_df[col] = ((df[col] - first_value) / first_value) * 100

    return normalized_df


def strike_spread_sum(weekth, yyyymm, strike):
    """Calculates the bid-ask spread sum for a call and put option."""
    try:
        options_group = getattr(api.Contracts.Options, weekth)
        call = options_group[f"{weekth}{yyyymm}{strike}C"]
        put = options_group[f"{weekth}{yyyymm}{strike}P"]
    except (AttributeError, KeyError):
        return f"Contracts for {weekth} {yyyymm} {strike} not found."

    snapshots = api.snapshots([call, put])
    if len(snapshots) < 2:
        return "Could not fetch snapshot data for both contracts."

    snapshot_map = {s.code: s for s in snapshots}
    call_snap = snapshot_map.get(call.code)
    put_snap = snapshot_map.get(put.code)

    if not call_snap or not put_snap:
        return "Snapshot data missing for one or both contracts."

    call_spread = call_snap.sell_price - call_snap.buy_price
    put_spread = put_snap.sell_price - put_snap.buy_price

    print(f"{strike}, Call: {call_snap.buy_price} ⇄ {call_snap.sell_price}, Put: {put_snap.buy_price} ⇄ {put_snap.sell_price}")
    return call_spread + put_spread




def rsi(plot = False):
    ''' 超過年一以上的大趨勢、只會買貴不會買錯
    AI伺服器組裝廠(緯穎、廣達、緯創、神達)
    關鍵零組件如IC設計與矽智財(世芯-KY、創意、矽統)
    散熱(奇鋐、健策)、熱能交換(高力)
    PCB/載板(欣興、南電、景碩、臻鼎-KY、金像電、台光電、健鼎)
    重電設備(華城、東元)
    電池(康普*、AES-KY) 
    車用線束(貿聯-KY)
    廠務工程(漢唐、亞翔)
    自動化設備(大量、和椿、所羅門)
    軍工與航太(雷虎、事欣科)
    '''
    watchlist = ["1513","2363","2455","3443","3078",
                 "2449","8046","6515","3376","6139",
                 "2408","3037","3653","3030","3076",
                 "8028","2429","6191","4985","3035",
                 "3661","6805","6442","1519","4979" 
                 ]
    two_week_ago = (date.today() - timedelta(weeks=2)).strftime('%Y-%m-%d')
    def get_contract(stock_id):
        try:
            return eval(f"api.Contracts.Stocks.TSE.TSE{stock_id}")
        except AttributeError:
            return eval(f"api.Contracts.Stocks.OTC.OTC{stock_id}")

    series_list = [pd.Series(api.kbars(api.Contracts.Indexs.TSE.TSE001, start=two_week_ago)['Close'], name='TSE001')]
    series_list.extend([pd.Series(api.kbars(get_contract(stock_id), start=two_week_ago)['Close'], name=stock_id)
                    for stock_id in watchlist])

    df = pd.concat(series_list, axis=1).ffill()
    normalized_df = df.apply(lambda col: ((col - col.dropna().iloc[0]) / col.dropna().iloc[0]) * 100)
    normalized_df = normalized_df.reindex(columns=normalized_df.iloc[-1].sort_values(ascending=False).index)

    if plot:
        import matplotlib.pyplot as plt
        normalized_df.plot()
        plt.legend(loc=3)
        plt.show()

    return normalized_df

# ==================
# candle_stick chart
# ==================
def ohlc_chart(interval:str):
    week_ago = (date.today() - timedelta(weeks = 1)).strftime('%Y-%m-%d')
    kbars = api.kbars(api.Contracts.Futures.TXF.TXFR1, start = week_ago)
    if not kbars.ts:
        return pd.DataFrame()
    df = pd.DataFrame({**kbars})
    df.ts = pd.to_datetime(df.ts)
    df.set_index('ts', inplace = True)
    df_resample = df.resample(interval, closed = 'right', label = 'right').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
    })
    df_resample.dropna(inplace = True)
    return df_resample.tail(120)


def query_positions_unitshare():
    data = []
    for pos in api.list_positions(api.stock_account, unit=sj.constant.Unit.Share):
        data.append({
            "code": pos.code,
            "direction": pos.direction.value,
            "quantity": pos.quantity,
            "price": pos.price,
            "last_price": pos.last_price,
            "pnl": f"{pos.pnl:.2f}"
        })
    return data


if __name__ == "__main__":
    print("\n--- strike_atm ---")
