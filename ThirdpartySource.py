import pandas as pd
import requests
from datetime import datetime


def getQuoteListOption(yyyymm_weekth: str, premium_upper_bound: int, premium_lower_bound: int) -> pd.DataFrame:
    market_time = datetime.now().strftime('%H:%M')
    if not ('08:45' < market_time < '13:45' or market_time > '15:00' or market_time < '05:00'):
        return print("請在台指期交易時間內使用")
    arg1, arg2, market_type = ("TXFI5-F", "TXO-Q", '0') if '08:45' < market_time < '13:45' else ("TXFI5-M", "TXO-R", '1')
    last_close = float(requests.post("https://mis.taifex.com.tw/futures/api/getQuoteDetail", json={
        "SymbolID": ["TXF-S", arg1, arg2] #! 這裡的 TXF-S 是台指期當月契約
        }).json()["RtData"]['QuoteList'][1]['CLastPrice'])
    strike_range = [str(int(last_close / 50) * 50 + i * 50) for i in range(-3, 5)]
    quote_table = requests.post("https://mis.taifex.com.tw/futures/api/getQuoteListOption", json={
        "MarketType": market_type, 
        "SymbolType": "O", 
        "KindID": "1", 
        "CID": "TXO", #! 這裡的 TXO 是台指選當月契約
        "ExpireMonth": yyyymm_weekth,  #! 例如 "2024063" 代表 2024 年 6 月第 3 週到期
        "RowSize": "全部", 
        "PageNo": "", 
        "SortColumn": "", 
        "AscDesc": "A"
        }).json()["RtData"]['QuoteList']
    data = {item['DispEName'][-6:]: (float(item['CBestAskPrice']) + float(item['CBestBidPrice'])) / 2 for item in quote_table if item['DispEName'][-6:-1] in strike_range}
    call_data, put_data = {k: v for k, v in data.items() if k.endswith('C')}, {k: v for k, v in data.items() if k.endswith('P')}
    strikes = sorted(set(k[:5] for k in call_data.keys()), reverse=True)
    df = pd.DataFrame({'call': [call_data.get(s+'C', 0) for s in strikes], 'put': [put_data.get(s+'P', 0) for s in strikes]}, index=strikes).astype(object)
    df.loc[~df['call'].between(premium_lower_bound, premium_upper_bound), 'call'] = ""
    df.loc[~df['put'].between(premium_lower_bound, premium_upper_bound), 'put'] = ""
    return df
    

def amount_rank(chg_pct = 1, vola_pct = 4, twse_weight = 50):
    df = pd.read_html("https://histock.tw/stock/rank.aspx?m=13&d=1&p=all")[0][['代號▼', '名稱▼', '漲跌幅▼', '振幅▼', '開盤▼', '最高▼', '最低▼', '價格▼', '昨收▼', '成交值(億)▼']].head(20)
    df = df[df["代號▼"].apply(lambda x: len(x) == 4)]
    df["漲跌幅▼"] = df["漲跌幅▼"].replace("--", "0.0%").apply(lambda x: float(x[:-1]))
    df["振幅▼"] = df["振幅▼"].apply(lambda x: float(x[:-1]))
    float_columns = ['開盤▼', '最高▼', '最低▼', '價格▼']
    int_columns = ['昨收▼', '成交值(億)▼']
    
    for col in float_columns:
        df[col] = df[col].astype(float)
    for col in int_columns:
        df[col] = df[col].astype(int)
    for new_col, price_col in [('O', '開盤▼'), ('H', '最高▼'), ('L', '最低▼'), ('C', '價格▼')]:
        df[new_col] = 100 * (df[price_col] - df['昨收▼']) / df['昨收▼']
    
    df = df[
        ((df["漲跌幅▼"] > chg_pct) | (df["漲跌幅▼"] < -chg_pct)) 
        & (df["振幅▼"] > vola_pct) 
        & (df["成交值(億)▼"] > twse_weight)]
    [["名稱▼", "O", "H", "L", "C"]]
    return df.round(2)

