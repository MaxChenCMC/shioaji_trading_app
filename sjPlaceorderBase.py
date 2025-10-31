import pandas as pd
import numpy as np
import json
import time
import sys
import os
import shioaji as sj
import pytz
from datetime import datetime, date, timedelta

# --- 全域變數 ---
api = None


# =================================
# Callback 委託與成交後券商msg加工
# =================================
def unified_order_callback(stat, msg):
    """
    一個統一的委託回報函式，能同時處理一般委託和組合單委託。
    """
    try:
        # 判斷是否為組合單
        is_combo = msg.get('order', {}).get('combo', False)

        # --- 處理組合單回報 ---
        if is_combo:
            if stat == sj.constant.Status.Failed:
                error_msg = msg.get('operation', {}).get('op_msg', '組合單原因不明')
                print(f"組合單委託失敗: {error_msg}")
            elif stat == sj.constant.Status.Filled:
                order_info = msg.get('order', {})
                price = order_info.get('price', 'N/A')
                quantity = order_info.get('quantity', 'N/A')
                print(f"組合單成交: Price={price}, Quantity={quantity}")
            elif stat == sj.constant.Status.Cancelled:
                order_info = msg.get('order', {})
                price = order_info.get('price', 'N/A')
                quantity = order_info.get('quantity', 'N/A')
                print(f"組合單取消: Price={price}, Quantity={quantity}")
            elif stat in [sj.constant.Status.Submitted, sj.constant.Status.PreSubmitted]:
                pass  # 組合單送出時保持靜默

        # --- 處理一般委託回報 ---
        else:
            if stat == sj.constant.Status.Failed:
                error_msg = msg.get('status', {}).get('msg', '一般單原因不明')
                print(f"委託失敗: {error_msg}")
            elif stat == sj.constant.Status.Filled:
                contract_info = msg.get('contract', {})
                order_info = msg.get('order', {})
                _ = {
                    "狀態": "完全成交",
                    "商品": contract_info.get("code", "N/A"),
                    "月份": contract_info.get("delivery_month", "N/A"),
                    "履約價": contract_info.get("strike_price", "N/A"),
                    "買賣": order_info.get("action", "N/A"),
                    "C/P": contract_info.get("option_right", "")[6:] if contract_info.get("option_right", "") else "N/A",
                    "數量": order_info.get("quantity", "N/A"),
                    "成交價": msg.get("price", "N/A"),
                    "倉別": order_info.get("oc_type", "N/A"),
                }
                print(f"成交回報: { _}")
            elif stat == sj.constant.Status.Cancelled:
                order_info = msg.get('order', {})
                status_info = msg.get('status', {})
                contract_info = msg.get('contract', {})
                act = order_info.get('action', 'N/A')
                price = order_info.get('price', 'N/A')
                contract_code = contract_info.get('code', 'N/A')
                cancel_qty = status_info.get('cancel_quantity', 'N/A')
                print(f"委託取消：{act} {contract_code} at {price}, Qty: {cancel_qty}")
            elif stat in [sj.constant.Status.Submitted, sj.constant.Status.PreSubmitted]:
                pass  # 一般委託送出時保持靜默

    except Exception as e:
        print(f"Callback 處理時發生未預期錯誤: {e}")
        import traceback
        traceback.print_exc()

    sys.stdout.flush()


def 零股下單():
    api.snapshots([api.Contracts.Stocks["2330"]])[0].close
    contract = api.Contracts.Stocks.TSE.TSE3706
    order = api.Order(action = "Sell", price = 88.9, quantity = 500, order_lot = "IntradayOdd" ,price_type = "LMT", order_type = "ROD")
    trade = api.place_order(contract, order)
    return trade


# --- API 初始化函式 ---
def initialize_shioaji_api():
    """初始化 Shioaji API 物件並設定委託回報。"""
    global api
    if api is not None:
        print("Shioaji API 已初始化。 সন")
        return api
    try:
        # 讀取設定檔
        with open(r'Sinopac.json', 'r', encoding='utf-8') as f:
            file = json.load(f)
        
        # 登入並啟用 CA
        api = sj.Shioaji()
        api.login(file.get('API_Key'), file.get('Secret_Key'))
        api.activate_ca(
            ca_path=r"Sinopac.pfx",
            ca_passwd=file.get('ca_passwd'),
            person_id=file.get('person_id')
        )
        
        api.set_order_callback(unified_order_callback)
        
        print("Shioaji API 初始化成功並已設定 Callback。")
        return api
        
    except FileNotFoundError:
        print("錯誤: 找不到設定檔 'Sinopac.json' 或 'Sinopac.pfx'。")
        sys.exit(1)
    except Exception as e:
        print(f"API 初始化過程中發生錯誤: {e}")
        sys.exit(1)

# --- 主程式區塊 ---
# 確保 Windows 控制台編碼正確
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# 模組載入時即執行初始化
initialize_shioaji_api()


# =================================
# Place Order and cancel
# =================================
def place_order_universal(action: str, price: float, octype: str, contract_type: str, **kwargs):
    """通用下單函式"""
    contract = None
    if contract_type.lower() == 'future':
        fut = kwargs.get('fut', 'MXF')
        contract = getattr(api.Contracts.Futures, fut)[f"{fut}R1"]
    elif contract_type.lower() == 'option':
        weekth = kwargs.get('weekth', 'TX1')
        yyyymm = kwargs.get('yyyymm', '202509')
        strike = kwargs.get('strike', '24050')
        call_put = kwargs.get('call_put')
        if not call_put:
            raise ValueError("For options, 'call_put' is a required argument。")
        contract_id = f"{weekth}{yyyymm}{strike}{call_put.upper()}"
        contract = getattr(api.Contracts.Options, weekth)[contract_id]
    else:
        raise ValueError(f"Invalid contract_type: '{contract_type}'. Must be 'Future' or 'Option'。")

    order = api.Order(
        action=action,
        price=price,
        quantity=1,
        octype=octype,
        price_type="LMT",
        order_type="ROD"
    )
    return api.place_order(contract, order)


def new_combo(weekth= 'TXO', yyyymm = '202509', strike = '24000', price = None):
    SC = eval(f"api.Contracts.Options.{weekth}.{weekth}{yyyymm}{strike}C")
    SP = eval(f"api.Contracts.Options.{weekth}.{weekth}{yyyymm}{strike}P")
    combo_contract = sj.contracts.ComboContract(legs=[sj.contracts.ComboBase(action= "Sell", **SC.dict()),
                                                      sj.contracts.ComboBase(action= "Sell", **SP.dict())])
    order = api.ComboOrder(
        price_type = "LMT",
        price = price,
        quantity = 555,
        order_type = "IOC",
        octype = "New",
    )
    return api.place_comboorder(combo_contract, order)


def cancel_all_orders():
    """取消所有可取消的委託。"""
    api.update_status()
    cancellable_trades = [
        trade for trade in api.list_trades()
        if trade.status.status in [
            sj.constant.Status.PendingSubmit,
            sj.constant.Status.PreSubmitted,
            sj.constant.Status.Submitted,
            sj.constant.Status.Cancelled 
        ]
    ]
    if not cancellable_trades:
        return "沒有可取消的委託。"
    for trade in cancellable_trades:
        api.cancel_order(trade)
    time.sleep(1.5)
    api.update_status()

    
def pending():
    api.update_status()
    res = [{
    "t":i.status.order_datetime.strftime("%H:%M:%S"),
    "code":i.contract.code[3:8] + ("C" if i.contract.code[-2] < "M" else "P") if len(i.contract.code) == 10 else i.contract.code, 
    "action":i.order.action.value, 
    "price":i.order.price, 
    "octype":i.order.octype.value, 
    "status":i.status.status.value,
    "modified":i.status.modified_price,
    "deals":i.status.deals,
    } for i in api.list_trades() if i.status.status == 'Submitted']
    return res

# ==========================
# check existed position
# ==========================
def query_positions(mode: str = "summary", weekth: str = "TX1"):
    """查詢目前的倉位資訊，可回傳倉位摘要或權益delta。"""
    api.update_status()
    try:
        positions = api.list_positions(api.futopt_account)
    except Exception:
        positions = api.list_positions()

    positions = positions or []
    if not positions:
        return [] if mode == "summary" else "沒有倉位可計算delta。"

    if mode == "summary":
        data = []
        for pos in positions:
            _code = pos.code[3:8] + ("C" if pos.code[-2] < "M" else "P") if len(pos.code) == 10 else pos.code
            data.append({
                "code": _code, 
                "direction": pos.direction.value,
                "quantity": pos.quantity,
                "price": pos.price,
                "last_price": pos.last_price,
                "pnl": f"{pos.pnl:.2f}"
            })
        return data

    elif mode == "delta":
        try:
            options_contracts = getattr(api.Contracts.Options, weekth)
        except AttributeError:
            return f"無法取得 '{weekth}' 的合約資訊，通常是因為該週選尚未開放交易(非程式bug)"

        # 做多時這樣調才對，但為何之前做空時那種寫法也對！？怪了
        mxf_dir = next((-1 if p.direction == "Sell" else 1 for p in positions if "MXF" in p.code), 1)
        positions_sell = {p.code: p.quantity for p in positions if len(p.code) > 7 and p.direction == "Sell"}
        positions_buy = {p.code: p.quantity for p in positions if len(p.code) > 7 and p.direction == "Buy"}

        def calculate_premium_sum(codes_dict):
            if not codes_dict:
                return 0.0
            filtered_contracts = [c for c in options_contracts if c.code in codes_dict]
            if not filtered_contracts:
                return 0.0
            snapshots = {s.code: s.change_price for s in api.snapshots(filtered_contracts)}
            return sum(codes_dict.get(code, 0) * snapshots.get(code, 0) for code in codes_dict)

        premium_sell_sum = calculate_premium_sum(positions_sell)
        premium_buy_sum = calculate_premium_sum(positions_buy)

        mxf_delta = 0
        if any("MXF" in p.code for p in positions):
            try:
                mxf_chg = api.snapshots([api.Contracts.Futures.MXF.MXFR1])[0].change_price
            except (IndexError, AttributeError):
                print("無法獲取MXF的快照資訊。")
                mxf_delta = 0
                
        # 做多時這樣調才對，但為何之前做空時那種寫法也對！？怪了
        net_delta = premium_sell_sum*-1 - premium_buy_sum + mxf_delta
        sell_str = f"賣方{'被軋' if premium_sell_sum > 0 else '收錢'}{abs(premium_sell_sum):.2f}"
        buy_str = f"買方{-premium_buy_sum:.2f}"
        net_str = f"綜合delta：{net_delta:.2f}"
        log = api.margin()
        return f"{sell_str}  ▎{buy_str}  ▎{net_str}  ▎可用保證金: {log.available_margin}  ▎淨值: {log.equity_amount}"
    else:
        return "無效的模式，請使用 'summary' 或 'delta'。"


if __name__ == "__main__":
    print("\n--- 因為是輔助性只需被main import故main這裡不會觸發才對 ---")
