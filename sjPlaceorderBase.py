import pandas as pd
import shioaji as sj
import time

import shioaji_connector


# =================================
# Place Order and cancel
# =================================
def place_order_universal(
    action: str, price: float, octype: str, contract_type: str, **kwargs
):
    """通用下單函式"""
    contract = None
    if contract_type.lower() == "future":
        fut = kwargs.get("fut", "MXF")
        contract = getattr(shioaji_connector.api.Contracts.Futures, fut)[f"{fut}R1"]
    elif contract_type.lower() == "option":
        weekth = kwargs.get("weekth", "TX1")
        yyyymm = kwargs.get("yyyymm", "202509")
        strike = kwargs.get("strike", "24050")
        call_put = kwargs.get("call_put")
        if not call_put:
            raise ValueError("For options, 'call_put' is a required argument。")
        contract_id = f"{weekth}{yyyymm}{strike}{call_put.upper()}"
        options_group = getattr(shioaji_connector.api.Contracts.Options, weekth)
        contract = getattr(options_group, contract_id)
    else:
        raise ValueError(
            f"Invalid contract_type: '{contract_type}'. Must be 'Future' or 'Option'。"
        )

    order = shioaji_connector.api.Order(
        action=action,
        price=price,
        quantity=1,
        octype=octype,
        price_type="LMT",
        order_type="ROD",
    )
    return shioaji_connector.api.place_order(contract, order)


def new_combo(weekth="TXO", yyyymm="202509", strike="24000", price=None):
    options_group = getattr(shioaji_connector.api.Contracts.Options, weekth)
    sc_contract = getattr(options_group, f"{weekth}{yyyymm}{strike}C")
    sp_contract = getattr(options_group, f"{weekth}{yyyymm}{strike}P")

    combo_contract = sj.contracts.ComboContract(
        legs=[
            sj.contracts.ComboBase(action="Sell", **sc_contract.dict()),
            sj.contracts.ComboBase(action="Sell", **sp_contract.dict()),
        ]
    )
    order = shioaji_connector.api.ComboOrder(
        price_type="LMT",
        price=price,
        quantity=555,
        order_type="IOC",
        octype="New",
    )
    return shioaji_connector.api.place_comboorder(combo_contract, order)


def cancel_all_orders():
    """取消所有可取消的委託。"""
    shioaji_connector.api.update_status()
    cancellable_trades = [
        trade
        for trade in shioaji_connector.api.list_trades()
        if trade.status.status
        in [
            sj.constant.Status.PendingSubmit,
            sj.constant.Status.PreSubmitted,
            sj.constant.Status.Submitted,
            sj.constant.Status.Cancelled,
        ]
    ]
    if not cancellable_trades:
        return "沒有可取消的委託。"
    for trade in cancellable_trades:
        shioaji_connector.api.cancel_order(trade)
    time.sleep(1.5)
    shioaji_connector.api.update_status()


def pending():
    shioaji_connector.api.update_status()
    res = [
        {
            "t": i.status.order_datetime.strftime("%H:%M:%S"),
            "code": (
                i.contract.code[3:8] + ("C" if i.contract.code[-2] < "M" else "P")
                if len(i.contract.code) == 10
                else i.contract.code
            ),
            "action": i.order.action.value,
            "price": i.order.price,
            "octype": i.order.octype.value,
            "status": i.status.status.value,
            "modified": i.status.modified_price,
            "deals": i.status.deals,
        }
        for i in shioaji_connector.api.list_trades()
        if i.status.status == "Submitted"
    ]
    return res


# ==========================
# check existed position
# ==========================
def query_positions(mode: str = "summary", weekth: str = "TX1"):
    """查詢目前的倉位資訊，可回傳倉位摘要或權益delta。"""
    shioaji_connector.api.update_status()
    try:
        positions = shioaji_connector.api.list_positions(shioaji_connector.api.futopt_account)
    except Exception:
        positions = shioaji_connector.api.list_positions()

    positions = positions or []
    if not positions:
        return [] if mode == "summary" else "沒有倉位可計算delta。"

    if mode == "summary":
        data = []
        for pos in positions:
            _code = (
                pos.code[3:8] + ("C" if pos.code[-2] < "M" else "P")
                if len(pos.code) == 10
                else pos.code
            )
            data.append(
                {
                    "code": _code,
                    "direction": pos.direction.value,
                    "quantity": pos.quantity,
                    "price": pos.price,
                    "last_price": pos.last_price,
                    "pnl": f"{pos.pnl:.2f}",
                }
            )
        return data

    elif mode == "delta":
        try:
            options_contracts = getattr(shioaji_connector.api.Contracts.Options, weekth)
        except AttributeError:
            return f"無法取得 '{weekth}' 的合約資訊，通常是因為該週選尚未開放交易(非程式bug)"

        mxf_dir = next(
            (-1 if p.direction == "Sell" else 1 for p in positions if "MXF" in p.code),
            1,
        )
        positions_sell = {
            p.code: p.quantity
            for p in positions
            if len(p.code) > 7 and p.direction == "Sell"
        }
        positions_buy = {
            p.code: p.quantity
            for p in positions
            if len(p.code) > 7 and p.direction == "Buy"
        }

        def calculate_premium_sum(codes_dict):
            if not codes_dict:
                return 0.0
            filtered_contracts = [c for c in options_contracts if c.code in codes_dict]
            if not filtered_contracts:
                return 0.0
            snapshots = {
                s.code: s.change_price
                for s in shioaji_connector.api.snapshots(filtered_contracts)
            }
            return sum(
                codes_dict.get(code, 0) * snapshots.get(code, 0) for code in codes_dict
            )

        premium_sell_sum = calculate_premium_sum(positions_sell)
        premium_buy_sum = calculate_premium_sum(positions_buy)

        mxf_delta = 0
        if any("MXF" in p.code for p in positions):
            try:
                mxf_chg = shioaji_connector.api.snapshots([shioaji_connector.api.Contracts.Futures.MXF.MXFR1])[0].change_price
            except (IndexError, AttributeError):
                print("無法獲取MXF的快照資訊。")
                mxf_delta = 0

        net_delta = premium_sell_sum * -1 - premium_buy_sum + mxf_delta
        sell_str = f"賣方{'被軋' if premium_sell_sum > 0 else '收錢'}{abs(premium_sell_sum):.2f}"
        buy_str = f"買方{-premium_buy_sum:.2f}"
        net_str = f"綜合delta：{net_delta:.2f}"
        log = shioaji_connector.api.margin()
        return f"{sell_str}  ▎{buy_str}  ▎{net_str}  ▎可用保證金: {log.available_margin}  ▎淨值: {log.equity_amount}"
    else:
        return "無效的模式，請使用 'summary' 或 'delta'。"


if __name__ == "__main__":
    print("\n--- 因為是輔助性只需被main import故main這裡不會觸發才對 ---")