import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import traceback
import shioaji as sj
import numpy as np
import pandas as pd
import time
import json
from datetime import date, timedelta, datetime
import collections

# 從您的函式庫匯入 api 物件和下單函式, 會觸發 sjPlaceorderBase.py 中的 initialize_shioaji_api()
from sjFunctionBase import (
    ohlc_chart,
    signal_enum,
    query_positions_unitshare,
    strike_atm,
    parity_premium_trend,
)
from sjPlaceorderBase import (
    place_order_universal,
    new_combo,
    cancel_all_orders,
    query_positions,
    pending,
)
from ThirdpartySource import amount_rank, getQuoteListOption
from shioaji_connector import initialize_shioaji_api

# --- FastAPI App ---
app = FastAPI(title="Shioaji Universal Order API")


@app.on_event("startup")
async def startup_event():
    print("Application startup: Initializing Shioaji API...")
    try:
        initialize_shioaji_api()
    except Exception as e:
        print(f"FATAL: Shioaji API initialization failed during startup: {e}")
        # In a real production app, you might want to handle this more gracefully
        # or prevent the app from serving requests if the API is critical.


@app.get("/api/signal_enum")
async def get_signal_enum():
    """
    提供 signal_enum 函式的回傳值。
    """
    try:
        signal_output = signal_enum()
        return JSONResponse(content=signal_output)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )


@app.get("/api/ohlc_data")
async def get_ohlc_data(interval: str = "1T"):
    """
    提供 OHLC K線圖數據。
    """
    try:
        df = ohlc_chart(interval=interval)
        if df.empty:
            return JSONResponse(content=[])

        # 轉換為圖表庫所需的格式
        df.reset_index(inplace=True)

        # Shioaji K棒時間為台北時區，轉換為 UTC Unix timestamp
        # 確保 'ts' 是 datetime 物件且為台北時區
        if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
            df["ts"] = pd.to_datetime(df["ts"])
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize("Asia/Taipei")

        df["time"] = df["ts"].apply(lambda x: int(x.timestamp()))

        # 將欄位名稱改為小寫以符合 lightweight-charts 的需求
        chart_data = [
            {
                "time": d["time"],
                "open": d["Open"],
                "high": d["High"],
                "low": d["Low"],
                "close": d["Close"],
                "value": d["Volume"],  # 交易量序列
            }
            for d in df.to_dict(orient="records")
        ]

        return JSONResponse(content=chart_data)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )


@app.get("/api/atm_strike")
async def get_atm_strike_endpoint():
    """
    取得選擇權的價平履約價。
    """
    try:
        # Per user feedback, strike_atm() now returns a single value.
        strike = strike_atm()
        if strike is None:
            raise ValueError("Could not determine ATM strike price.")
        return JSONResponse(content={"status": "success", "atm_strike": int(strike)})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )


@app.get("/api/unitshare")
async def get_unitshare_endpoint():
    """
    取得目前的零股庫存資訊。
    """
    try:
        data = query_positions_unitshare()
        return JSONResponse(content={"status": "success", "positions": data})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=400, content={"status": "error", "message": str(e)}
        )


@app.get("/api/premium_trend")
async def get_premium_trend(weekth: str, yyyymm: str, strike: str):
    """
    Provides data for the parity premium trend chart.
    """
    try:
        df = parity_premium_trend(weekth, yyyymm, strike)
        if df.empty:
            return JSONResponse(
                content={"status": "success", "column_names": [], "data": []}
            )

        # The 'time' column is a string like '09-05 11:30:00'. It needs to be converted to a timestamp.
        current_year = datetime.now().year
        # The time format from the function is '%m-%d %H:%M:%S'
        # We need to add the year to parse it correctly.
        df["ts"] = pd.to_datetime(df["time"], format="%m-%d %H:%M:%S").apply(
            lambda dt: dt.replace(year=current_year)
        )

        # Localize to Taipei time as this is Taiwan market data
        df["ts"] = df["ts"].dt.tz_localize("Asia/Taipei")

        # Rename the original string time column to avoid confusion
        df.rename(columns={"time": "time_str"}, inplace=True)
        # Create the UNIX timestamp column that the chart needs, named 'time'
        df["time"] = df["ts"].apply(lambda x: int(x.timestamp()))

        # Get the dynamic column names for the trend lines
        data_columns = [col for col in df.columns if col.endswith("_sum")]

        # FIX: Use df.to_json and json.loads to ensure data is JSON serializable
        json_str = df[["time"] + data_columns].to_json(orient="records")
        chart_data = json.loads(json_str)

        return JSONResponse(
            content={
                "status": "success",
                "column_names": data_columns,
                "data": chart_data,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )


@app.post("/api/place_order_universal")
async def place_order_endpoint(request: Request):
    """
    接收前端請求，並呼叫 place_order_universal 函式
    """
    try:
        # 直接從前端接收 JSON payload
        payload = await request.json()
        print(f"接收到 Payload: {payload}")

        # 呼叫您在 sjPlaceorderBase.py 中定義的真實下單函式
        # api 物件已經被初始化
        trade = place_order_universal(**payload)

        # Shioaji 的 trade 物件可能無法直接序列化為 JSON
        # 我們擷取需要回傳的資訊，並將 enum 轉為字串
        trade_info = {
            "status": trade.status.status.value,
            "action": trade.order.action.value,
            "price": trade.order.price,
            "quantity": trade.order.quantity,
            "code": trade.contract.code,
            "message": trade.status.msg,
        }
        return JSONResponse(content={"status": "success", "trade_details": trade_info})

    except Exception as e:
        # 印出詳細的錯誤堆疊，方便除錯
        traceback.print_exc()
        return JSONResponse(
            status_code=400, content={"status": "error", "message": str(e)}
        )


@app.post("/api/new_combo")
async def api_new_combo(request: Request):
    data = await request.json()
    try:
        trade = new_combo(
            weekth=data.get("weekth"),
            yyyymm=data.get("yyyymm"),
            strike=data.get("strike"),
            price=float(data.get("price")),
        )

        # new_combo 回傳的 trade 物件在失敗時，其內容可能不是 JSON Serializable
        # 我們需要手動解析它，特別是處理 namedtuple 的情況
        if isinstance(trade, collections.abc.Sequence) and hasattr(trade, "_asdict"):
            trade_dict = trade._asdict()
            # 提取使用者想看到的易讀訊息
            op_msg = trade_dict.get("operation", {}).get("op_msg", "不明的錯誤")
            return JSONResponse(
                content={"status": "error", "message": op_msg, "details": trade_dict}
            )

        # 處理 ComboTrade 物件或其他可序列化的成功回應
        # 這裡假設成功的 trade 物件有 status 和 order 屬性
        response_data = {
            "status": trade.status.status.value,
            "price": trade.order.price,
            "quantity": trade.order.quantity,
            "message": trade.status.msg,
        }
        return JSONResponse(content=response_data)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=400, content={"status": "error", "message": str(e)}
        )


@app.post("/api/cancel_all_orders")
async def cancel_all_orders_endpoint():
    """
    取消所有可取消的委託。
    """
    try:
        # 更新狀態以獲取最新的委託列表
        api.update_status()

        # 篩選出處於可取消狀態的委託
        cancellable_trades = [
            trade
            for trade in api.list_trades()
            if trade.status.status
            in [
                sj.constant.Status.PendingSubmit,  # 送單中
                sj.constant.Status.PreSubmitted,  # 預約單
                sj.constant.Status.Submitted,
                # sj.constant.Status.Filling,  # 部分成交
                sj.constant.Status.Cancelled,
            ]
        ]

        if not cancellable_trades:
            return JSONResponse(
                content={"status": "success", "message": "沒有可取消的委託。"}
            )

        # 逐一送出取消請求
        for trade in cancellable_trades:
            api.cancel_order(trade)

        # 等待一段時間讓交易所處理取消回報
        time.sleep(1.5)

        # 再次更新狀態
        api.update_status()

        return JSONResponse(
            content={
                "status": "success",
                "message": f"已對 {len(cancellable_trades)} 筆委託送出取消請求。請在主控台檢視最終回報。",
            }
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=400, content={"status": "error", "message": str(e)}
        )


@app.get("/api/positions")
async def get_positions_endpoint(mode: str = "summary", weekth: str = "TX1"):
    """
    取得目前的倉位資訊或計算 delta。
    - mode=summary: 回傳倉位列表。
    - mode=delta: 回傳綜合 delta。
    - weekth: 計算 delta 時使用的週別。
    """
    try:
        if mode not in ["summary", "delta"]:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "無效的 mode 參數，請使用 'summary' 或 'delta'。",
                },
            )

        # 將 weekth 參數傳遞給後端函式
        data = query_positions(mode=mode, weekth=weekth)

        response_key = "positions" if mode == "summary" else "delta_info"
        return JSONResponse(content={"status": "success", response_key: data})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=400, content={"status": "error", "message": str(e)}
        )


@app.get("/api/pending")
async def get_pending_endpoint():
    """
    取得目前的掛單資訊。
    """
    try:
        data = pending()
        return JSONResponse(content={"status": "success", "pending_orders": data})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=400, content={"status": "error", "message": str(e)}
        )


@app.get("/api/quote_list_option")
async def get_quote_list_option_endpoint():
    """
    取得選擇權權利金列表
    """
    try:
        df = getQuoteListOption("202509", 400, 140)
        html = df.to_html(classes="positions-table", border=0)
        return JSONResponse(content={"status": "success", "html": html})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )


@app.get("/api/stock_ohlc_comparison")
async def get_stock_ohlc_comparison():
    """
    Provides OHLC data for multiple stocks for a single-day comparison.
    """
    try:
        df = amount_rank()
        if df.empty:
            return JSONResponse(content=[])
        # Convert dataframe to a list of dictionaries for the API response
        chart_data = df.to_dict(orient="records")
        return JSONResponse(content=chart_data)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500, content={"status": "error", "message": str(e)}
        )


# ======================================================================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    提供前端互動頁面
    """
    # 這裡我們直接回傳 HTML 內容，您也可以讀取一個 index.html 檔案
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>找不到 index.html</h1><p>請先建立前端檔案。</p>",
            status_code=404,
        )


if __name__ == "__main__":
    # 使用 uvicorn 啟動伺服器，監聽在 8000 port, reload=True 讓您修改程式碼後，伺服器會自動重啟
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
