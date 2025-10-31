import json
import sys
import shioaji as sj

# --- Global API Object ---
api = None

# --- Callback Function ---
def unified_order_callback(stat, msg):
    """
    A unified callback function for handling both regular and combo order status.
    """
    try:
        # For now, just print the raw status and message for debugging.
        # A more robust implementation would parse this and log it structuredly.
        print(f"Callback received: Status={stat}, Msg={msg}")
        sys.stdout.flush()
    except Exception as e:
        print(f"An unexpected error occurred in the callback: {e}")
        import traceback
        traceback.print_exc()

# --- API Initialization ---
def initialize_shioaji_api():
    """
    Initializes the global Shioaji API object, logs in, and sets the order callback.
    This function ensures the API is only initialized once.
    """
    global api
    if api is not None:
        print("Shioaji API is already initialized.")
        return

    try:
        print("Initializing Shioaji API...")
        with open(r"Sinopac.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        api = sj.Shioaji()
        api.login(config.get("API_Key"), config.get("Secret_Key"))

        # The CA activation is critical for live trading.
        api.activate_ca(
            ca_path=r"Sinopac.pfx",
            ca_passwd=config.get("ca_passwd"),
            person_id=config.get("person_id"),
        )

        api.set_order_callback(unified_order_callback)
        print("Shioaji API initialized successfully and callback is set.")

    except FileNotFoundError as e:
        print("Error: Configuration file 'Sinopac.json' or 'Sinopac.pfx' not found.")
        print("Please ensure the credential files are in the root directory.")
        raise e
    except Exception as e:
        print(f"An error occurred during API initialization: {e}")
        raise e
