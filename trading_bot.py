import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import hmac
import hashlib

class TradingBot:
    def __init__(self, api_key, api_secret, symbol="BTCUSDT", timeframe="15", leverage=10, pause=15, frequency=10, risk_perc=-0.2):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api-demo.bybit.com"  # Testnet URL
        self.symbol = symbol
        self.timeframe = timeframe
        self.leverage = leverage
        self.pause = timedelta(minutes=pause)
        self.frequency = frequency
        self.risk_perc = risk_perc
        self.last_close_time = datetime.now() - self.pause
        self.position = None
        self.position_mode = "one_way"  # Set position mode to either "hedge" or "one_way"

        # Set leverage on initialization
        self.set_leverage()

    def _generate_signature(self, params):
        param_str = '&'.join([f'{k}={params[k]}' for k in sorted(params)])
        return hmac.new(self.api_secret.encode('utf-8'), param_str.encode('utf-8'), hashlib.sha256).hexdigest()

    def _get_timestamp(self):
        return str(int(time.time() * 1000))

    def send_request(self, method, endpoint, params=None):
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        params['timestamp'] = self._get_timestamp()
        
        # Ensure 'reduceOnly' is a lowercase string
        if 'reduceOnly' in params:
            params['reduceOnly'] = 'true' if params['reduceOnly'] else 'false'
        
        # Sort parameters alphabetically and create signature
        sorted_params = dict(sorted(params.items()))
        params['sign'] = self._generate_signature(sorted_params)

        url = f"{self.base_url}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, params=params)
            elif method == "POST":
                response = requests.post(url, json=params)
            
            # Log the request details
            print(f"Request sent to {url}, Method: {method}, Params: {params}")
            
            # Check response status code and handle non-JSON responses
            if response.status_code != 200:
                print(f"Error: Received status code {response.status_code}")
                print("Response text:", response.text)
                return None
            
            # Attempt to parse JSON
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print("Network error:", e)
            return None
        
        except requests.exceptions.JSONDecodeError:
            print("Error: Failed to parse JSON response")
            print("Status Code:", response.status_code)
            print("Response Headers:", response.headers)
            print("Response text:", response.text)
            return None



    def fetch_data(self):
        """Fetches historical data and calculates indicators."""
        print("Fetching historical data...")
        endpoint = "/v5/market/kline"
        params = {
            "category": "linear",
            "symbol": self.symbol,
            "interval": self.timeframe,
            "limit": 200
        }
        data = self.send_request("GET", endpoint, params)
        if data['retCode'] != 0:
            print(f"Error fetching data: {data['retMsg']}")
            return None
        
        try:
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if len(data['result']['list'][0]) > 6:
                columns.extend([f"extra_{i}" for i in range(len(data['result']['list'][0]) - 6)])

            for row in data['result']['list']:
                row[0] = int(float(row[0]) / 1000)  # Convert to seconds if in milliseconds

            df = pd.DataFrame(data['result']['list'], columns=columns)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['close'] = df['close'].astype(float)
            print("Historical data fetched successfully.")
        
        except ValueError as e:
            print(f"Data format error: {e}")
            return None

        # Calculate indicators
        df['EMA200'] = df['close'].ewm(span=200).mean()
        df['EMA90'] = df['close'].ewm(span=90).mean()

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=12).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=12).mean()
        df['RSI12'] = 100 - (100 / (1 + gain / loss))

        df['MA20'] = df['close'].rolling(window=20).mean()
        df['BB_upper'] = df['MA20'] + (df['close'].rolling(window=20).std() * 2)
        df['BB_lower'] = df['MA20'] - (df['close'].rolling(window=20).std() * 2)

        df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Calculate ATR for volatility
        df['ATR'] = (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())
        
        return df

    def set_leverage(self):
        """Sets leverage for the trading symbol."""
        print(f"Setting leverage to {self.leverage}x for {self.symbol}.")
        endpoint = "/v5/position/set-leverage"
        params = {
            "category": "linear",
            "symbol": self.symbol,
            "buyLeverage": str(self.leverage),
            "sellLeverage": str(self.leverage)
        }
        response = self.send_request("POST", endpoint, params)
        if response['retCode'] == 0:
            print(f"Leverage set to {self.leverage}x for {self.symbol}.")
        else:
            print(f"Error setting leverage: {response['retMsg']}")

    def get_balance(self):
        """Fetches the USDT balance for trading."""
        print("Fetching account balance...")
        endpoint = "/v5/account/wallet-balance"
        params = {"accountType": "UNIFIED"}
        balance_data = self.send_request("GET", endpoint, params)
        
        # Check if balance_data is valid
        if balance_data is None:
            print("Error: No valid response received for account balance.")
            return 0

        try:
            # Traverse the nested structure based on the provided raw response
            if balance_data.get('retCode') == 0:
                coins = balance_data['result']['list'][0]['coin']
                for coin in coins:
                    if coin['coin'] == 'USDT':
                        balance = float(coin['walletBalance'])
                        print(f"Current balance: {balance} USDT")
                        return balance
                print("USDT balance not found.")
                return 0
            else:
                print(f"Error fetching balance: {balance_data.get('retMsg', 'Unknown error')}")
                return 0
        except (KeyError, TypeError) as e:
            print(f"Error processing balance data: {e}")
            print("Raw response:", balance_data)
            return 0

    def determine_position(self, df):
        """Determines if we should be in a long or short position based on EMA."""
        if df['EMA200'].iloc[-1] > df['EMA90'].iloc[-1]:
            print("Determined position: short")
            return "short"
        elif df['EMA200'].iloc[-1] < df['EMA90'].iloc[-1]:
            print("Determined position: long")
            return "long"
        print("No clear position determined.")
        return None

    def calculate_volatility_based_risk(self, df):
        """Calculates stop-loss and take-profit using ATR for dynamic risk management."""
        if 'ATR' not in df:
            df['ATR'] = (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())
        
        atr = df['ATR'].iloc[-1]
        stop_loss = atr * 1.5
        take_profit = atr * 2.5
        print(f"Calculated stop loss: {stop_loss}, take profit: {take_profit}")
        return stop_loss, take_profit

    def check_entry_conditions(self, df, position):
        """Checks entry conditions based on position and other indicators."""
        if position == "long":
            entry_condition = (df['RSI12'].iloc[-1] < 40 or 
                               df['close'].iloc[-1] < df['BB_lower'].iloc[-1] or 
                               df['MACD'].iloc[-1] < df['Signal'].iloc[-1])
        elif position == "short":
            entry_condition = (df['RSI12'].iloc[-1] > 60 or 
                               df['close'].iloc[-1] > df['BB_upper'].iloc[-1] or 
                               df['MACD'].iloc[-1] > df['Signal'].iloc[-1])
        else:
            entry_condition = False

        print(f"Entry conditions for {position}: {'Met' if entry_condition else 'Not met'}")
        return entry_condition

    def place_order(self, side, amount):
        """Places a market order with conditional positionIdx based on position mode."""
        print(f"Placing {side} order with amount: {amount}")

        endpoint = "/v5/order/create"
        params = {
            "category": "linear",
            "symbol": self.symbol,
            "side": "Buy" if side == "buy" else "Sell",
            "orderType": "Market",
            "qty": str(amount)
        }

        # Only include positionIdx if in Hedge Mode
        if self.position_mode == "hedge":
            position_idx = 2 if side.lower() == 'sell' else 1
            params["positionIdx"] = position_idx

        response = self.send_request("POST", endpoint, params)
        if response and response.get('retCode') == 0:
            print(f"{side.capitalize()} order placed successfully.")
            return response['result']
        else:
            print(f"Error placing order: {response.get('retMsg', 'Unknown error')}")
            return None


    def run(self):
        """Main function to run the bot."""
        MIN_ORDER_SIZE = 0.001  # Set minimum order size for BTCUSDT

        while True:
            now = datetime.now()
            if now - self.last_close_time < self.pause:
                print("Waiting due to cooldown.")
                time.sleep(self.frequency)
                continue

            df = self.fetch_data()
            if df is None:
                continue

            position = self.determine_position(df)

            if abs(df['EMA200'].iloc[-1] - df['EMA90'].iloc[-1]) < df['ATR'].iloc[-1] * 0.1:
                print("Low volatility, skipping trade.")
                time.sleep(self.frequency)
                continue

            if position and not self.position:
                if self.check_entry_conditions(df, position):
                    balance = self.get_balance()
                    if balance == 0:
                        print("No available balance; skipping trade.")
                        time.sleep(self.frequency)
                        continue

                    # Calculate amount and ensure it's positive
                    amount = abs(balance * self.risk_perc / df['close'].iloc[-1])
                    
                    # Ensure the amount meets the minimum order size
                    if amount < MIN_ORDER_SIZE:
                        print(f"Calculated amount {amount} is below minimum order size. Adjusting to minimum size.")
                        amount = MIN_ORDER_SIZE

                    stop_loss, take_profit = self.calculate_volatility_based_risk(df)
                    order = self.place_order(position, 0.2)
                    
                    # Only update self.position if the order was placed successfully
                    if order:
                        self.position = order
                        self.position['stop_loss'] = stop_loss
                        self.position['take_profit'] = take_profit
                        self.position['entry_price'] = df['close'].iloc[-1]
                    else:
                        print("Order placement failed; retrying on next cycle.")

            # Check if there’s an open position and evaluate conditions to close it
            if self.position:
                current_price = df['close'].iloc[-1]
                if (current_price <= self.position['entry_price'] - self.position['stop_loss'] or
                    current_price >= self.position['entry_price'] + self.position['take_profit']):
                    self.place_order('sell' if self.position['side'] == 'buy' else 'buy', self.position['amount'])
                    print(f"Position closed at {current_price}")
                    self.position = None
                    self.last_close_time = now

            print(f"Indicators - Time: {now}, EMA200: {df['EMA200'].iloc[-1]}, EMA90: {df['EMA90'].iloc[-1]}, RSI: {df['RSI12'].iloc[-1]}")
            time.sleep(self.frequency)



# Initialize and run the bot
api_key = "T6noPR0fneMpM9FOhn"
api_secret = "rcYnddAZRZikD0JMvtGyAWDKMjs1Tp28yDre"
bot = TradingBot(api_key, api_secret)
bot.run()
