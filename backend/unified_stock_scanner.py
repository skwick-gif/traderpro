#!/usr/bin/env python3
"""
Unified AI-Powered Stock Scanner
Advanced multi-factor stock analysis system with technical, fundamental, sentiment, and pattern recognition.
"""

import yfinance as yf
import pandas as pd
import numpy as np
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available, using basic indicators")

from textblob import TextBlob
import requests
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging
import json
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ScanSettings:
    """Configuration for stock scanning"""
    stock_universe: str = "NASDAQ100"
    min_score: float = 7.0
    max_results: int = 25
    technical_weight: float = 0.35
    fundamental_weight: float = 0.25
    sentiment_weight: float = 0.25
    pattern_weight: float = 0.15
    min_price: float = 10.0
    max_price: float = 1000.0
    min_volume: int = 100000
    min_market_cap: float = 1000000000

@dataclass
class StockScore:
    """Stock analysis results"""
    symbol: str
    current_price: float
    volume: int
    market_cap: float
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    pattern_score: float
    total_score: float
    risk_level: str
    prediction: str
    timestamp: datetime

class StockUniverse:
    """Stock universe definitions"""
    
    @staticmethod
    def get_nasdaq100() -> List[str]:
        """Get NASDAQ 100 symbols"""
        try:
            # Download NASDAQ 100 list from reliable source
            url = "https://en.wikipedia.org/wiki/NASDAQ-100"
            tables = pd.read_html(url)
            nasdaq_df = tables[4]  # The main table is usually the 5th (index 4)
            symbols = nasdaq_df['Ticker'].tolist()
            return [s.replace('.', '-') for s in symbols if isinstance(s, str)]
        except:
            # Fallback to predefined list
            return [
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'AVGO', 'PEP',
                'COST', 'ADBE', 'NFLX', 'TMUS', 'CSCO', 'INTC', 'TXN', 'QCOM', 'INTU', 'AMAT',
                'HON', 'AMD', 'SBUX', 'GILD', 'BKNG', 'MDLZ', 'FISV', 'ADP', 'MU', 'CSX',
                'REGN', 'ADI', 'ISRG', 'VRTX', 'PANW', 'PYPL', 'KLAC', 'SNPS', 'CDNS', 'MRVL',
                'MAR', 'ORLY', 'FTNT', 'MCHP', 'KDP', 'CTAS', 'LRCX', 'ABNB', 'NXPI', 'WDAY'
            ]
    
    @staticmethod
    def get_sp500() -> List[str]:
        """Get S&P 500 symbols"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_df = tables[0]
            return sp500_df['Symbol'].tolist()
        except:
            # Fallback to major S&P 500 symbols
            return [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
                'V', 'WMT', 'PG', 'MA', 'HD', 'CVX', 'JPM', 'ABBV', 'XOM', 'LLY', 'KO', 'MRK',
                'AVGO', 'ORCL', 'PEP', 'COST', 'TMO', 'WFC', 'DIS', 'ACN', 'ABT', 'CRM', 'VZ',
                'ADBE', 'MCD', 'NFLX', 'NKE', 'BMY', 'PM', 'LIN', 'T', 'TXN', 'NEE', 'UPS',
                'QCOM', 'LOW', 'HON', 'RTX', 'BA', 'IBM', 'COP', 'SPGI', 'INTU', 'CAT'
            ]

class TechnicalAnalyzer:
    """Technical analysis engine"""
    
    def analyze(self, data: pd.DataFrame) -> float:
        """Perform comprehensive technical analysis"""
        try:
            # Ensure we have enough data
            if len(data) < 50:
                return 5.0
                
            scores = []
            
            # 1. Trend Analysis (25%)
            trend_score = self._analyze_trend(data)
            scores.append(trend_score * 0.25)
            
            # 2. Momentum Analysis (25%)
            momentum_score = self._analyze_momentum(data)
            scores.append(momentum_score * 0.25)
            
            # 3. Volatility Analysis (25%)
            volatility_score = self._analyze_volatility(data)
            scores.append(volatility_score * 0.25)
            
            # 4. Volume Analysis (25%)
            volume_score = self._analyze_volume(data)
            scores.append(volume_score * 0.25)
            
            return sum(scores)
            
        except Exception as e:
            logger.warning(f"Technical analysis error: {e}")
            return 5.0
    
    def _analyze_trend(self, data: pd.DataFrame) -> float:
        """Analyze trend strength and direction"""
        try:
            closes = data['Close'].values
            
            # Moving averages
            if TALIB_AVAILABLE:
                sma_20 = talib.SMA(closes, timeperiod=20)
                sma_50 = talib.SMA(closes, timeperiod=50)
                ema_12 = talib.EMA(closes, timeperiod=12)
                ema_26 = talib.EMA(closes, timeperiod=26)
            else:
                sma_20 = pd.Series(closes).rolling(20).mean().values
                sma_50 = pd.Series(closes).rolling(50).mean().values
                ema_12 = pd.Series(closes).ewm(span=12).mean().values
                ema_26 = pd.Series(closes).ewm(span=26).mean().values
            
            score = 5.0
            
            # Price vs moving averages
            current_price = closes[-1]
            if current_price > sma_20[-1]:
                score += 1
            if current_price > sma_50[-1]:
                score += 1
            if sma_20[-1] > sma_50[-1]:
                score += 1
            
            # MACD
            if TALIB_AVAILABLE:
                macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
            else:
                macd = ema_12 - ema_26
                macd_signal = pd.Series(macd).ewm(span=9).mean().values
                macd_hist = macd - macd_signal
            
            if macd[-1] > macd_signal[-1]:
                score += 1
            if macd_hist[-1] > 0:
                score += 1
                
            return min(score, 10.0)
            
        except Exception:
            return 5.0
    
    def _analyze_momentum(self, data: pd.DataFrame) -> float:
        """Analyze momentum indicators"""
        try:
            closes = data['Close'].values
            highs = data['High'].values
            lows = data['Low'].values
            
            score = 5.0
            
            # RSI
            if TALIB_AVAILABLE:
                rsi = talib.RSI(closes, timeperiod=14)
            else:
                delta = pd.Series(closes).diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = (100 - (100 / (1 + rs))).values
            
            current_rsi = rsi[-1]
            if 30 < current_rsi < 70:  # Not overbought/oversold
                score += 1
            if current_rsi > rsi[-5]:  # RSI trending up
                score += 0.5
            
            # Stochastic
            if TALIB_AVAILABLE:
                slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowd_period=3)
                if slowk[-1] > slowd[-1]:
                    score += 1
            else:
                # Simple stochastic approximation
                low_14 = pd.Series(lows).rolling(14).min()
                high_14 = pd.Series(highs).rolling(14).max()
                k_percent = 100 * ((pd.Series(closes) - low_14) / (high_14 - low_14))
                if k_percent.iloc[-1] > k_percent.rolling(3).mean().iloc[-1]:
                    score += 1
            
            # Williams %R
            if TALIB_AVAILABLE:
                willr = talib.WILLR(highs, lows, closes, timeperiod=14)
                if willr[-1] > -50:
                    score += 1
            else:
                # Williams %R approximation
                high_14 = pd.Series(highs).rolling(14).max()
                low_14 = pd.Series(lows).rolling(14).min()
                willr = -100 * ((high_14 - pd.Series(closes)) / (high_14 - low_14))
                if willr.iloc[-1] > -50:
                    score += 1
            
            # Price Rate of Change
            if TALIB_AVAILABLE:
                roc = talib.ROC(closes, timeperiod=10)
            else:
                roc = pd.Series(closes).pct_change(periods=10) * 100
            
            if roc.iloc[-1] > 0:
                score += 1.5
                
            return min(score, 10.0)
            
        except Exception:
            return 5.0
    
    def _analyze_volatility(self, data: pd.DataFrame) -> float:
        """Analyze volatility patterns"""
        try:
            closes = data['Close'].values
            highs = data['High'].values
            lows = data['Low'].values
            
            score = 5.0
            
            # Bollinger Bands
            if TALIB_AVAILABLE:
                upper, middle, lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)
            else:
                middle = pd.Series(closes).rolling(20).mean()
                std = pd.Series(closes).rolling(20).std()
                upper = middle + (std * 2)
                lower = middle - (std * 2)
                upper = upper.values
                middle = middle.values
                lower = lower.values
            
            current_price = closes[-1]
            bb_position = (current_price - lower[-1]) / (upper[-1] - lower[-1])
            
            # Prefer stocks in middle range (not too extreme)
            if 0.2 < bb_position < 0.8:
                score += 2
            elif bb_position > 0.8:  # Near upper band
                score += 1
            
            # Average True Range (volatility)
            if TALIB_AVAILABLE:
                atr = talib.ATR(highs, lows, closes, timeperiod=14)
            else:
                high_low = pd.Series(highs) - pd.Series(lows)
                high_close = np.abs(pd.Series(highs) - pd.Series(closes).shift())
                low_close = np.abs(pd.Series(lows) - pd.Series(closes).shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(14).mean().values
            
            atr_pct = (atr[-1] / closes[-1]) * 100
            
            # Prefer moderate volatility
            if 1 < atr_pct < 4:
                score += 2
            elif atr_pct < 1:
                score += 1
                
            return min(score, 10.0)
            
        except Exception:
            return 5.0
    
    def _analyze_volume(self, data: pd.DataFrame) -> float:
        """Analyze volume patterns"""
        try:
            volumes = data['Volume'].values
            closes = data['Close'].values
            
            score = 5.0
            
            # Volume moving averages
            if TALIB_AVAILABLE:
                vol_sma_20 = talib.SMA(volumes.astype(float), timeperiod=20)
            else:
                vol_sma_20 = pd.Series(volumes).rolling(20).mean().values
            
            current_volume = volumes[-1]
            
            # Above average volume
            if current_volume > vol_sma_20[-1]:
                score += 2
            
            # On Balance Volume
            if TALIB_AVAILABLE:
                obv = talib.OBV(closes, volumes.astype(float))
            else:
                price_change = pd.Series(closes).diff()
                obv = (volumes * np.sign(price_change)).cumsum()
                obv = obv.values
            
            if obv[-1] > obv[-5]:  # OBV trending up
                score += 1.5
            
            # Volume Price Trend
            if TALIB_AVAILABLE:
                vpt = talib.AD(data['High'].values, data['Low'].values, closes, volumes.astype(float))
            else:
                money_flow_multiplier = ((closes - data['Low'].values) - (data['High'].values - closes)) / (data['High'].values - data['Low'].values)
                money_flow_volume = money_flow_multiplier * volumes
                vpt = money_flow_volume.cumsum()
            
            if vpt[-1] > vpt[-5]:  # VPT trending up
                score += 1.5
                
            return min(score, 10.0)
            
        except Exception:
            return 5.0

class FundamentalAnalyzer:
    """Fundamental analysis engine"""
    
    def analyze(self, ticker: yf.Ticker, symbol: str) -> float:
        """Perform fundamental analysis"""
        try:
            info = ticker.info
            score = 5.0
            
            # 1. Valuation Metrics (30%)
            valuation_score = self._analyze_valuation(info)
            score += valuation_score * 0.3
            
            # 2. Growth Metrics (30%)
            growth_score = self._analyze_growth(info)
            score += growth_score * 0.3
            
            # 3. Profitability Metrics (25%)
            profitability_score = self._analyze_profitability(info)
            score += profitability_score * 0.25
            
            # 4. Financial Health (15%)
            health_score = self._analyze_financial_health(info)
            score += health_score * 0.15
            
            return min(score, 10.0)
            
        except Exception as e:
            logger.warning(f"Fundamental analysis error for {symbol}: {e}")
            return 5.0
    
    def _analyze_valuation(self, info: dict) -> float:
        """Analyze valuation metrics"""
        score = 0.0
        
        # P/E Ratio
        pe = info.get('trailingPE', 0)
        if 0 < pe < 15:
            score += 2.0
        elif 15 <= pe < 25:
            score += 1.0
        elif 25 <= pe < 40:
            score += 0.5
        
        # PEG Ratio
        peg = info.get('pegRatio', 0)
        if 0 < peg < 1:
            score += 2.0
        elif 1 <= peg < 2:
            score += 1.0
        
        # Price to Book
        pb = info.get('priceToBook', 0)
        if 0 < pb < 2:
            score += 1.0
        elif 2 <= pb < 4:
            score += 0.5
        
        return min(score, 5.0)
    
    def _analyze_growth(self, info: dict) -> float:
        """Analyze growth metrics"""
        score = 0.0
        
        # Revenue Growth
        rev_growth = info.get('revenueGrowth', 0)
        if rev_growth > 0.2:
            score += 2.0
        elif rev_growth > 0.1:
            score += 1.0
        elif rev_growth > 0:
            score += 0.5
        
        # Earnings Growth
        earnings_growth = info.get('earningsGrowth', 0)
        if earnings_growth > 0.15:
            score += 2.0
        elif earnings_growth > 0.05:
            score += 1.0
        
        # Forward P/E vs Trailing P/E (improvement expected)
        forward_pe = info.get('forwardPE', 0)
        trailing_pe = info.get('trailingPE', 0)
        if forward_pe > 0 and trailing_pe > 0 and forward_pe < trailing_pe:
            score += 1.0
        
        return min(score, 5.0)
    
    def _analyze_profitability(self, info: dict) -> float:
        """Analyze profitability metrics"""
        score = 0.0
        
        # Return on Equity
        roe = info.get('returnOnEquity', 0)
        if roe > 0.15:
            score += 1.5
        elif roe > 0.1:
            score += 1.0
        elif roe > 0.05:
            score += 0.5
        
        # Return on Assets
        roa = info.get('returnOnAssets', 0)
        if roa > 0.1:
            score += 1.5
        elif roa > 0.05:
            score += 1.0
        
        # Profit Margin
        profit_margin = info.get('profitMargins', 0)
        if profit_margin > 0.15:
            score += 2.0
        elif profit_margin > 0.1:
            score += 1.0
        elif profit_margin > 0.05:
            score += 0.5
        
        return min(score, 5.0)
    
    def _analyze_financial_health(self, info: dict) -> float:
        """Analyze financial health"""
        score = 0.0
        
        # Debt to Equity
        debt_to_equity = info.get('debtToEquity', 0)
        if debt_to_equity < 30:
            score += 2.0
        elif debt_to_equity < 60:
            score += 1.0
        
        # Current Ratio
        current_ratio = info.get('currentRatio', 0)
        if current_ratio > 2:
            score += 1.5
        elif current_ratio > 1.5:
            score += 1.0
        elif current_ratio > 1:
            score += 0.5
        
        # Free Cash Flow
        free_cash_flow = info.get('freeCashflow', 0)
        if free_cash_flow > 0:
            score += 1.5
        
        return min(score, 5.0)

class SentimentAnalyzer:
    """News and sentiment analysis engine"""
    
    def analyze(self, symbol: str) -> float:
        """Analyze news sentiment for stock"""
        try:
            # Get recent news
            news_data = self._get_news(symbol)
            sentiment_score = self._analyze_news_sentiment(news_data)
            
            # Get social sentiment (placeholder - can be extended)
            social_score = self._get_social_sentiment(symbol)
            
            # Combine scores
            total_score = (sentiment_score * 0.7) + (social_score * 0.3)
            return min(total_score, 10.0)
            
        except Exception as e:
            logger.warning(f"Sentiment analysis error for {symbol}: {e}")
            return 5.0
    
    def _get_news(self, symbol: str) -> List[str]:
        """Get recent news for symbol"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            headlines = [article.get('title', '') for article in news[:10]]
            return headlines
        except:
            return []
    
    def _analyze_news_sentiment(self, headlines: List[str]) -> float:
        """Analyze sentiment of news headlines"""
        if not headlines:
            return 5.0
        
        sentiments = []
        for headline in headlines:
            if headline:
                blob = TextBlob(headline)
                # Convert polarity (-1 to 1) to score (0 to 10)
                score = (blob.sentiment.polarity + 1) * 5
                sentiments.append(score)
        
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            return min(avg_sentiment, 10.0)
        
        return 5.0
    
    def _get_social_sentiment(self, symbol: str) -> float:
        """Get social media sentiment (placeholder)"""
        # This could be extended to include Twitter, Reddit, etc.
        # For now, return neutral
        return 5.0

class PatternRecognizer:
    """Pattern recognition engine"""
    
    def analyze(self, data: pd.DataFrame) -> Tuple[float, str]:
        """Recognize chart patterns and candlestick patterns"""
        try:
            if len(data) < 30:
                return 5.0, "Insufficient data"
            
            pattern_score = 5.0
            patterns_found = []
            
            # Candlestick Patterns
            candlestick_score, candlestick_patterns = self._analyze_candlesticks(data)
            pattern_score += candlestick_score
            patterns_found.extend(candlestick_patterns)
            
            # Chart Patterns
            chart_score, chart_patterns = self._analyze_chart_patterns(data)
            pattern_score += chart_score
            patterns_found.extend(chart_patterns)
            
            # Support/Resistance
            sr_score = self._analyze_support_resistance(data)
            pattern_score += sr_score
            
            prediction = self._generate_pattern_prediction(patterns_found, pattern_score)
            
            return min(pattern_score, 10.0), prediction
            
        except Exception as e:
            logger.warning(f"Pattern recognition error: {e}")
            return 5.0, "Pattern analysis unavailable"
    
    def _analyze_candlesticks(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Analyze candlestick patterns"""
        opens = data['Open'].values
        highs = data['High'].values
        lows = data['Low'].values
        closes = data['Close'].values
        
        score = 0.0
        patterns = []
        
        try:
            if TALIB_AVAILABLE:
                # Bullish patterns
                hammer = talib.CDLHAMMER(opens, highs, lows, closes)
                if hammer[-1] > 0:
                    score += 1.0
                    patterns.append("Hammer")
                
                engulfing = talib.CDLENGULFING(opens, highs, lows, closes)
                if engulfing[-1] > 0:
                    score += 1.5
                    patterns.append("Bullish Engulfing")
                
                morning_star = talib.CDLMORNINGSTAR(opens, highs, lows, closes)
                if morning_star[-1] > 0:
                    score += 2.0
                    patterns.append("Morning Star")
                
                # Bearish patterns (negative score)
                if engulfing[-1] < 0:
                    score -= 1.0
                    patterns.append("Bearish Engulfing")
                
                evening_star = talib.CDLEVENINGSTAR(opens, highs, lows, closes)
                if evening_star[-1] < 0:
                    score -= 1.5
                    patterns.append("Evening Star")
            else:
                # Basic pattern recognition without TA-Lib
                # Hammer pattern approximation
                for i in range(-5, 0):
                    body = abs(closes[i] - opens[i])
                    upper_shadow = highs[i] - max(opens[i], closes[i])
                    lower_shadow = min(opens[i], closes[i]) - lows[i]
                    
                    if lower_shadow > 2 * body and upper_shadow < body * 0.1:
                        score += 0.5
                        patterns.append("Hammer-like")
                        break
                        
        except Exception:
            pass
        
        return score, patterns
    
    def _analyze_chart_patterns(self, data: pd.DataFrame) -> Tuple[float, List[str]]:
        """Analyze chart patterns"""
        closes = data['Close'].values
        highs = data['High'].values
        lows = data['Low'].values
        
        score = 0.0
        patterns = []
        
        try:
            # Trend analysis
            if len(closes) >= 20:
                # Simple trend detection
                recent_trend = np.polyfit(range(10), closes[-10:], 1)[0]
                longer_trend = np.polyfit(range(20), closes[-20:], 1)[0]
                
                if recent_trend > 0 and longer_trend > 0:
                    score += 1.5
                    patterns.append("Uptrend")
                elif recent_trend > 0 > longer_trend:
                    score += 2.0
                    patterns.append("Trend Reversal Up")
            
            # Breakout detection
            if len(closes) >= 20:
                recent_high = np.max(highs[-20:-1])
                current_price = closes[-1]
                
                if current_price > recent_high * 1.02:  # 2% breakout
                    score += 2.0
                    patterns.append("Breakout")
                    
        except Exception:
            pass
        
        return score, patterns
    
    def _analyze_support_resistance(self, data: pd.DataFrame) -> float:
        """Analyze support and resistance levels"""
        try:
            closes = data['Close'].values
            if len(closes) < 20:
                return 0.0
            
            current_price = closes[-1]
            recent_prices = closes[-20:]
            
            # Simple support/resistance
            resistance = np.max(recent_prices)
            support = np.min(recent_prices)
            
            # Price position
            price_position = (current_price - support) / (resistance - support)
            
            # Prefer stocks near support (potential bounce)
            if 0.1 <= price_position <= 0.3:
                return 1.0
            elif 0.7 <= price_position <= 0.9:
                return 0.5  # Near resistance
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _generate_pattern_prediction(self, patterns: List[str], score: float) -> str:
        """Generate pattern-based prediction"""
        if not patterns:
            return "No significant patterns detected"
        
        bullish_patterns = ["Hammer", "Hammer-like", "Bullish Engulfing", "Morning Star", "Uptrend", "Trend Reversal Up", "Breakout"]
        bearish_patterns = ["Bearish Engulfing", "Evening Star"]
        
        bullish_count = sum(1 for p in patterns if p in bullish_patterns)
        bearish_count = sum(1 for p in patterns if p in bearish_patterns)
        
        if bullish_count > bearish_count:
            return f"Bullish patterns: {', '.join([p for p in patterns if p in bullish_patterns])}"
        elif bearish_count > bullish_count:
            return f"Bearish patterns: {', '.join([p for p in patterns if p in bearish_patterns])}"
        else:
            return f"Mixed signals: {', '.join(patterns[:2])}"

class MarketRegimeDetector:
    """Market regime analysis"""
    
    def get_current_regime(self) -> Dict:
        """Detect current market regime"""
        try:
            # Get VIX data
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1mo")
            current_vix = vix_data['Close'][-1]
            
            # Get SPY data
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="3mo")
            spy_returns = spy_data['Close'].pct_change().dropna()
            
            # Determine regime
            if current_vix < 15:
                regime_type = "LOW_VOLATILITY"
                confidence = 85
            elif current_vix < 25:
                regime_type = "NORMAL"
                confidence = 70
            else:
                regime_type = "HIGH_VOLATILITY"
                confidence = 80
            
            # Check trend
            recent_performance = (spy_data['Close'][-1] / spy_data['Close'][-20] - 1) * 100
            
            if recent_performance > 5:
                market_trend = "BULL"
            elif recent_performance < -5:
                market_trend = "BEAR"
            else:
                market_trend = "NEUTRAL"
            
            return {
                "type": f"{market_trend}_{regime_type}",
                "confidence": confidence,
                "vix_level": current_vix,
                "market_performance": recent_performance
            }
            
        except Exception as e:
            logger.warning(f"Market regime detection error: {e}")
            return {
                "type": "UNKNOWN",
                "confidence": 50,
                "vix_level": 20,
                "market_performance": 0
            }

class UnifiedStockScanner:
    """Main stock scanner engine"""
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
        self.market_detector = MarketRegimeDetector()
        self.db_path = "stock_scanner.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for storing results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scan_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    total_score REAL,
                    technical_score REAL,
                    fundamental_score REAL,
                    sentiment_score REAL,
                    pattern_score REAL,
                    current_price REAL,
                    volume INTEGER,
                    risk_level TEXT,
                    prediction TEXT
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def scan_stocks(self, settings: ScanSettings) -> List[StockScore]:
        """Main scanning function"""
        logger.info(f"Starting stock scan with settings: {settings}")
        
        # Get stock universe
        if settings.stock_universe == "NASDAQ100":
            symbols = StockUniverse.get_nasdaq100()
        elif settings.stock_universe == "SP500":
            symbols = StockUniverse.get_sp500()
        else:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]  # Default fallback
        
        results = []
        processed = 0
        
        for symbol in symbols:
            try:
                logger.info(f"Processing {symbol} ({processed + 1}/{len(symbols)})")
                
                # Get stock data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="6mo")  # 6 months of data
                
                if len(data) < 30:  # Need sufficient data
                    continue
                
                # Basic filters
                current_price = data['Close'][-1]
                volume = int(data['Volume'][-1])
                
                if not (settings.min_price <= current_price <= settings.max_price):
                    continue
                if volume < settings.min_volume:
                    continue
                
                # Get market cap
                info = ticker.info
                market_cap = info.get('marketCap', 0)
                if market_cap < settings.min_market_cap:
                    continue
                
                # Perform analysis
                technical_score = self.technical_analyzer.analyze(data)
                fundamental_score = self.fundamental_analyzer.analyze(ticker, symbol)
                sentiment_score = self.sentiment_analyzer.analyze(symbol)
                pattern_score, prediction = self.pattern_recognizer.analyze(data)
                
                # Calculate weighted total score
                total_score = (
                    technical_score * settings.technical_weight +
                    fundamental_score * settings.fundamental_weight +
                    sentiment_score * settings.sentiment_weight +
                    pattern_score * settings.pattern_weight
                )
                
                # Determine risk level
                risk_level = self._calculate_risk_level(data, info)
                
                # Create result
                if total_score >= settings.min_score:
                    stock_score = StockScore(
                        symbol=symbol,
                        current_price=current_price,
                        volume=volume,
                        market_cap=market_cap,
                        technical_score=technical_score,
                        fundamental_score=fundamental_score,
                        sentiment_score=sentiment_score,
                        pattern_score=pattern_score,
                        total_score=total_score,
                        risk_level=risk_level,
                        prediction=prediction,
                        timestamp=datetime.now()
                    )
                    results.append(stock_score)
                    
                    # Store in database
                    self._store_result(stock_score)
                
                processed += 1
                
                # Limit results
                if len(results) >= settings.max_results:
                    break
                    
            except Exception as e:
                logger.warning(f"Error processing {symbol}: {e}")
                continue
        
        # Sort by total score (descending)
        results.sort(key=lambda x: x.total_score, reverse=True)
        
        logger.info(f"Scan completed. Found {len(results)} stocks above threshold")
        return results[:settings.max_results]
    
    def _calculate_risk_level(self, data: pd.DataFrame, info: dict) -> str:
        """Calculate risk level for stock"""
        try:
            # Volatility (ATR approximation)
            highs = data['High'].values
            lows = data['Low'].values
            closes = data['Close'].values
            
            if TALIB_AVAILABLE:
                atr = talib.ATR(highs, lows, closes, timeperiod=14)
                volatility = (atr[-1] / closes[-1]) * 100
            else:
                high_low = pd.Series(highs) - pd.Series(lows)
                high_close = np.abs(pd.Series(highs) - pd.Series(closes).shift())
                low_close = np.abs(pd.Series(lows) - pd.Series(closes).shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(14).mean()
                volatility = (atr.iloc[-1] / closes[-1]) * 100
            
            # Beta
            beta = info.get('beta', 1.0)
            
            # Debt to Equity
            debt_to_equity = info.get('debtToEquity', 0)
            
            risk_score = 0
            
            if volatility > 4:
                risk_score += 2
            elif volatility > 2:
                risk_score += 1
            
            if beta > 1.5:
                risk_score += 2
            elif beta > 1.2:
                risk_score += 1
            
            if debt_to_equity > 100:
                risk_score += 2
            elif debt_to_equity > 50:
                risk_score += 1
            
            if risk_score >= 4:
                return "HIGH"
            elif risk_score >= 2:
                return "MEDIUM"
            else:
                return "LOW"
                
        except Exception:
            return "MEDIUM"
    
    def _store_result(self, result: StockScore):
        """Store scan result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO scan_results 
                (timestamp, symbol, total_score, technical_score, fundamental_score, 
                 sentiment_score, pattern_score, current_price, volume, risk_level, prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.timestamp.isoformat(),
                result.symbol,
                result.total_score,
                result.technical_score,
                result.fundamental_score,
                result.sentiment_score,
                result.pattern_score,
                result.current_price,
                result.volume,
                result.risk_level,
                result.prediction
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing result: {e}")
    
    def export_results_to_json(self, results: List[StockScore], filename: str = None) -> str:
        """Export results to JSON"""
        if filename is None:
            filename = f"stock_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "market_regime": self.market_detector.get_current_regime(),
            "results": [
                {
                    "symbol": r.symbol,
                    "total_score": round(r.total_score, 2),
                    "current_price": round(r.current_price, 2),
                    "volume": r.volume,
                    "market_cap": r.market_cap,
                    "technical_score": round(r.technical_score, 1),
                    "fundamental_score": round(r.fundamental_score, 1),
                    "sentiment_score": round(r.sentiment_score, 1),
                    "pattern_score": round(r.pattern_score, 1),
                    "risk_level": r.risk_level,
                    "prediction": r.prediction
                }
                for r in results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results exported to {filename}")
        return filename

def main():
    """Main function for testing"""
    scanner = UnifiedStockScanner()
    
    # Test with default settings
    settings = ScanSettings(
        stock_universe="NASDAQ100",
        min_score=7.5,
        max_results=15,
        min_price=50.0,
        min_volume=500000
    )
    
    print("🚀 Starting AI-Powered Stock Scanner...")
    print(f"⚙️ Settings: {settings.stock_universe}, Min Score: {settings.min_score}")
    
    # Get market regime
    regime = scanner.market_detector.get_current_regime()
    print(f"📊 Market Regime: {regime['type']} ({regime['confidence']}% confidence)")
    
    # Run scan
    results = scanner.scan_stocks(settings)
    
    if results:
        print(f"\n🎯 Found {len(results)} stocks above threshold:\n")
        print("Rank Symbol Score Tech Fund Sent Patt Price    Risk   Prediction")
        print("─" * 80)
        
        for i, stock in enumerate(results, 1):
            print(f"{i:2d}   {stock.symbol:6s} {stock.total_score:4.1f}  "
                  f"{stock.technical_score:4.1f} {stock.fundamental_score:4.1f} "
                  f"{stock.sentiment_score:4.1f} {stock.pattern_score:4.1f} "
                  f"${stock.current_price:7.2f} {stock.risk_level:6s} {stock.prediction[:50]}...")
        
        # Export results
        export_file = scanner.export_results_to_json(results)
        print(f"\n📁 Results exported to: {export_file}")
        
    else:
        print("❌ No stocks found matching criteria")

if __name__ == "__main__":
    main()