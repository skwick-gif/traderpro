#!/usr/bin/env python3
"""
Advanced Data Provider Manager
Multi-source data aggregation with intelligent fallback and rate limiting
"""

import yfinance as yf
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import logging
import json
import os
from abc import ABC, abstractmethod
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class DataProviderConfig:
    """Configuration for data providers"""
    name: str
    api_key: Optional[str] = None
    base_url: str = ""
    rate_limit_per_minute: int = 60
    data_quality_score: float = 8.0
    real_time: bool = False
    supports_fundamentals: bool = True
    supports_options: bool = False
    supports_news: bool = True
    supports_technicals: bool = True
    cost_per_request: float = 0.0
    priority: int = 1  # 1 = highest priority
    enabled: bool = True

@dataclass
class RateLimitTracker:
    """Track API usage and rate limits"""
    requests_made: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    requests_per_minute: int = 60
    
    def can_make_request(self) -> bool:
        now = datetime.now()
        if (now - self.last_reset).total_seconds() >= 60:
            self.requests_made = 0
            self.last_reset = now
        
        return self.requests_made < self.requests_per_minute
    
    def record_request(self):
        self.requests_made += 1

class BaseDataProvider(ABC):
    """Base class for all data providers"""
    
    def __init__(self, config: DataProviderConfig):
        self.config = config
        self.rate_limiter = RateLimitTracker(requests_per_minute=config.rate_limit_per_minute)
        self.session = requests.Session()
        
        # Add API key to headers if needed
        if config.api_key:
            self._setup_authentication()
    
    @abstractmethod
    def _setup_authentication(self):
        """Setup authentication headers"""
        pass
    
    @abstractmethod
    def get_stock_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Get historical stock data"""
        pass
    
    @abstractmethod
    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        """Get fundamental data"""
        pass
    
    @abstractmethod
    def get_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict]]:
        """Get news data"""
        pass
    
    def _make_request(self, url: str, params: Dict = None, timeout: int = 30) -> Optional[Dict]:
        """Make rate-limited API request"""
        if not self.rate_limiter.can_make_request():
            logger.warning(f"{self.config.name}: Rate limit exceeded")
            return None
        
        try:
            self.rate_limiter.record_request()
            response = self.session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"{self.config.name} request error: {e}")
            return None

class TwelveDataProvider(BaseDataProvider):
    """Twelve Data API provider"""
    
    def _setup_authentication(self):
        self.session.headers.update({
            'Authorization': f'apikey {self.config.api_key}'
        })
    
    def get_stock_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        try:
            # Convert period to Twelve Data format
            interval_map = {
                "1d": "1day", "5d": "1day", "1mo": "1day",
                "3mo": "1day", "6mo": "1day", "1y": "1day", "2y": "1week"
            }
            interval = interval_map.get(period, "1day")
            
            # Calculate outputsize based on period
            outputsize_map = {
                "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730
            }
            outputsize = outputsize_map.get(period, 180)
            
            url = f"{self.config.base_url}/time_series"
            params = {
                'symbol': symbol,
                'interval': interval,
                'outputsize': min(outputsize, 5000),  # Max allowed
                'format': 'JSON'
            }
            
            data = self._make_request(url, params)
            if not data or 'values' not in data:
                return None
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df.sort_index()
            
            # Convert to standard format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            
            return df
            
        except Exception as e:
            logger.error(f"Twelve Data error for {symbol}: {e}")
            return None
    
    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        try:
            # Get company overview
            url = f"{self.config.base_url}/profile"
            params = {'symbol': symbol}
            profile_data = self._make_request(url, params)
            
            # Get key statistics
            url = f"{self.config.base_url}/statistics"
            stats_data = self._make_request(url, params)
            
            if not profile_data or not stats_data:
                return None
            
            # Combine and normalize data
            fundamentals = {
                'marketCap': stats_data.get('market_capitalization'),
                'trailingPE': stats_data.get('pe_ratio'),
                'forwardPE': stats_data.get('forward_pe'),
                'priceToBook': stats_data.get('price_to_book'),
                'debtToEquity': stats_data.get('debt_to_equity'),
                'returnOnEquity': stats_data.get('return_on_equity'),
                'returnOnAssets': stats_data.get('return_on_assets'),
                'profitMargins': stats_data.get('profit_margin'),
                'revenueGrowth': stats_data.get('revenue_growth'),
                'earningsGrowth': stats_data.get('earnings_growth'),
                'currentRatio': stats_data.get('current_ratio'),
                'beta': stats_data.get('beta')
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Twelve Data fundamentals error for {symbol}: {e}")
            return None
    
    def get_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict]]:
        try:
            url = f"{self.config.base_url}/news"
            params = {
                'symbol': symbol,
                'limit': min(limit, 50)
            }
            
            data = self._make_request(url, params)
            if not data or 'feed' not in data:
                return None
            
            news_items = []
            for item in data['feed'][:limit]:
                news_items.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'published': item.get('time_published', ''),
                    'source': item.get('source', 'Twelve Data')
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Twelve Data news error for {symbol}: {e}")
            return None

class PolygonProvider(BaseDataProvider):
    """Polygon.io API provider"""
    
    def _setup_authentication(self):
        # Polygon uses API key as query parameter
        pass
    
    def get_stock_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        try:
            # Calculate date range
            end_date = datetime.now()
            period_map = {
                "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
                "6mo": 180, "1y": 365, "2y": 730
            }
            days_back = period_map.get(period, 180)
            start_date = end_date - timedelta(days=days_back)
            
            url = f"{self.config.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {
                'apikey': self.config.api_key,
                'adjusted': 'true',
                'sort': 'asc'
            }
            
            data = self._make_request(url, params)
            if not data or 'results' not in data:
                return None
            
            # Convert to DataFrame
            results = data['results']
            df_data = []
            for result in results:
                df_data.append({
                    'Date': pd.to_datetime(result['t'], unit='ms'),
                    'Open': result['o'],
                    'High': result['h'],
                    'Low': result['l'],
                    'Close': result['c'],
                    'Volume': result['v']
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('Date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Polygon error for {symbol}: {e}")
            return None
    
    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        try:
            # Get company details
            url = f"{self.config.base_url}/v3/reference/tickers/{symbol}"
            params = {'apikey': self.config.api_key}
            
            data = self._make_request(url, params)
            if not data or 'results' not in data:
                return None
            
            result = data['results']
            fundamentals = {
                'marketCap': result.get('market_cap'),
                'shareClassSharesOutstanding': result.get('share_class_shares_outstanding'),
                'weightedSharesOutstanding': result.get('weighted_shares_outstanding'),
                'totalEmployees': result.get('total_employees')
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Polygon fundamentals error for {symbol}: {e}")
            return None
    
    def get_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict]]:
        try:
            url = f"{self.config.base_url}/v2/reference/news"
            params = {
                'ticker': symbol,
                'limit': min(limit, 50),
                'apikey': self.config.api_key
            }
            
            data = self._make_request(url, params)
            if not data or 'results' not in data:
                return None
            
            news_items = []
            for item in data['results'][:limit]:
                news_items.append({
                    'title': item.get('title', ''),
                    'summary': item.get('description', ''),
                    'url': item.get('article_url', ''),
                    'published': item.get('published_utc', ''),
                    'source': item.get('publisher', {}).get('name', 'Polygon')
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Polygon news error for {symbol}: {e}")
            return None

class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage API provider"""
    
    def _setup_authentication(self):
        # Alpha Vantage uses API key as query parameter
        pass
    
    def get_stock_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        try:
            # Use daily data for all periods
            url = self.config.base_url
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'apikey': self.config.api_key,
                'outputsize': 'full' if period in ['1y', '2y'] else 'compact'
            }
            
            data = self._make_request(url, params)
            if not data or 'Time Series (Daily)' not in data:
                return None
            
            # Convert to DataFrame
            ts_data = data['Time Series (Daily)']
            df = pd.DataFrame.from_dict(ts_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns to standard format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividend', 'Split']
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            
            # Filter by period
            period_map = {
                "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
                "6mo": 180, "1y": 365, "2y": 730
            }
            days_back = period_map.get(period, 180)
            cutoff_date = datetime.now() - timedelta(days=days_back)
            df = df[df.index >= cutoff_date]
            
            return df
            
        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
            return None
    
    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        try:
            # Get company overview
            url = self.config.base_url
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.config.api_key
            }
            
            data = self._make_request(url, params)
            if not data:
                return None
            
            # Convert string values to appropriate types
            def safe_float(value):
                try:
                    return float(value) if value and value != 'None' else None
                except:
                    return None
            
            fundamentals = {
                'marketCap': safe_float(data.get('MarketCapitalization')),
                'trailingPE': safe_float(data.get('PERatio')),
                'forwardPE': safe_float(data.get('ForwardPE')),
                'priceToBook': safe_float(data.get('PriceToBookRatio')),
                'pegRatio': safe_float(data.get('PEGRatio')),
                'debtToEquity': safe_float(data.get('DebtToEquityRatio')),
                'returnOnEquity': safe_float(data.get('ReturnOnEquityTTM')),
                'returnOnAssets': safe_float(data.get('ReturnOnAssetsTTM')),
                'profitMargins': safe_float(data.get('ProfitMargin')),
                'revenueGrowth': safe_float(data.get('QuarterlyRevenueGrowthYOY')),
                'earningsGrowth': safe_float(data.get('QuarterlyEarningsGrowthYOY')),
                'currentRatio': safe_float(data.get('CurrentRatio')),
                'beta': safe_float(data.get('Beta')),
                'dividendYield': safe_float(data.get('DividendYield')),
                '52WeekHigh': safe_float(data.get('52WeekHigh')),
                '52WeekLow': safe_float(data.get('52WeekLow')),
                'bookValue': safe_float(data.get('BookValue')),
                'ebitda': safe_float(data.get('EBITDA')),
                'revenuePerShare': safe_float(data.get('RevenuePerShareTTM'))
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Alpha Vantage fundamentals error for {symbol}: {e}")
            return None
    
    def get_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict]]:
        try:
            url = self.config.base_url
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'limit': min(limit, 50),
                'apikey': self.config.api_key
            }
            
            data = self._make_request(url, params)
            if not data or 'feed' not in data:
                return None
            
            news_items = []
            for item in data['feed'][:limit]:
                news_items.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'published': item.get('time_published', ''),
                    'source': item.get('source', 'Alpha Vantage'),
                    'sentiment_score': item.get('overall_sentiment_score', 0),
                    'sentiment_label': item.get('overall_sentiment_label', 'Neutral')
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Alpha Vantage news error for {symbol}: {e}")
            return None

class FinnhubProvider(BaseDataProvider):
    """Finnhub API provider"""
    
    def _setup_authentication(self):
        self.session.headers.update({
            'X-Finnhub-Token': self.config.api_key
        })
    
    def get_stock_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        try:
            # Calculate timestamps
            end_time = int(datetime.now().timestamp())
            period_map = {
                "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
                "6mo": 180, "1y": 365, "2y": 730
            }
            days_back = period_map.get(period, 180)
            start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
            
            url = f"{self.config.base_url}/stock/candle"
            params = {
                'symbol': symbol,
                'resolution': 'D',
                'from': start_time,
                'to': end_time
            }
            
            data = self._make_request(url, params)
            if not data or data.get('s') != 'ok':
                return None
            
            # Convert to DataFrame
            df_data = {
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            }
            
            df = pd.DataFrame(df_data)
            df.index = pd.to_datetime(data['t'], unit='s')
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            logger.error(f"Finnhub error for {symbol}: {e}")
            return None
    
    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        try:
            # Get company profile
            url = f"{self.config.base_url}/stock/profile2"
            params = {'symbol': symbol}
            profile_data = self._make_request(url, params)
            
            # Get basic financials
            url = f"{self.config.base_url}/stock/metric"
            params = {'symbol': symbol, 'metric': 'all'}
            metrics_data = self._make_request(url, params)
            
            if not profile_data or not metrics_data:
                return None
            
            # Extract metrics
            metrics = metrics_data.get('metric', {})
            fundamentals = {
                'marketCap': profile_data.get('marketCapitalization'),
                'trailingPE': metrics.get('peBasicExclExtraTTM'),
                'priceToBook': metrics.get('pbQuarterly'),
                'returnOnEquity': metrics.get('roeTTM'),
                'returnOnAssets': metrics.get('roaTTM'),
                'debtToEquity': metrics.get('totalDebt/totalEquityQuarterly'),
                'currentRatio': metrics.get('currentRatioQuarterly'),
                'beta': metrics.get('beta'),
                '52WeekHigh': metrics.get('52WeekHigh'),
                '52WeekLow': metrics.get('52WeekLow')
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Finnhub fundamentals error for {symbol}: {e}")
            return None
    
    def get_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict]]:
        try:
            # Get news from last 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            url = f"{self.config.base_url}/company-news"
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }
            
            data = self._make_request(url, params)
            if not data:
                return None
            
            news_items = []
            for item in data[:limit]:
                news_items.append({
                    'title': item.get('headline', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('url', ''),
                    'published': datetime.fromtimestamp(item.get('datetime', 0)).isoformat(),
                    'source': item.get('source', 'Finnhub'),
                    'category': item.get('category', ''),
                    'image': item.get('image', '')
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Finnhub news error for {symbol}: {e}")
            return None

class FMPProvider(BaseDataProvider):
    """Financial Modeling Prep API provider"""
    
    def _setup_authentication(self):
        # FMP uses API key as query parameter
        pass
    
    def get_stock_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        try:
            url = f"{self.config.base_url}/historical-price-full/{symbol}"
            params = {
                'apikey': self.config.api_key
            }
            
            data = self._make_request(url, params)
            if not data or 'historical' not in data:
                return None
            
            # Convert to DataFrame
            historical = data['historical']
            df = pd.DataFrame(historical)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            # Rename columns to standard format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Filter by period
            period_map = {
                "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
                "6mo": 180, "1y": 365, "2y": 730
            }
            days_back = period_map.get(period, 180)
            cutoff_date = datetime.now() - timedelta(days=days_back)
            df = df[df.index >= cutoff_date]
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            logger.error(f"FMP error for {symbol}: {e}")
            return None
    
    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        try:
            # Get company profile
            url = f"{self.config.base_url}/profile/{symbol}"
            params = {'apikey': self.config.api_key}
            profile_data = self._make_request(url, params)
            
            # Get key metrics
            url = f"{self.config.base_url}/key-metrics-ttm/{symbol}"
            metrics_data = self._make_request(url, params)
            
            # Get ratios
            url = f"{self.config.base_url}/ratios-ttm/{symbol}"
            ratios_data = self._make_request(url, params)
            
            if not profile_data or not metrics_data or not ratios_data:
                return None
            
            profile = profile_data[0] if profile_data else {}
            metrics = metrics_data[0] if metrics_data else {}
            ratios = ratios_data[0] if ratios_data else {}
            
            fundamentals = {
                'marketCap': profile.get('mktCap'),
                'trailingPE': ratios.get('peRatioTTM'),
                'priceToBook': ratios.get('priceToBookRatioTTM'),
                'pegRatio': ratios.get('pegRatioTTM'),
                'debtToEquity': ratios.get('debtEquityRatioTTM'),
                'returnOnEquity': ratios.get('returnOnEquityTTM'),
                'returnOnAssets': ratios.get('returnOnAssetsTTM'),
                'currentRatio': ratios.get('currentRatioTTM'),
                'beta': profile.get('beta'),
                'dividendYield': ratios.get('dividendYieldTTM'),
                'freeCashflow': metrics.get('freeCashFlowTTM'),
                'revenue': metrics.get('revenueTTM'),
                'netIncome': metrics.get('netIncomeTTM'),
                'bookValue': metrics.get('bookValuePerShareTTM')
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"FMP fundamentals error for {symbol}: {e}")
            return None
    
    def get_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict]]:
        try:
            url = f"{self.config.base_url}/stock_news"
            params = {
                'tickers': symbol,
                'limit': min(limit, 50),
                'apikey': self.config.api_key
            }
            
            data = self._make_request(url, params)
            if not data:
                return None
            
            news_items = []
            for item in data[:limit]:
                news_items.append({
                    'title': item.get('title', ''),
                    'summary': item.get('text', ''),
                    'url': item.get('url', ''),
                    'published': item.get('publishedDate', ''),
                    'source': item.get('site', 'FMP'),
                    'symbol': item.get('symbol', symbol)
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"FMP news error for {symbol}: {e}")
            return None

class YahooFinanceProvider(BaseDataProvider):
    """Yahoo Finance provider (free, no API key needed)"""
    
    def _setup_authentication(self):
        # No authentication needed for Yahoo Finance
        pass
    
    def get_stock_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                return None
            
            # Yahoo Finance already provides the right format
            return data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return None
    
    def get_fundamentals(self, symbol: str) -> Optional[Dict]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            return info  # Yahoo Finance provides comprehensive fundamental data
            
        except Exception as e:
            logger.error(f"Yahoo Finance fundamentals error for {symbol}: {e}")
            return None
    
    def get_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict]]:
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                return None
            
            news_items = []
            for item in news[:limit]:
                news_items.append({
                    'title': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'url': item.get('link', ''),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat(),
                    'source': item.get('publisher', 'Yahoo Finance'),
                    'type': item.get('type', 'STORY')
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Yahoo Finance news error for {symbol}: {e}")
            return None

class DataProviderManager:
    """Intelligent data provider manager with fallback and aggregation"""
    
    def __init__(self, config_file: str = "data_providers_config.json"):
        self.providers: Dict[str, BaseDataProvider] = {}
        self.config_file = config_file
        self.load_configuration()
        self.setup_providers()
    
    def load_configuration(self):
        """Load provider configuration from file"""
        default_config = {
            "twelve_data": {
                "name": "Twelve Data",
                "api_key": os.getenv("TWELVE_DATA_API_KEY"),
                "base_url": "https://api.twelvedata.com",
                "rate_limit_per_minute": 55,
                "data_quality_score": 9.5,
                "real_time": True,
                "priority": 1,
                "enabled": True
            },
            "polygon": {
                "name": "Polygon",
                "api_key": os.getenv("POLYGON_API_KEY"),
                "base_url": "https://api.polygon.io",
                "rate_limit_per_minute": 5,
                "data_quality_score": 9.0,
                "real_time": True,
                "priority": 2,
                "enabled": True
            },
            "alpha_vantage": {
                "name": "Alpha Vantage",
                "api_key": os.getenv("ALPHA_VANTAGE_API_KEY"),
                "base_url": "https://www.alphavantage.co/query",
                "rate_limit_per_minute": 5,
                "data_quality_score": 8.5,
                "real_time": False,
                "priority": 3,
                "enabled": True
            },
            "finnhub": {
                "name": "Finnhub",
                "api_key": os.getenv("FINNHUB_API_KEY"),
                "base_url": "https://finnhub.io/api/v1",
                "rate_limit_per_minute": 60,
                "data_quality_score": 8.0,
                "real_time": True,
                "priority": 4,
                "enabled": True
            },
            "fmp": {
                "name": "Financial Modeling Prep",
                "api_key": os.getenv("FMP_API_KEY"),
                "base_url": "https://financialmodelingprep.com/api/v3",
                "rate_limit_per_minute": 250,
                "data_quality_score": 8.5,
                "real_time": False,
                "priority": 5,
                "enabled": True
            },
            "yahoo_finance": {
                "name": "Yahoo Finance",
                "api_key": None,
                "base_url": "",
                "rate_limit_per_minute": 2000,
                "data_quality_score": 7.0,
                "real_time": False,
                "priority": 6,
                "enabled": True
            }
        }
        
        # Try to load from file
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading config file: {e}, using defaults")
                self.config = default_config
        else:
            self.config = default_config
            self.save_configuration()
    
    def save_configuration(self):
        """Save current configuration to file"""
        try:
            # Don't save API keys to file for security
            safe_config = {}
            for provider_name, config in self.config.items():
                safe_config[provider_name] = {k: v for k, v in config.items() if k != 'api_key'}
            
            with open(self.config_file, 'w') as f:
                json.dump(safe_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def setup_providers(self):
        """Initialize all enabled providers"""
        provider_classes = {
            "twelve_data": TwelveDataProvider,
            "polygon": PolygonProvider,
            "alpha_vantage": AlphaVantageProvider,
            "finnhub": FinnhubProvider,
            "fmp": FMPProvider,
            "yahoo_finance": YahooFinanceProvider
        }
        
        for provider_name, config_data in self.config.items():
            if not config_data.get("enabled", True):
                continue
            
            if provider_name not in provider_classes:
                continue
            
            # Skip if API key required but not provided
            if config_data.get("api_key") is None and provider_name != "yahoo_finance":
                logger.warning(f"Skipping {provider_name}: No API key provided")
                continue
            
            try:
                provider_config = DataProviderConfig(**config_data)
                provider_class = provider_classes[provider_name]
                self.providers[provider_name] = provider_class(provider_config)
                logger.info(f"Initialized {provider_name} provider")
            except Exception as e:
                logger.error(f"Error initializing {provider_name}: {e}")
    
    def get_sorted_providers(self, data_type: str = "stock_data") -> List[Tuple[str, BaseDataProvider]]:
        """Get providers sorted by priority and capability"""
        available_providers = []
        
        for name, provider in self.providers.items():
            config = provider.config
            
            # Check if provider supports the requested data type
            if data_type == "fundamentals" and not config.supports_fundamentals:
                continue
            elif data_type == "news" and not config.supports_news:
                continue
            elif data_type == "options" and not config.supports_options:
                continue
            
            # Check if provider can make requests
            if provider.rate_limiter.can_make_request():
                available_providers.append((name, provider))
        
        # Sort by priority (lower number = higher priority)
        available_providers.sort(key=lambda x: x[1].config.priority)
        return available_providers
    
    def get_stock_data(self, symbol: str, period: str = "6mo", max_attempts: int = 3) -> Optional[pd.DataFrame]:
        """Get stock data with intelligent fallback"""
        providers = self.get_sorted_providers("stock_data")
        
        for attempt, (provider_name, provider) in enumerate(providers[:max_attempts]):
            try:
                logger.info(f"Attempting to get {symbol} data from {provider_name}")
                data = provider.get_stock_data(symbol, period)
                
                if data is not None and not data.empty:
                    logger.info(f"Successfully got {symbol} data from {provider_name}")
                    return self._validate_and_clean_data(data, symbol, provider_name)
                
            except Exception as e:
                logger.warning(f"Error getting {symbol} data from {provider_name}: {e}")
                continue
        
        logger.error(f"Failed to get {symbol} data from all providers")
        return None
    
    def get_fundamentals(self, symbol: str, max_attempts: int = 3) -> Optional[Dict]:
        """Get fundamental data with intelligent fallback"""
        providers = self.get_sorted_providers("fundamentals")
        
        for attempt, (provider_name, provider) in enumerate(providers[:max_attempts]):
            try:
                logger.info(f"Attempting to get {symbol} fundamentals from {provider_name}")
                data = provider.get_fundamentals(symbol)
                
                if data is not None:
                    logger.info(f"Successfully got {symbol} fundamentals from {provider_name}")
                    return self._validate_fundamentals(data, symbol, provider_name)
                
            except Exception as e:
                logger.warning(f"Error getting {symbol} fundamentals from {provider_name}: {e}")
                continue
        
        logger.warning(f"Failed to get {symbol} fundamentals from all providers")
        return {}
    
    def get_news(self, symbol: str, limit: int = 10, max_attempts: int = 2) -> List[Dict]:
        """Get news data with intelligent fallback"""
        providers = self.get_sorted_providers("news")
        all_news = []
        
        for attempt, (provider_name, provider) in enumerate(providers[:max_attempts]):
            try:
                logger.info(f"Attempting to get {symbol} news from {provider_name}")
                news = provider.get_news(symbol, limit)
                
                if news:
                    logger.info(f"Successfully got {len(news)} news items from {provider_name}")
                    all_news.extend(news)
                
            except Exception as e:
                logger.warning(f"Error getting {symbol} news from {provider_name}: {e}")
                continue
        
        # Remove duplicates and sort by date
        unique_news = self._deduplicate_news(all_news)
        return sorted(unique_news, key=lambda x: x.get('published', ''), reverse=True)[:limit]
    
    def get_aggregated_fundamentals(self, symbol: str) -> Dict:
        """Get fundamentals from multiple sources and aggregate"""
        all_fundamentals = {}
        providers = self.get_sorted_providers("fundamentals")
        
        for provider_name, provider in providers:
            try:
                data = provider.get_fundamentals(symbol)
                if data:
                    all_fundamentals[provider_name] = data
            except Exception as e:
                logger.warning(f"Error getting fundamentals from {provider_name}: {e}")
                continue
        
        if not all_fundamentals:
            return {}
        
        # Aggregate the data intelligently
        return self._aggregate_fundamental_data(all_fundamentals)
    
    def _validate_and_clean_data(self, data: pd.DataFrame, symbol: str, provider: str) -> pd.DataFrame:
        """Validate and clean stock data"""
        try:
            # Check for required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            # Remove rows with all NaN values
            data = data.dropna(how='all')
            
            # Forward fill missing values
            data = data.fillna(method='ffill')
            
            # Remove unrealistic values
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                data = data[data[col] > 0]  # Remove negative prices
                data = data[data[col] < data[col].quantile(0.99) * 10]  # Remove extreme outliers
            
            # Validate OHLC relationships
            valid_ohlc = (
                (data['High'] >= data['Open']) & 
                (data['High'] >= data['Close']) & 
                (data['Low'] <= data['Open']) & 
                (data['Low'] <= data['Close'])
            )
            data = data[valid_ohlc]
            
            # Add metadata
            data.attrs = {
                'symbol': symbol,
                'provider': provider,
                'last_updated': datetime.now().isoformat()
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error validating data for {symbol}: {e}")
            return data  # Return original data if validation fails
    
    def _validate_fundamentals(self, data: Dict, symbol: str, provider: str) -> Dict:
        """Validate and clean fundamental data"""
        cleaned_data = {}
        
        for key, value in data.items():
            if value is None:
                continue
            
            # Convert string percentages to floats
            if isinstance(value, str):
                if value.endswith('%'):
                    try:
                        cleaned_data[key] = float(value.rstrip('%')) / 100
                        continue
                    except:
                        pass
                
                # Try to convert to number
                try:
                    # Handle values like "1.5B", "2.3M", etc.
                    if value.endswith('B'):
                        cleaned_data[key] = float(value.rstrip('B')) * 1e9
                    elif value.endswith('M'):
                        cleaned_data[key] = float(value.rstrip('M')) * 1e6
                    elif value.endswith('K'):
                        cleaned_data[key] = float(value.rstrip('K')) * 1e3
                    else:
                        cleaned_data[key] = float(value)
                except:
                    cleaned_data[key] = value  # Keep as string if conversion fails
            else:
                cleaned_data[key] = value
        
        # Add metadata
        cleaned_data['_metadata'] = {
            'symbol': symbol,
            'provider': provider,
            'last_updated': datetime.now().isoformat()
        }
        
        return cleaned_data
    
    def _deduplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """Remove duplicate news items"""
        seen_titles = set()
        unique_news = []
        
        for item in news_list:
            title = item.get('title', '').lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(item)
        
        return unique_news
    
    def _aggregate_fundamental_data(self, all_fundamentals: Dict) -> Dict:
        """Intelligently aggregate fundamental data from multiple sources"""
        aggregated = {}
        
        # Get all unique keys
        all_keys = set()
        for provider_data in all_fundamentals.values():
            all_keys.update(provider_data.keys())
        
        for key in all_keys:
            if key.startswith('_'):  # Skip metadata keys
                continue
            
            values = []
            sources = []
            
            for provider, data in all_fundamentals.items():
                if key in data and data[key] is not None:
                    try:
                        val = float(data[key])
                        if not (np.isnan(val) or np.isinf(val)):
                            values.append(val)
                            sources.append(provider)
                    except:
                        continue
            
            if values:
                # Use weighted average based on provider quality scores
                weights = [self.providers[source].config.data_quality_score for source in sources]
                aggregated[key] = np.average(values, weights=weights)
                aggregated[f'{key}_sources'] = sources
                aggregated[f'{key}_count'] = len(values)
        
        # Add aggregation metadata
        aggregated['_aggregation_info'] = {
            'providers_used': list(all_fundamentals.keys()),
            'aggregation_timestamp': datetime.now().isoformat()
        }
        
        return aggregated
    
    def get_provider_status(self) -> Dict:
        """Get status of all providers"""
        status = {}
        
        for name, provider in self.providers.items():
            config = provider.config
            rate_limiter = provider.rate_limiter
            
            status[name] = {
                'enabled': config.enabled,
                'priority': config.priority,
                'data_quality_score': config.data_quality_score,
                'real_time': config.real_time,
                'requests_made': rate_limiter.requests_made,
                'requests_remaining': rate_limiter.requests_per_minute - rate_limiter.requests_made,
                'can_make_request': rate_limiter.can_make_request(),
                'last_reset': rate_limiter.last_reset.isoformat(),
                'api_key_configured': config.api_key is not None
            }
        
        return status
    
    def update_provider_config(self, provider_name: str, updates: Dict):
        """Update provider configuration"""
        if provider_name in self.config:
            self.config[provider_name].update(updates)
            self.save_configuration()
            
            # Reinitialize provider if enabled
            if updates.get('enabled', True):
                self.setup_providers()

def main():
    """Test the data provider manager"""
    # Setup environment variables for testing
    os.environ.setdefault("TWELVE_DATA_API_KEY", "your_key_here")
    os.environ.setdefault("POLYGON_API_KEY", "your_key_here")
    os.environ.setdefault("ALPHA_VANTAGE_API_KEY", "your_key_here")
    os.environ.setdefault("FINNHUB_API_KEY", "your_key_here")
    os.environ.setdefault("FMP_API_KEY", "your_key_here")
    
    # Initialize manager
    manager = DataProviderManager()
    
    # Test symbol
    symbol = "AAPL"
    
    print(f"🚀 Testing Data Provider Manager for {symbol}")
    print("=" * 50)
    
    # Test provider status
    print("\n📊 Provider Status:")
    status = manager.get_provider_status()
    for provider, info in status.items():
        print(f"  {provider}: {'✅' if info['enabled'] else '❌'} "
              f"(Quality: {info['data_quality_score']}, "
              f"Requests: {info['requests_made']}/{info['requests_made'] + info['requests_remaining']})")
    
    # Test stock data
    print(f"\n📈 Getting stock data for {symbol}...")
    stock_data = manager.get_stock_data(symbol, "3mo")
    if stock_data is not None:
        print(f"  ✅ Got {len(stock_data)} days of data")
        print(f"  📅 Date range: {stock_data.index[0]} to {stock_data.index[-1]}")
        print(f"  💰 Current price: ${stock_data['Close'][-1]:.2f}")
    else:
        print("  ❌ Failed to get stock data")
    
    # Test fundamentals
    print(f"\n📊 Getting fundamentals for {symbol}...")
    fundamentals = manager.get_aggregated_fundamentals(symbol)
    if fundamentals:
        print(f"  ✅ Got {len(fundamentals)} fundamental metrics")
        for key, value in list(fundamentals.items())[:5]:
            if not key.startswith('_'):
                print(f"    {key}: {value}")
    else:
        print("  ❌ Failed to get fundamentals")
    
    # Test news
    print(f"\n📰 Getting news for {symbol}...")
    news = manager.get_news(symbol, limit=5)
    if news:
        print(f"  ✅ Got {len(news)} news items")
        for item in news[:2]:
            print(f"    📰 {item['title'][:60]}...")
    else:
        print("  ❌ Failed to get news")
    
    print("\n✨ Test completed!")

if __name__ == "__main__":
    main()