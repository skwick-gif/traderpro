#!/usr/bin/env python3
"""
Watchlist Backend API
Advanced watchlist management with real-time updates and multi-provider data
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
import threading
import time
from dataclasses import dataclass, asdict

from data_provider_manager import DataProviderManager

logger = logging.getLogger(__name__)

# Pydantic models
class WatchlistItem(BaseModel):
    symbol: str
    notes: Optional[str] = ""
    isFavorite: bool = False
    hasAlerts: bool = False
    addedDate: str
    alerts: List[Dict] = []

class AlertConfig(BaseModel):
    type: str  # price_above, price_below, volume_spike, percent_change
    value: float
    email_notification: bool = False
    enabled: bool = True

class WatchlistResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict] = None

@dataclass
class WatchlistItemData:
    """Complete watchlist item with market data"""
    symbol: str
    price: float
    change: float
    changePercent: float
    volume: int
    marketCap: Optional[float] = None
    pe: Optional[float] = None
    dividendYield: Optional[float] = None
    beta: Optional[float] = None
    high52w: Optional[float] = None
    low52w: Optional[float] = None
    avgVolume: Optional[int] = None
    rsi: Optional[float] = None
    macd: Optional[str] = None
    ema20: Optional[float] = None
    ema50: Optional[float] = None
    score: Optional[float] = None
    alerts: List[Dict] = None
    notes: str = ""
    isFavorite: bool = False
    hasAlerts: bool = False
    addedDate: str = ""
    lastUpdated: str = ""

class WatchlistManager:
    """Manage watchlist data and real-time updates"""
    
    def __init__(self, db_path: str = "watchlist.db"):
        self.db_path = db_path
        self.data_provider = DataProviderManager()
        self.watchlist_items: Dict[str, WatchlistItemData] = {}
        self.alerts: Dict[str, List[AlertConfig]] = {}
        self.subscribers: List[WebSocket] = []
        self.price_subscribers: List[WebSocket] = []
        self.update_thread = None
        self.running = False
        
        self._init_database()
        self._load_watchlist()
        self.start_price_updates()
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Watchlist table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS watchlist (
                    symbol TEXT PRIMARY KEY,
                    notes TEXT DEFAULT '',
                    is_favorite BOOLEAN DEFAULT FALSE,
                    has_alerts BOOLEAN DEFAULT FALSE,
                    added_date TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    alert_type TEXT,
                    target_value REAL,
                    email_notification BOOLEAN DEFAULT FALSE,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    triggered_at TIMESTAMP NULL,
                    FOREIGN KEY (symbol) REFERENCES watchlist (symbol)
                )
            ''')
            
            # Price history table for analytics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    price REAL,
                    volume INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol) REFERENCES watchlist (symbol)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def _load_watchlist(self):
        """Load watchlist from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT symbol, notes, is_favorite, has_alerts, added_date
                FROM watchlist ORDER BY added_date DESC
            ''')
            
            for row in cursor.fetchall():
                symbol, notes, is_favorite, has_alerts, added_date = row
                
                # Load alerts for this symbol
                cursor.execute('''
                    SELECT alert_type, target_value, email_notification, enabled
                    FROM alerts WHERE symbol = ? AND enabled = TRUE
                ''', (symbol,))
                
                alerts = []
                for alert_row in cursor.fetchall():
                    alerts.append({
                        'type': alert_row[0],
                        'value': alert_row[1],
                        'email_notification': alert_row[2],
                        'enabled': alert_row[3]
                    })
                
                # Create item with basic data
                item = WatchlistItemData(
                    symbol=symbol,
                    price=0.0,
                    change=0.0,
                    changePercent=0.0,
                    volume=0,
                    notes=notes or "",
                    isFavorite=bool(is_favorite),
                    hasAlerts=bool(has_alerts),
                    addedDate=added_date,
                    alerts=alerts,
                    lastUpdated=""
                )
                
                self.watchlist_items[symbol] = item
            
            conn.close()
            logger.info(f"Loaded {len(self.watchlist_items)} watchlist items")
            
        except Exception as e:
            logger.error(f"Error loading watchlist: {e}")
    
    def add_symbol(self, symbol: str, notes: str = "", added_date: str = None) -> bool:
        """Add symbol to watchlist"""
        try:
            if added_date is None:
                added_date = datetime.now().isoformat()
            
            # Add to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO watchlist 
                (symbol, notes, added_date, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (symbol, notes, added_date))
            
            conn.commit()
            conn.close()
            
            # Create item data
            item = WatchlistItemData(
                symbol=symbol,
                price=0.0,
                change=0.0,
                changePercent=0.0,
                volume=0,
                notes=notes,
                isFavorite=False,
                hasAlerts=False,
                addedDate=added_date,
                alerts=[],
                lastUpdated=""
            )
            
            self.watchlist_items[symbol] = item
            
            # Fetch initial data
            asyncio.create_task(self._update_symbol_data(symbol))
            
            logger.info(f"Added {symbol} to watchlist")
            return True
            
        except Exception as e:
            logger.error(f"Error adding {symbol} to watchlist: {e}")
            return False
    
    def remove_symbol(self, symbol: str) -> bool:
        """Remove symbol from watchlist"""
        try:
            # Remove from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM alerts WHERE symbol = ?', (symbol,))
            cursor.execute('DELETE FROM watchlist WHERE symbol = ?', (symbol,))
            cursor.execute('DELETE FROM price_history WHERE symbol = ?', (symbol,))
            
            conn.commit()
            conn.close()
            
            # Remove from memory
            if symbol in self.watchlist_items:
                del self.watchlist_items[symbol]
            
            logger.info(f"Removed {symbol} from watchlist")
            return True
            
        except Exception as e:
            logger.error(f"Error removing {symbol} from watchlist: {e}")
            return False
    
    def toggle_favorite(self, symbol: str) -> bool:
        """Toggle favorite status"""
        try:
            if symbol not in self.watchlist_items:
                return False
            
            new_status = not self.watchlist_items[symbol].isFavorite
            
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE watchlist SET is_favorite = ?, updated_at = CURRENT_TIMESTAMP
                WHERE symbol = ?
            ''', (new_status, symbol))
            
            conn.commit()
            conn.close()
            
            # Update memory
            self.watchlist_items[symbol].isFavorite = new_status
            
            return True
            
        except Exception as e:
            logger.error(f"Error toggling favorite for {symbol}: {e}")
            return False
    
    def add_alert(self, symbol: str, alert_config: AlertConfig) -> bool:
        """Add alert for symbol"""
        try:
            if symbol not in self.watchlist_items:
                return False
            
            # Add to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts 
                (symbol, alert_type, target_value, email_notification, enabled)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                symbol, 
                alert_config.type, 
                alert_config.value,
                alert_config.email_notification,
                alert_config.enabled
            ))
            
            # Update watchlist has_alerts flag
            cursor.execute('''
                UPDATE watchlist SET has_alerts = TRUE, updated_at = CURRENT_TIMESTAMP
                WHERE symbol = ?
            ''', (symbol,))
            
            conn.commit()
            conn.close()
            
            # Update memory
            self.watchlist_items[symbol].hasAlerts = True
            self.watchlist_items[symbol].alerts.append(asdict(alert_config))
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding alert for {symbol}: {e}")
            return False
    
    async def _update_symbol_data(self, symbol: str):
        """Update data for a single symbol"""
        try:
            if symbol not in self.watchlist_items:
                return
            
            # Get price data
            stock_data = self.data_provider.get_stock_data(symbol, "5d")
            if stock_data is None or stock_data.empty:
                logger.warning(f"No stock data for {symbol}")
                return
            
            # Calculate current metrics
            current_price = stock_data['Close'].iloc[-1]
            prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
            change = current_price - prev_price
            change_percent = (change / prev_price) * 100 if prev_price > 0 else 0
            current_volume = int(stock_data['Volume'].iloc[-1])
            
            # Get fundamentals
            fundamentals = self.data_provider.get_fundamentals(symbol)
            
            # Technical indicators
            closes = stock_data['Close'].values
            ema_20 = None
            ema_50 = None
            rsi = None
            
            try:
                import talib
                if len(closes) >= 20:
                    ema_20 = talib.EMA(closes, timeperiod=20)[-1]
                if len(closes) >= 50:
                    ema_50 = talib.EMA(closes, timeperiod=50)[-1]
                if len(closes) >= 14:
                    rsi = talib.RSI(closes, timeperiod=14)[-1]
            except ImportError:
                logger.warning("TA-Lib not available for technical indicators")
            
            # Update item
            item = self.watchlist_items[symbol]
            item.price = float(current_price)
            item.change = float(change)
            item.changePercent = float(change_percent)
            item.volume = current_volume
            item.lastUpdated = datetime.now().isoformat()
            
            # Update fundamentals if available
            if fundamentals:
                item.marketCap = fundamentals.get('marketCap')
                item.pe = fundamentals.get('trailingPE')
                item.dividendYield = fundamentals.get('dividendYield')
                item.beta = fundamentals.get('beta')
                item.high52w = fundamentals.get('52WeekHigh')
                item.low52w = fundamentals.get('52WeekLow')
            
            # Update technical indicators
            item.ema20 = ema_20
            item.ema50 = ema_50
            item.rsi = rsi
            
            # Store price history
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO price_history (symbol, price, volume)
                VALUES (?, ?, ?)
            ''', (symbol, current_price, current_volume))
            conn.commit()
            conn.close()
            
            # Check alerts
            await self._check_alerts(symbol, item)
            
        except Exception as e:
            logger.error(f"Error updating data for {symbol}: {e}")
    
    async def _check_alerts(self, symbol: str, item: WatchlistItemData):
        """Check if any alerts should be triggered"""
        try:
            for alert in item.alerts:
                if not alert.get('enabled', True):
                    continue
                
                should_trigger = False
                alert_type = alert['type']
                target_value = alert['value']
                current_price = item.price
                
                if alert_type == 'price_above' and current_price > target_value:
                    should_trigger = True
                elif alert_type == 'price_below' and current_price < target_value:
                    should_trigger = True
                elif alert_type == 'percent_change' and abs(item.changePercent) > target_value:
                    should_trigger = True
                elif alert_type == 'volume_spike':
                    # Check if volume is X times average (target_value = multiplier)
                    if item.avgVolume and item.volume > (item.avgVolume * target_value):
                        should_trigger = True
                
                if should_trigger:
                    await self._trigger_alert(symbol, alert, item)
                    
        except Exception as e:
            logger.error(f"Error checking alerts for {symbol}: {e}")
    
    async def _trigger_alert(self, symbol: str, alert: Dict, item: WatchlistItemData):
        """Trigger an alert"""
        try:
            # Mark alert as triggered in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE alerts SET triggered_at = CURRENT_TIMESTAMP
                WHERE symbol = ? AND alert_type = ? AND target_value = ?
            ''', (symbol, alert['type'], alert['value']))
            conn.commit()
            conn.close()
            
            # Send alert to WebSocket subscribers
            alert_message = {
                'type': 'alert_triggered',
                'symbol': symbol,
                'alert': alert,
                'current_price': item.price,
                'change_percent': item.changePercent,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.broadcast_to_subscribers(alert_message)
            
            # TODO: Send email notification if enabled
            if alert.get('email_notification', False):
                logger.info(f"Email alert triggered for {symbol}: {alert}")
            
        except Exception as e:
            logger.error(f"Error triggering alert for {symbol}: {e}")
    
    async def update_all_symbols(self):
        """Update data for all symbols in watchlist"""
        try:
            logger.info(f"Updating {len(self.watchlist_items)} watchlist symbols")
            
            # Update in batches to avoid rate limits
            symbols = list(self.watchlist_items.keys())
            batch_size = 5
            
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                tasks = [self._update_symbol_data(symbol) for symbol in batch]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Send updated data to subscribers
                await self.broadcast_watchlist_data()
                
                # Small delay between batches
                if i + batch_size < len(symbols):
                    await asyncio.sleep(1)
            
            logger.info("Watchlist update completed")
            
        except Exception as e:
            logger.error(f"Error updating watchlist: {e}")
    
    def start_price_updates(self):
        """Start background price updates"""
        if self.running:
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._price_update_loop, daemon=True)
        self.update_thread.start()
        logger.info("Price update thread started")
    
    def stop_price_updates(self):
        """Stop background price updates"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
        logger.info("Price update thread stopped")
    
    def _price_update_loop(self):
        """Background loop for price updates"""
        while self.running:
            try:
                # Update every 30 seconds during market hours
                # Update every 5 minutes outside market hours
                now = datetime.now()
                is_market_hours = (9 <= now.hour <= 16) and (now.weekday() < 5)
                update_interval = 30 if is_market_hours else 300
                
                # Run async update in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.update_all_symbols())
                loop.close()
                
                time.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in price update loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    async def add_subscriber(self, websocket: WebSocket):
        """Add WebSocket subscriber"""
        self.subscribers.append(websocket)
        
        # Send current watchlist data
        await self.send_watchlist_data(websocket)
    
    def remove_subscriber(self, websocket: WebSocket):
        """Remove WebSocket subscriber"""
        if websocket in self.subscribers:
            self.subscribers.remove(websocket)
    
    async def add_price_subscriber(self, websocket: WebSocket):
        """Add price update subscriber"""
        self.price_subscribers.append(websocket)
    
    def remove_price_subscriber(self, websocket: WebSocket):
        """Remove price update subscriber"""
        if websocket in self.price_subscribers:
            self.price_subscribers.remove(websocket)
    
    async def send_watchlist_data(self, websocket: WebSocket):
        """Send watchlist data to specific websocket"""
        try:
            data = [asdict(item) for item in self.watchlist_items.values()]
            await websocket.send_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending watchlist data: {e}")
    
    async def broadcast_watchlist_data(self):
        """Broadcast watchlist data to all subscribers"""
        if not self.subscribers:
            return
        
        data = [asdict(item) for item in self.watchlist_items.values()]
        message = json.dumps(data)
        
        disconnected = []
        for websocket in self.subscribers:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected websockets
        for ws in disconnected:
            self.remove_subscriber(ws)
    
    async def broadcast_price_updates(self):
        """Broadcast price updates to price subscribers"""
        if not self.price_subscribers:
            return
        
        # Send only price-related data
        price_data = []
        for item in self.watchlist_items.values():
            price_data.append({
                'symbol': item.symbol,
                'price': item.price,
                'change': item.change,
                'changePercent': item.changePercent,
                'volume': item.volume,
                'lastUpdated': item.lastUpdated
            })
        
        message = json.dumps(price_data)
        
        disconnected = []
        for websocket in self.price_subscribers:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected websockets
        for ws in disconnected:
            self.remove_price_subscriber(ws)
    
    async def broadcast_to_subscribers(self, message: dict):
        """Broadcast message to all subscribers"""
        if not self.subscribers:
            return
        
        message_str = json.dumps(message)
        
        disconnected = []
        for websocket in self.subscribers:
            try:
                await websocket.send_text(message_str)
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected websockets
        for ws in disconnected:
            self.remove_subscriber(ws)
    
    def get_watchlist_data(self) -> List[Dict]:
        """Get current watchlist data"""
        return [asdict(item) for item in self.watchlist_items.values()]
    
    def get_symbol_data(self, symbol: str) -> Optional[Dict]:
        """Get data for specific symbol"""
        if symbol in self.watchlist_items:
            return asdict(self.watchlist_items[symbol])
        return None

# Global manager instance
watchlist_manager = WatchlistManager()

# FastAPI app setup
app = FastAPI(title="Watchlist API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket endpoints
@app.websocket("/ws/watchlist")
async def websocket_watchlist(websocket: WebSocket):
    await websocket.accept()
    await watchlist_manager.add_subscriber(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        watchlist_manager.remove_subscriber(websocket)
        logger.info("Watchlist WebSocket disconnected")
    except Exception as e:
        logger.error(f"Watchlist WebSocket error: {e}")
        watchlist_manager.remove_subscriber(websocket)

@app.websocket("/ws/price-updates")
async def websocket_price_updates(websocket: WebSocket):
    await websocket.accept()
    await watchlist_manager.add_price_subscriber(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        watchlist_manager.remove_price_subscriber(websocket)
        logger.info("Price updates WebSocket disconnected")
    except Exception as e:
        logger.error(f"Price updates WebSocket error: {e}")
        watchlist_manager.remove_price_subscriber(websocket)

# REST API endpoints
@app.get("/api/watchlist", response_model=List[Dict])
async def get_watchlist():
    """Get current watchlist"""
    return watchlist_manager.get_watchlist_data()

@app.post("/api/watchlist/add", response_model=WatchlistResponse)
async def add_to_watchlist(item: WatchlistItem):
    """Add symbol to watchlist"""
    try:
        success = watchlist_manager.add_symbol(
            symbol=item.symbol.upper(),
            notes=item.notes,
            added_date=item.addedDate
        )
        
        if success:
            return WatchlistResponse(
                status="success",
                message=f"Added {item.symbol} to watchlist"
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to add symbol")
            
    except Exception as e:
        logger.error(f"Error adding {item.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/watchlist/remove/{symbol}")
async def remove_from_watchlist(symbol: str):
    """Remove symbol from watchlist"""
    try:
        success = watchlist_manager.remove_symbol(symbol.upper())
        
        if success:
            return WatchlistResponse(
                status="success",
                message=f"Removed {symbol} from watchlist"
            )
        else:
            raise HTTPException(status_code=404, detail="Symbol not found")
            
    except Exception as e:
        logger.error(f"Error removing {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/watchlist/favorite/{symbol}")
async def toggle_favorite(symbol: str):
    """Toggle favorite status"""
    try:
        success = watchlist_manager.toggle_favorite(symbol.upper())
        
        if success:
            return WatchlistResponse(
                status="success",
                message=f"Toggled favorite for {symbol}"
            )
        else:
            raise HTTPException(status_code=404, detail="Symbol not found")
            
    except Exception as e:
        logger.error(f"Error toggling favorite for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/watchlist/alert/{symbol}")
async def add_alert(symbol: str, alert: AlertConfig):
    """Add alert for symbol"""
    try:
        success = watchlist_manager.add_alert(symbol.upper(), alert)
        
        if success:
            return WatchlistResponse(
                status="success",
                message=f"Added alert for {symbol}"
            )
        else:
            raise HTTPException(status_code=404, detail="Symbol not found")
            
    except Exception as e:
        logger.error(f"Error adding alert for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/watchlist/refresh")
async def refresh_watchlist(background_tasks: BackgroundTasks):
    """Refresh all watchlist data"""
    try:
        background_tasks.add_task(watchlist_manager.update_all_symbols)
        return WatchlistResponse(
            status="success",
            message="Watchlist refresh initiated"
        )
    except Exception as e:
        logger.error(f"Error refreshing watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/watchlist/{symbol}")
async def get_symbol_data(symbol: str):
    """Get data for specific symbol"""
    try:
        data = watchlist_manager.get_symbol_data(symbol.upper())
        
        if data:
            return data
        else:
            raise HTTPException(status_code=404, detail="Symbol not found")
            
    except Exception as e:
        logger.error(f"Error getting data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/watchlist/analytics/{symbol}")
async def get_symbol_analytics(symbol: str, days: int = 30):
    """Get analytics for symbol"""
    try:
        conn = sqlite3.connect(watchlist_manager.db_path)
        cursor = conn.cursor()
        
        # Get price history
        cursor.execute('''
            SELECT price, volume, timestamp
            FROM price_history
            WHERE symbol = ? AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp
        '''.format(days), (symbol.upper(),))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'price': row[0],
                'volume': row[1],
                'timestamp': row[2]
            })
        
        conn.close()
        
        # Calculate analytics
        if history:
            prices = [h['price'] for h in history]
            volumes = [h['volume'] for h in history]
            
            analytics = {
                'symbol': symbol.upper(),
                'period_days': days,
                'price_history': history,
                'min_price': min(prices),
                'max_price': max(prices),
                'avg_price': sum(prices) / len(prices),
                'avg_volume': sum(volumes) / len(volumes),
                'price_change': ((prices[-1] - prices[0]) / prices[0]) * 100 if len(prices) > 1 else 0,
                'volatility': np.std(prices) if len(prices) > 1 else 0
            }
        else:
            analytics = {
                'symbol': symbol.upper(),
                'period_days': days,
                'message': 'No historical data available'
            }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting analytics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/watchlist/export")
async def export_watchlist():
    """Export watchlist data"""
    try:
        data = watchlist_manager.get_watchlist_data()
        
        # Create CSV content
        if data:
            headers = list(data[0].keys())
            csv_content = ",".join(headers) + "\n"
            
            for item in data:
                row = []
                for header in headers:
                    value = item.get(header, "")
                    if isinstance(value, str) and "," in value:
                        value = f'"{value}"'
                    row.append(str(value))
                csv_content += ",".join(row) + "\n"
        else:
            csv_content = "No data to export\n"
        
        return {
            "content": csv_content,
            "filename": f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
        
    except Exception as e:
        logger.error(f"Error exporting watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data-providers/status")
async def get_data_providers_status():
    """Get status of all data providers"""
    try:
        status = watchlist_manager.data_provider.get_provider_status()
        return status
    except Exception as e:
        logger.error(f"Error getting provider status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data-providers/{provider_name}/config")
async def update_provider_config(provider_name: str, config: Dict):
    """Update data provider configuration"""
    try:
        watchlist_manager.data_provider.update_provider_config(provider_name, config)
        return WatchlistResponse(
            status="success",
            message=f"Updated {provider_name} configuration"
        )
    except Exception as e:
        logger.error(f"Error updating {provider_name} config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "watchlist_items": len(watchlist_manager.watchlist_items),
        "active_subscribers": len(watchlist_manager.subscribers),
        "price_subscribers": len(watchlist_manager.price_subscribers)
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("Watchlist API starting up...")
    # Initial data refresh
    await watchlist_manager.update_all_symbols()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Watchlist API shutting down...")
    watchlist_manager.stop_price_updates()

if __name__ == "__main__":
    import uvicorn
    import numpy as np
    
    print("🚀 Starting Watchlist Backend Server...")
    print("📊 Watchlist API will be available at http://localhost:8000")
    print("🔌 WebSocket endpoints:")
    print("   - ws://localhost:8000/ws/watchlist")
    print("   - ws://localhost:8000/ws/price-updates")
    print("📖 API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "watchlist_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )