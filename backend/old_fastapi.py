#!/usr/bin/env python3
"""
FastAPI Backend for Stock Scanner Integration
WebSocket and REST API endpoints for the stock scanner
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
import logging
from datetime import datetime
import threading

from unified_stock_scanner import UnifiedStockScanner, ScanSettings, StockScore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Scanner API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global scanner instance
scanner = UnifiedStockScanner()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.scan_status = "idle"
        self.current_results = []
        self.market_regime = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def update_scan_status(self, status: str):
        self.scan_status = status
        await self.broadcast({
            "type": "scan_status",
            "status": status,
            "timestamp": datetime.now().isoformat()
        })

    async def send_results(self, results: List[StockScore]):
        self.current_results = results
        # Convert StockScore objects to dictionaries
        results_data = []
        for result in results:
            results_data.append({
                "symbol": result.symbol,
                "total_score": round(result.total_score, 2),
                "current_price": round(result.current_price, 2),
                "volume": result.volume,
                "market_cap": result.market_cap,
                "technical_score": round(result.technical_score, 1),
                "fundamental_score": round(result.fundamental_score, 1),
                "sentiment_score": round(result.sentiment_score, 1),
                "pattern_score": round(result.pattern_score, 1),
                "risk_level": result.risk_level,
                "prediction": result.prediction,
                "timestamp": result.timestamp.isoformat()
            })
        
        await self.broadcast({
            "type": "scan_results",
            "results": results_data,
            "count": len(results_data),
            "timestamp": datetime.now().isoformat()
        })

    async def send_market_regime(self, regime: dict):
        self.market_regime = regime
        await self.broadcast({
            "type": "market_regime",
            "regime": regime,
            "timestamp": datetime.now().isoformat()
        })

manager = ConnectionManager()

# Pydantic models
class ScanSettingsModel(BaseModel):
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

class ScanResponse(BaseModel):
    status: str
    message: str
    scan_id: Optional[str] = None

# Background scan task
async def run_stock_scan(settings_dict: dict):
    """Run stock scan in background"""
    try:
        logger.info("Starting background stock scan")
        await manager.update_scan_status("running")
        
        # Convert dict to ScanSettings
        settings = ScanSettings(**settings_dict)
        
        # Get market regime first
        regime = scanner.market_detector.get_current_regime()
        await manager.send_market_regime(regime)
        
        # Run the scan (this is CPU intensive, so we run it in a thread)
        def scan_thread():
            return scanner.scan_stocks(settings)
        
        # Run in thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, scan_thread)
        
        # Send results via WebSocket
        await manager.send_results(results)
        await manager.update_scan_status("completed")
        
        logger.info(f"Scan completed with {len(results)} results")
        
    except Exception as e:
        logger.error(f"Scan error: {e}")
        await manager.update_scan_status("error")
        await manager.broadcast({
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        })

# WebSocket endpoints
@app.websocket("/ws/stock-scanner")
async def websocket_stock_scanner(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send current status on connect
        await manager.send_personal_message({
            "type": "connected",
            "status": manager.scan_status,
            "timestamp": datetime.now().isoformat()
        }, websocket)
        
        # Send current market regime if available
        if manager.market_regime:
            await manager.send_personal_message({
                "type": "market_regime",
                "regime": manager.market_regime,
                "timestamp": datetime.now().isoformat()
            }, websocket)
        
        # Send current results if available
        if manager.current_results:
            await manager.send_results(manager.current_results)
        
        while True:
            # Keep connection alive
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Stock scanner WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# REST API endpoints
@app.post("/api/stock-scanner/start", response_model=ScanResponse)
async def start_stock_scan(settings: ScanSettingsModel):
    """Start a new stock scan"""
    try:
        if manager.scan_status == "running":
            raise HTTPException(status_code=400, detail="Scan already in progress")
        
        # Start scan in background
        asyncio.create_task(run_stock_scan(settings.dict()))
        
        return ScanResponse(
            status="started",
            message="Stock scan initiated",
            scan_id=f"scan_{int(datetime.now().timestamp())}"
        )
        
    except Exception as e:
        logger.error(f"Error starting scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock-scanner/status")
async def get_scan_status():
    """Get current scan status"""
    return {
        "status": manager.scan_status,
        "results_count": len(manager.current_results),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stock-scanner/results")
async def get_scan_results():
    """Get latest scan results"""
    if not manager.current_results:
        return {"results": [], "count": 0}
    
    results_data = []
    for result in manager.current_results:
        results_data.append({
            "symbol": result.symbol,
            "total_score": round(result.total_score, 2),
            "current_price": round(result.current_price, 2),
            "volume": result.volume,
            "market_cap": result.market_cap,
            "technical_score": round(result.technical_score, 1),
            "fundamental_score": round(result.fundamental_score, 1),
            "sentiment_score": round(result.sentiment_score, 1),
            "pattern_score": round(result.pattern_score, 1),
            "risk_level": result.risk_level,
            "prediction": result.prediction,
            "timestamp": result.timestamp.isoformat()
        })
    
    return {
        "results": results_data,
        "count": len(results_data),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/market-regime")
async def get_market_regime():
    """Get current market regime"""
    try:
        regime = scanner.market_detector.get_current_regime()
        return regime
    except Exception as e:
        logger.error(f"Error getting market regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stock-scanner/stop")
async def stop_scan():
    """Stop current scan"""
    try:
        await manager.update_scan_status("stopped")
        return {"status": "stopped", "message": "Scan stopped"}
    except Exception as e:
        logger.error(f"Error stopping scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "scanner_status": manager.scan_status
    }

# Get stock universes
@app.get("/api/stock-universes")
async def get_stock_universes():
    """Get available stock universes"""
    return {
        "universes": {
            "NASDAQ100": "NASDAQ 100",
            "SP500": "S&P 500",
            "CUSTOM": "Custom List"
        }
    }

# WebSocket for general stats (existing functionality)
@app.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send mock stats data
            stats = [
                {"label": "Active Scans", "value": "3", "trend": "up"},
                {"label": "Stocks Analyzed", "value": "1,247", "trend": "up"},
                {"label": "Avg Score", "value": "7.8", "trend": "up"},
                {"label": "Market Regime", "value": manager.market_regime["type"] if manager.market_regime else "UNKNOWN", "trend": "neutral"}
            ]
            await websocket.send_json(stats)
            await asyncio.sleep(30)  # Update every 30 seconds
    except WebSocketDisconnect:
        pass

@app.websocket("/ws/scans")
async def websocket_scans(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Send recent scan results as mock "recent scans"
            recent_scans = []
            if manager.current_results:
                for result in manager.current_results[:5]:  # Top 5 results
                    recent_scans.append({
                        "symbol": result.symbol,
                        "price": f"${result.current_price:.2f}",
                        "volume": result.volume,
                        "type": "Stock"
                    })
            
            await websocket.send_json(recent_scans)
            await asyncio.sleep(60)  # Update every minute
    except WebSocketDisconnect:
        pass

# Additional WebSocket endpoints for dashboard compatibility
@app.websocket("/ws/ibkr")
async def websocket_ibkr(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Mock IBKR connection status
            await websocket.send_json({"status": "connected", "account": "DU1234567"})
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass

@app.websocket("/ws/data")
async def websocket_data(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Mock data feed status
            await websocket.send_json({"status": "active", "feed": "real-time"})
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass

@app.websocket("/ws/vix")
async def websocket_vix(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Get real VIX data if available
            vix_value = 20.0  # Default
            if manager.market_regime and "vix_level" in manager.market_regime:
                vix_value = manager.market_regime["vix_level"]
            
            await websocket.send_json({"value": f"{vix_value:.1f}"})
            await asyncio.sleep(300)  # Update every 5 minutes
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Stock Scanner Backend Server...")
    print("📊 Stock Scanner API will be available at http://localhost:8000")
    print("🔌 WebSocket endpoints:")
    print("   - ws://localhost:8000/ws/stock-scanner")
    print("   - ws://localhost:8000/ws/stats")
    print("   - ws://localhost:8000/ws/scans")
    print("📖 API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        "backend_stock_scanner:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )