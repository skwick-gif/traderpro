import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RefreshCw, Download, Settings, ChevronUp, ChevronDown, AlertTriangle } from 'lucide-react';
import styles from './StockScanner.module.css';

const STOCK_UNIVERSES = [
  { value: 'NASDAQ100', label: 'NASDAQ 100' },
  { value: 'SP500', label: 'S&P 500' },
  { value: 'RUSSELL2000', label: 'Russell 2000' },
  { value: 'NYSE', label: 'NYSE Composite' },
  { value: 'CUSTOM', label: 'Custom List' }
];

function StockScanner() {
  const [scanSettings, setScanSettings] = useState({
    stock_universe: 'NASDAQ100',
    min_score: 5.0,
    max_results: 25,
    technical_weight: 0.35,
    fundamental_weight: 0.25,
    sentiment_weight: 0.25,
    pattern_weight: 0.15,
    min_price: 10.0,
    max_price: 1000.0,
    min_volume: 100000,
    min_market_cap: 1000000000
  });

  const [scanResults, setScanResults] = useState([]);
  const [isScanning, setIsScanning] = useState(false);
  const [scanStatus, setScanStatus] = useState('idle');
  const [marketRegime, setMarketRegime] = useState(null);
  const [lastScanTime, setLastScanTime] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [errorMessage, setErrorMessage] = useState('');
  const [universeStatus, setUniverseStatus] = useState({});
  const wsRef = useRef(null);

  // WebSocket connection
  useEffect(() => {
    connectWebSocket();
    
    // Listen for external scan triggers
    const handleExternalScan = (event) => {
      if (event.detail.action === 'startScan') {
        handleStartScan();
      }
    };
    
    window.addEventListener('stockScannerAction', handleExternalScan);
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      window.removeEventListener('stockScannerAction', handleExternalScan);
    };
  }, []);

  const connectWebSocket = () => {
    try {
      wsRef.current = new WebSocket('ws://localhost:8000/ws/stock-scanner');
      
      wsRef.current.onopen = () => {
        console.log('Stock Scanner WebSocket connected');
        setConnectionStatus('connected');
        setErrorMessage('');
      };
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'scan_status':
            setScanStatus(data.status);
            setIsScanning(data.status === 'running');
            break;
          case 'scan_results':
            if (data.results && data.results.length > 0) {
              setScanResults(data.results);
              setLastScanTime(new Date().toLocaleTimeString());
              setErrorMessage('');
            } else {
              setScanResults([]);
              setErrorMessage('No stocks found matching the criteria. Try adjusting your filters.');
            }
            setIsScanning(false);
            setScanStatus('completed');
            break;
          case 'market_regime':
            setMarketRegime(data.regime);
            break;
          case 'universe_status':
            setUniverseStatus(data.status);
            break;
          case 'error':
            console.error('Scan error:', data.message);
            setErrorMessage(data.message);
            setIsScanning(false);
            setScanStatus('error');
            break;
          case 'no_data':
            setErrorMessage('No real-time data available. Please check your data connection.');
            setIsScanning(false);
            setScanStatus('error');
            break;
        }
      };
      
      wsRef.current.onclose = () => {
        console.log('Stock Scanner WebSocket disconnected');
        setConnectionStatus('disconnected');
        setErrorMessage('Backend connection lost. Please check if the backend is running.');
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
        setErrorMessage('Failed to connect to backend. Make sure the backend server is running on port 8000.');
      };
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setConnectionStatus('error');
      setErrorMessage('WebSocket connection failed.');
    }
  };

  const handleStartScan = async () => {
    if (isScanning) return;
    
    if (connectionStatus !== 'connected') {
      setErrorMessage('Backend not connected. Please start the backend service.');
      return;
    }
    
    setIsScanning(true);
    setScanStatus('running');
    setScanResults([]);
    setErrorMessage('');
    
    try {
      const response = await fetch('http://localhost:8000/api/scan/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(scanSettings)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
    } catch (error) {
      console.error('Failed to start scan:', error);
      setIsScanning(false);
      setScanStatus('error');
      setErrorMessage(`Failed to start scan: ${error.message}. Make sure the backend is running on port 8000.`);
    }
  };

  const handleSettingChange = (key, value) => {
    setScanSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const exportResults = () => {
    if (scanResults.length === 0) {
      setErrorMessage('No results to export');
      return;
    }
    
    const csvContent = [
      ['Symbol', 'Score', 'Price', 'Volume', 'Technical', 'Fundamental', 'Sentiment', 'Pattern', 'Risk'].join(','),
      ...scanResults.map(stock => [
        stock.symbol,
        stock.total_score,
        stock.current_price,
        stock.volume,
        stock.technical_score,
        stock.fundamental_score,
        stock.sentiment_score,
        stock.pattern_score,
        stock.risk_level
      ].join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `stock_scan_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const checkUniverseStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/universe/status');
      if (response.ok) {
        const status = await response.json();
        setUniverseStatus(status);
      }
    } catch (error) {
      console.error('Failed to check universe status:', error);
    }
  };

  const checkBackendConnection = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/connection/test');
      if (response.ok) {
        const data = await response.json();
        setConnectionStatus(data.backend_status);
        if (!data.data_available) {
          setErrorMessage('Backend connected but no real-time data available');
        }
      } else {
        setConnectionStatus('error');
        setErrorMessage('Backend connection failed');
      }
    } catch (error) {
      setConnectionStatus('disconnected');
      setErrorMessage('Cannot connect to backend. Make sure backend_stock_scanner.py is running on port 8000.');
    }
  };

  useEffect(() => {
    // Check backend connection every 30 seconds
    checkBackendConnection();
    const interval = setInterval(checkBackendConnection, 30000);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h2 className={styles.title}>Stock Scanner</h2>
          <div className={styles.statusIndicators}>
            <div className={`${styles.connectionStatus} ${styles[connectionStatus]}`}>
              Backend: {connectionStatus}
            </div>
            {lastScanTime && (
              <span className={styles.lastScan}>Last scan: {lastScanTime}</span>
            )}
          </div>
        </div>
        <div className={styles.headerRight}>
          <button
            onClick={handleStartScan}
            disabled={isScanning || connectionStatus !== 'connected'}
            className={`${styles.button} ${isScanning ? styles.scanning : styles.start}`}
          >
            {isScanning ? <RefreshCw className={styles.spinning} size={16} /> : <Play size={16} />}
            {isScanning ? 'Scanning...' : 'Start Scan'}
          </button>
          {scanResults.length > 0 && (
            <button onClick={exportResults} className={`${styles.button} ${styles.export}`}>
              <Download size={16} />
              Export
            </button>
          )}
        </div>
      </div>

      {/* Error Message */}
      {errorMessage && (
        <div className={styles.errorMessage}>
          <AlertTriangle size={16} />
          <span>{errorMessage}</span>
        </div>
      )}

      <div className={styles.content}>
        {/* Settings Panel */}
        <div className={styles.settingsPanel}>
          <div className={styles.panelHeader}>
            <h3 className={styles.panelTitle}>Scanner Settings</h3>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className={styles.toggleButton}
            >
              {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              Advanced
            </button>
          </div>

          {/* Basic Settings */}
          <div className={styles.settingsGrid}>
            <div className={styles.setting}>
              <label className={styles.label}>Universe</label>
              <select
                value={scanSettings.stock_universe}
                onChange={(e) => handleSettingChange('stock_universe', e.target.value)}
                className={styles.select}
              >
                {STOCK_UNIVERSES.map(universe => (
                  <option key={universe.value} value={universe.value}>
                    {universe.label}
                    {universeStatus[universe.value] && ` (${universeStatus[universe.value]} stocks)`}
                  </option>
                ))}
              </select>
              {universeStatus[scanSettings.stock_universe] && (
                <div className={styles.universeInfo}>
                  Status: {universeStatus[scanSettings.stock_universe] > 0 ? 'Loaded' : 'Loading...'}
                </div>
              )}
            </div>

            <div className={styles.setting}>
              <label className={styles.label}>Min Score: {scanSettings.min_score}</label>
              <input
                type="range"
                min="1"
                max="10"
                step="0.1"
                value={scanSettings.min_score}
                onChange={(e) => handleSettingChange('min_score', parseFloat(e.target.value))}
                className={styles.slider}
              />
            </div>

            <div className={styles.setting}>
              <label className={styles.label}>Max Results: {scanSettings.max_results}</label>
              <input
                type="range"
                min="5"
                max="100"
                step="5"
                value={scanSettings.max_results}
                onChange={(e) => handleSettingChange('max_results', parseInt(e.target.value))}
                className={styles.slider}
              />
            </div>

            <div className={styles.setting}>
              <label className={styles.label}>Min Price: ${scanSettings.min_price}</label>
              <input
                type="range"
                min="1"
                max="100"
                step="1"
                value={scanSettings.min_price}
                onChange={(e) => handleSettingChange('min_price', parseFloat(e.target.value))}
                className={styles.slider}
              />
            </div>
          </div>

          {/* Advanced Settings */}
          {showAdvanced && (
            <div className={styles.advancedSettings}>
              <h4 className={styles.advancedTitle}>Weight Distribution</h4>
              <div className={styles.settingsGrid}>
                <div className={styles.setting}>
                  <label className={styles.label}>Technical: {(scanSettings.technical_weight * 100).toFixed(0)}%</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={scanSettings.technical_weight}
                    onChange={(e) => handleSettingChange('technical_weight', parseFloat(e.target.value))}
                    className={styles.slider}
                  />
                </div>

                <div className={styles.setting}>
                  <label className={styles.label}>Fundamental: {(scanSettings.fundamental_weight * 100).toFixed(0)}%</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={scanSettings.fundamental_weight}
                    onChange={(e) => handleSettingChange('fundamental_weight', parseFloat(e.target.value))}
                    className={styles.slider}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Market Regime */}
          {marketRegime && (
            <div className={styles.marketRegime}>
              <h4 className={styles.regimeTitle}>Market Regime</h4>
              <div className={styles.regimeInfo}>
                <span className={styles.regimeType}>{marketRegime.type}</span>
                <span className={styles.regimeConfidence}>{marketRegime.confidence}% confidence</span>
              </div>
            </div>
          )}
        </div>

        {/* Results */}
        <div className={styles.resultsPanel}>
          <div className={styles.panelHeader}>
            <h3 className={styles.panelTitle}>
              Scan Results {scanResults.length > 0 && `(${scanResults.length})`}
            </h3>
            <div className={styles.scanStatus}>
              <span className={`${styles.statusIndicator} ${styles[scanStatus]}`}>
                {scanStatus}
              </span>
            </div>
          </div>

          {isScanning && (
            <div className={styles.loadingContainer}>
              <RefreshCw className={styles.spinning} size={24} />
              <span>Analyzing stocks... This may take a few minutes.</span>
            </div>
          )}

          {scanResults.length > 0 && !isScanning && (
            <div className={styles.resultsTable}>
              <div className={styles.tableHeader}>
                <div className={styles.headerCell}>Symbol</div>
                <div className={styles.headerCell}>Score</div>
                <div className={styles.headerCell}>Price</div>
                <div className={styles.headerCell}>Volume</div>
                <div className={styles.headerCell}>Tech</div>
                <div className={styles.headerCell}>Fund</div>
                <div className={styles.headerCell}>Sent</div>
                <div className={styles.headerCell}>Pattern</div>
                <div className={styles.headerCell}>Risk</div>
              </div>

              <div className={styles.tableBody}>
                {scanResults.map((stock, index) => (
                  <div key={stock.symbol} className={styles.tableRow}>
                    <div className={styles.cell}>
                      <span className={styles.rank}>{index + 1}</span>
                      <span className={styles.symbol}>{stock.symbol}</span>
                    </div>
                    <div className={styles.cell}>
                      <span className={styles.score}>{stock.total_score?.toFixed(1) || 'N/A'}</span>
                    </div>
                    <div className={styles.cell}>${stock.current_price?.toFixed(2) || 'N/A'}</div>
                    <div className={styles.cell}>
                      {stock.volume ? (stock.volume / 1000000).toFixed(1) + 'M' : 'N/A'}
                    </div>
                    <div className={styles.cell}>{stock.technical_score?.toFixed(1) || 'N/A'}</div>
                    <div className={styles.cell}>{stock.fundamental_score?.toFixed(1) || 'N/A'}</div>
                    <div className={styles.cell}>{stock.sentiment_score?.toFixed(1) || 'N/A'}</div>
                    <div className={styles.cell}>{stock.pattern_score?.toFixed(1) || 'N/A'}</div>
                    <div className={`${styles.cell} ${styles.risk} ${styles[stock.risk_level?.toLowerCase()]}`}>
                      {stock.risk_level || 'N/A'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {scanResults.length === 0 && !isScanning && scanStatus !== 'idle' && (
            <div className={styles.noResults}>
              <AlertTriangle size={48} />
              <h3>No Results Found</h3>
              <p>No stocks found matching the criteria or no real-time data available.</p>
              <p>Try lowering the minimum score or check your data connection.</p>
            </div>
          )}

          {connectionStatus === 'disconnected' && (
            <div className={styles.noResults}>
              <AlertTriangle size={48} />
              <h3>Backend Not Connected</h3>
              <p>Please start the backend service to begin scanning.</p>
              <p>Run: python backend_stock_scanner.py</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default StockScanner;