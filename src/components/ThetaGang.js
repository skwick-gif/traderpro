import React, { useState } from 'react';
import { Play, Pause, RefreshCw } from 'lucide-react';
import styles from './ThetaGang.module.css';

function ThetaGang() {
  const [config, setConfig] = useState({
    account: {
      number: 'DU1234567',
      margin_usage: 0.5,
      cancel_orders: true,
      market_data_type: 1
    },
    target: {
      dte: 45,
      max_dte: 180,
      delta: 0.3,
      maximum_new_contracts_percent: 0.05,
      minimum_open_interest: 10
    },
    roll_when: {
      pnl: 0.9,
      dte: 15,
      min_pnl: 0.0
    },
    symbols: {
      SPY: { weight: 0.4, delta: 0.3 },
      QQQ: { weight: 0.3, delta: 0.5 },
      TLT: { weight: 0.2, delta: 0.4 },
      ABNB: { weight: 0.05, primary_exchange: 'NASDAQ' }
    },
    vix_call_hedge: {
      enabled: false,
      delta: 0.30,
      target_dte: 30,
      close_hedges_when_vix_exceeds: 50.0
    },
    write_when: {
      calls: {
        green: true,
        red: false,
        cap_factor: 1.0
      },
      puts: {
        green: false,
        red: true
      }
    }
  });

  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState([
    'ThetaGang initialized',
    'Waiting for configuration...'
  ]);

  const handleConfigChange = (section, key, value) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
    addLog(`Updated ${section}.${key} to ${value}`);
  };

  const handleSymbolChange = (symbol, key, value) => {
    setConfig(prev => ({
      ...prev,
      symbols: {
        ...prev.symbols,
        [symbol]: {
          ...prev.symbols[symbol],
          [key]: value
        }
      }
    }));
    addLog(`Updated ${symbol}.${key} to ${value}`);
  };

  const addLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev.slice(-19), `[${timestamp}] ${message}`]);
  };

  const toggleExecution = () => {
    setIsRunning(!isRunning);
    if (!isRunning) {
      addLog('ThetaGang execution started');
    } else {
      addLog('ThetaGang execution stopped');
    }
  };

  const refreshData = () => {
    addLog('Refreshing market data and positions...');
  };

  const exportConfig = () => {
    const tomlConfig = generateTOMLConfig(config);
    const blob = new Blob([tomlConfig], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'thetagang.toml';
    a.click();
    URL.revokeObjectURL(url);
    addLog('Configuration exported to thetagang.toml');
  };

  const generateTOMLConfig = (config) => {
    return `# ThetaGang Configuration
[account]
number = "${config.account.number}"
cancel_orders = ${config.account.cancel_orders}
margin_usage = ${config.account.margin_usage}
market_data_type = ${config.account.market_data_type}

[target]
dte = ${config.target.dte}
max_dte = ${config.target.max_dte}
delta = ${config.target.delta}
maximum_new_contracts_percent = ${config.target.maximum_new_contracts_percent}
minimum_open_interest = ${config.target.minimum_open_interest}

[roll_when]
pnl = ${config.roll_when.pnl}
dte = ${config.roll_when.dte}
min_pnl = ${config.roll_when.min_pnl}

[write_when]
  [write_when.calls]
  green = ${config.write_when.calls.green}
  red = ${config.write_when.calls.red}
  cap_factor = ${config.write_when.calls.cap_factor}
  
  [write_when.puts]
  green = ${config.write_when.puts.green}
  red = ${config.write_when.puts.red}

[symbols]
${Object.entries(config.symbols).map(([symbol, data]) => `
  [symbols.${symbol}]
  weight = ${data.weight}
  delta = ${data.delta}${data.primary_exchange ? `\n  primary_exchange = "${data.primary_exchange}"` : ''}`).join('')}

[vix_call_hedge]
enabled = ${config.vix_call_hedge.enabled}
delta = ${config.vix_call_hedge.delta}
target_dte = ${config.vix_call_hedge.target_dte}
close_hedges_when_vix_exceeds = ${config.vix_call_hedge.close_hedges_when_vix_exceeds}
`;
  };

  return (
    <div className={styles.container}>
      {/* Configuration Panel */}
      <div className={styles.configPanel}>
        <h3 className={styles.panelTitle}>ThetaGang Configuration</h3>

        {/* Account Settings */}
        <div className={styles.section}>
          <h4 className={styles.sectionTitle}>Account</h4>
          <div className={styles.field}>
            <label className={styles.label}>Account Number</label>
            <input
              type="text"
              value={config.account.number}
              onChange={(e) => handleConfigChange('account', 'number', e.target.value)}
              className={styles.input}
            />
          </div>
          <div className={styles.field}>
            <label className={styles.label}>Margin Usage (0-4)</label>
            <input
              type="number"
              min="0"
              max="4"
              step="0.1"
              value={config.account.margin_usage}
              onChange={(e) => handleConfigChange('account', 'margin_usage', parseFloat(e.target.value))}
              className={styles.input}
            />
          </div>
        </div>

        {/* Target Settings */}
        <div className={styles.section}>
          <h4 className={styles.sectionTitle}>Target</h4>
          <div className={styles.field}>
            <label className={styles.label}>DTE (Days to Expiry)</label>
            <input
              type="number"
              min="1"
              max="365"
              value={config.target.dte}
              onChange={(e) => handleConfigChange('target', 'dte', parseInt(e.target.value))}
              className={styles.input}
            />
          </div>
          <div className={styles.field}>
            <label className={styles.label}>Delta (0-1)</label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={config.target.delta}
              onChange={(e) => handleConfigChange('target', 'delta', parseFloat(e.target.value))}
              className={styles.input}
            />
          </div>
          <div className={styles.field}>
            <label className={styles.label}>Max New Contracts %</label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={config.target.maximum_new_contracts_percent}
              onChange={(e) => handleConfigChange('target', 'maximum_new_contracts_percent', parseFloat(e.target.value))}
              className={styles.input}
            />
          </div>
        </div>

        {/* Roll When Settings */}
        <div className={styles.section}>
          <h4 className={styles.sectionTitle}>Roll When</h4>
          <div className={styles.field}>
            <label className={styles.label}>P&L Threshold (0-1)</label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.01"
              value={config.roll_when.pnl}
              onChange={(e) => handleConfigChange('roll_when', 'pnl', parseFloat(e.target.value))}
              className={styles.input}
            />
          </div>
          <div className={styles.field}>
            <label className={styles.label}>DTE Threshold</label>
            <input
              type="number"
              min="1"
              max="60"
              value={config.roll_when.dte}
              onChange={(e) => handleConfigChange('roll_when', 'dte', parseInt(e.target.value))}
              className={styles.input}
            />
          </div>
        </div>

        {/* VIX Hedge Toggle */}
        <div className={styles.section}>
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={config.vix_call_hedge.enabled}
              onChange={(e) => handleConfigChange('vix_call_hedge', 'enabled', e.target.checked)}
              className={styles.checkbox}
            />
            Enable VIX Call Hedge
          </label>
        </div>

        {/* Control Buttons בתוך ה-main box בלבד */}
        <div className={styles.controls}>
          <button
            onClick={toggleExecution}
            className={`${styles.button} ${isRunning ? styles.stopButton : styles.startButton}`}
          >
            {isRunning ? <Pause size={16} /> : <Play size={16} />}
            {isRunning ? 'Stop ThetaGang' : 'Start ThetaGang'}
          </button>
          <button
            onClick={refreshData}
            className={`${styles.button} ${styles.refreshButton}`}
          >
            <RefreshCw size={16} />
            Refresh Data
          </button>
          <button
            onClick={exportConfig}
            className={`${styles.button} ${styles.exportButton}`}
          >
            Export TOML
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className={styles.mainContent}>
        {/* Symbols Configuration */}
        <div className={styles.symbolsSection}>
          <h3 className={styles.sectionTitle}>Symbols Configuration</h3>
          <div className={styles.symbolsGrid}>
            {Object.entries(config.symbols).map(([symbol, data]) => (
              <div key={symbol} className={styles.symbolCard}>
                <h4 className={styles.symbolName}>{symbol}</h4>
                <div className={styles.field}>
                  <label className={styles.label}>Weight</label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={data.weight}
                    onChange={(e) => handleSymbolChange(symbol, 'weight', parseFloat(e.target.value))}
                    className={styles.input}
                  />
                </div>
                <div className={styles.field}>
                  <label className={styles.label}>Delta</label>
                  <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={data.delta}
                    onChange={(e) => handleSymbolChange(symbol, 'delta', parseFloat(e.target.value))}
                    className={styles.input}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Execution Log */}
        <div className={styles.logSection}>
          <h3 className={styles.sectionTitle}>Execution Log</h3>
          <div className={styles.logContainer}>
            {logs.map((log, index) => (
              <div key={index} className={styles.logEntry}>
                {log}
              </div>
            ))}
            {isRunning && (
              <div className={styles.runningIndicator}>
                ThetaGang is running...
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default ThetaGang;