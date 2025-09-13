import React from 'react';
import { Wifi, WifiOff, Activity } from 'lucide-react';
import useWebSocket from '../hooks/useWebSocket';
import styles from './StatusBar.module.css';

function StatusBar() {
  const { data: ibkrStatus, isConnected: isIBKRConnected } = useWebSocket('ws://localhost:8000/ws/ibkr');
  const { data: dataStatus, isConnected: isDataConnected } = useWebSocket('ws://localhost:8000/ws/data');
  const { data: vixData } = useWebSocket('ws://localhost:8000/ws/vix');

  return (
    <header className={styles.statusBar}>
      <div className={styles.logo}>
        <span className={styles.logoIcon}>📈</span>
        <span className={styles.toolName}>TraderPro</span>
      </div>
      <div className={styles.statusItems}>
        <div className={styles.statusItem}>
          {isIBKRConnected ? <Wifi className={styles.connected} /> : <WifiOff className={styles.disconnected} />}
          <span>IBKR</span>
        </div>
        <div className={styles.statusItem}>
          <span className={isDataConnected ? styles.connectedTag : styles.disconnectedTag}>
            Data
          </span>
        </div>
        <div className={styles.statusItem}>
          <Activity className={styles.vixIcon} />
          <span className={styles.vixValue}>VIX: {vixData?.value || 'N/A'}</span>
        </div>
      </div>
    </header>
  );
}

export default StatusBar;