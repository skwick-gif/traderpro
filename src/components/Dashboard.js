import React from 'react';
   import styles from './Dashboard.module.css';
   import useWebSocket from '../hooks/useWebSocket';

   function Dashboard() {
    const [fredData, setFredData] = React.useState({ rate: null, inflation: null, calendar: [] });

    React.useEffect(() => {
      async function fetchFred() {
        try {
          const res = await fetch('/api/fred');
          const fred = await res.json();
          setFredData({ rate: fred.rate, inflation: fred.inflation, calendar: fred.calendar ?? [] });
        } catch (e) {
          setFredData({ rate: null, inflation: null, calendar: [] });
        }
      }
      fetchFred();
    }, []);

    return (
      <div className={styles.dashboard}>
        <div className={styles.statsGrid}>
          <div className={styles.statCard}>
            <h3 className={styles.statLabel}>Fed Rate</h3>
            <p className={styles.statValue}>{fredData.rate ?? 'N/A'}</p>
          </div>
          <div className={styles.statCard}>
            <h3 className={styles.statLabel}>Inflation (CPI)</h3>
            <p className={styles.statValue}>{fredData.inflation ?? 'N/A'}</p>
          </div>
        </div>
        <div className={styles.tableSection}>
          <h2 className={styles.sectionTitle}>Economic Calendar</h2>
          <div className={styles.tableContainer}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Event</th>
                </tr>
              </thead>
              <tbody>
                {fredData.calendar.map((item, idx) => (
                  <tr key={idx}>
                    <td>{item.date}</td>
                    <td>{item.event}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
   }

   export default Dashboard;