import React from 'react';
   import styles from './Dashboard.module.css';
   import useWebSocket from '../hooks/useWebSocket';

   function Dashboard() {
     const { data: statsData } = useWebSocket('ws://localhost:8000/ws/stats');
     const { data: scansData } = useWebSocket('ws://localhost:8000/ws/scans');

     const stats = statsData || [];
     const recentScans = scansData || [];

     return (
       <div className={styles.dashboard}>
         <div className={styles.statsGrid}>
           {stats.map((stat, index) => (
             <div key={index} className={styles.statCard}>
               <h3 className={styles.statLabel}>{stat.label}</h3>
               <p className={styles.statValue}>{stat.value}</p>
               <span className={`${styles.trend} ${styles[stat.trend]}`}>{stat.trend === 'up' ? '↑' : '↓'}</span>
             </div>
           ))}
         </div>
         <div className={styles.tableSection}>
           <h2 className={styles.sectionTitle}>Recent Scans</h2>
           <div className={styles.tableContainer}>
             <table className={styles.table}>
               <thead>
                 <tr>
                   <th>Symbol</th>
                   <th>Price</th>
                   <th>Volume</th>
                   <th>Type</th>
                 </tr>
               </thead>
               <tbody>
                 {recentScans.map((scan, index) => (
                   <tr key={index}>
                     <td>{scan.symbol}</td>
                     <td>{scan.price}</td>
                     <td>{scan.volume?.toLocaleString()}</td>
                     <td><span className={`${styles.tag} ${styles[`tag-${scan.type?.toLowerCase()}`]}`}>{scan.type}</span></td>
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