import React from 'react';
   import styles from './TopBar.module.css';

   function TopBar({ tabs, activeTab, onTabChange, activeSubTab, onSubTabChange }) {
     const currentTab = tabs.find(tab => tab.id === activeTab);
     const subTabs = currentTab?.subTabs || [];

     return (
       <nav className={styles.topBar}>
         <div className={styles.mainTabs}>
           {tabs.map(tab => (
             <button
               key={tab.id}
               className={`${styles.tab} ${activeTab === tab.id ? styles.active : ''}`}
               onClick={() => onTabChange(tab.id)}
             >
               {tab.label}
             </button>
           ))}
         </div>
         {subTabs.length > 0 && (
           <div className={styles.subTabs}>
             {subTabs.map(subTab => (
               <button
                 key={subTab.id}
                 className={`${styles.subTab} ${activeSubTab === subTab.id ? styles.active : ''}`}
                 onClick={() => onSubTabChange(subTab.id)}
               >
                 {subTab.label}
               </button>
             ))}
           </div>
         )}
       </nav>
     );
   }

   export default TopBar;