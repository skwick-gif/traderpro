
import React from 'react';
import styles from './TopBar.module.css';

   function TopBar({ tabs, activeTab, onTabChange, activeSubTab, onSubTabChange }) {
  console.log('TopBar props:', { tabs, activeTab });
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
        {/* הסרנו את שורת התתי-לשוניות */}
      </nav>
    );
   }

   export default TopBar;