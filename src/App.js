
import React, { useState, useRef, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import StatusBar from './components/StatusBar';
import { tabsConfig } from './config/tabsConfig';
import './styles/GlobalStyles.css';

console.log('App.js loaded');


function ErrorOverlay() {
  const [errors, setErrors] = useState([]);

  useEffect(() => {
    const handleError = (event) => {
      const msg = event.message || String(event.error);
      setErrors(prev => [...prev, msg]);
      console.log('[ErrorOverlay] window.onerror:', msg);
    };
    const handleRejection = (event) => {
      const msg = event.reason ? String(event.reason) : 'Unhandled promise rejection';
      setErrors(prev => [...prev, msg]);
      console.log('[ErrorOverlay] window.onunhandledrejection:', msg);
    };
    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleRejection);
    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleRejection);
    };
  }, []);

  if (errors.length === 0) return null;
  return (
    <div style={{ position: 'fixed', top: 0, left: 0, right: 0, background: '#b71c1c', color: 'white', padding: '1rem', zIndex: 9999, fontWeight: 'bold', fontSize: '1.2rem', textAlign: 'center' }}>
      <div>שגיאות מערכת:</div>
      <ul style={{ listStyle: 'none', margin: 0, padding: 0 }}>
        {errors.map((err, idx) => (
          <li key={idx} style={{ marginBottom: '0.5rem' }}>{err}</li>
        ))}
      </ul>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState('data');
  const [activeSubTab, setActiveSubTab] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [currentStrategy, setCurrentStrategy] = useState(null);
  const topBarRef = useRef(null);

  useEffect(() => {
    if (topBarRef.current) {
      const height = topBarRef.current.offsetHeight;
      document.documentElement.style.setProperty('--top-bar-height', `${height}px`);
    }
  }, [activeTab, activeSubTab]);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setActiveSubTab(null);
  };

  const handleSubTabChange = (subTab) => {
    setActiveSubTab(subTab);
  };

  const handleOpenModal = (strategy) => {
    setCurrentStrategy(strategy);
    setShowModal(true);
  };

  const handleSidebarAction = (action) => {
    console.log('Sidebar action:', action);
    if (activeTab === 'options' && action === 'showThetaGang') {
      setActiveSubTab('thetagang');
      return;
    }
    // Handle Scanner specific actions
    if (activeTab === 'scanners') {
      switch (activeSubTab) {
        case 'stock-scanner':
          if (action === 'startScan') {
            const scanEvent = new CustomEvent('stockScannerAction', {
              detail: { action: 'startScan' }
            });
            window.dispatchEvent(scanEvent);
          } else if (action === 'saveFilters') {
            const saveEvent = new CustomEvent('stockScannerAction', {
              detail: { action: 'saveFilters' }
            });
            window.dispatchEvent(saveEvent);
          }
          break;
        case 'watchlist':
          if (action === 'addSymbol') {
            const addEvent = new CustomEvent('watchlistAction', {
              detail: { action: 'addSymbol' }
            });
            window.dispatchEvent(addEvent);
          }
          break;
        default:
          break;
      }
    }
  };

  const currentTabConfig = tabsConfig.find(tab => tab.id === activeTab);
  const currentSubTabConfig = activeSubTab ? currentTabConfig.subTabs?.find(sub => sub.id === activeSubTab) : null;
  const sidebarItems = currentSubTabConfig?.sidebar || currentTabConfig?.sidebar || [];
  const hasSidebar = sidebarItems.length > 0;

  const renderContent = () => {
  console.log('renderContent called, activeTab:', activeTab);
    if (activeTab === 'data') {
      return (
  <div style={{ padding: '0 2rem', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', gap: '2rem' }}>
          <div style={{ border: '2px solid var(--border-primary)', borderRadius: '12px', background: 'var(--bg-card)', padding: '2rem', minWidth: '900px', minHeight: '180px', boxShadow: '0 2px 8px rgba(0,0,0,0.07)', display: 'flex', flexDirection: 'column', gap: '2.5rem', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ display: 'flex', flexDirection: 'row', gap: '2.5rem', alignItems: 'center', justifyContent: 'center', width: '100%' }}>
              {['S&P', 'NASDAQ100', 'NYSE', 'RUSSELL2000'].map((name, i) => (
                <label key={name} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1rem' }}>
                  <input type="checkbox" style={{ accentColor: '#2563eb', width: '22px', height: '22px' }} />
                  <button style={{ padding: '1.2rem 2.5rem', borderRadius: '6px', border: '1px solid var(--border-primary)', background: 'var(--bg-button-secondary)', color: 'var(--text-primary)', fontSize: '1.1rem', cursor: 'pointer', minWidth: '120px', minHeight: '60px', display: 'block' }}>
                    {name}
                  </button>
                </label>
              ))}
              <label style={{ fontWeight: 'bold', textAlign: 'right', display: 'flex', alignItems: 'center', gap: '0.5rem', marginLeft: '2rem' }}>
                תאריך מ:
                <input type="date" style={{ marginRight: '1rem', marginLeft: '0.5rem', fontSize: '1rem', padding: '0.25rem', width: '140px' }} />
              </label>
              <label style={{ fontWeight: 'bold', textAlign: 'right', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                תאריך עד:
                <input type="date" style={{ marginRight: '1rem', marginLeft: '0.5rem', fontSize: '1rem', padding: '0.25rem', width: '140px' }} />
              </label>
            </div>
            <div style={{ display: 'flex', flexDirection: 'row', gap: '2rem', alignItems: 'center', justifyContent: 'center', width: '100%' }}>
              <button style={{ padding: '1.2rem 2.5rem', borderRadius: '6px', border: '2px solid #2563eb', background: 'var(--bg-success)', color: 'var(--text-primary)', fontSize: '1.1rem', cursor: 'pointer', minWidth: '120px', minHeight: '60px', fontWeight: 'bold' }}>
                fetch all
              </button>
              <button style={{ padding: '1.2rem 2.5rem', borderRadius: '6px', border: '2px solid #2563eb', background: 'var(--bg-info)', color: 'var(--text-primary)', fontSize: '1.1rem', cursor: 'pointer', minWidth: '120px', minHeight: '60px', fontWeight: 'bold' }}>
                fetch from last
              </button>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="App">
      <ErrorOverlay />
      <StatusBar />
      <TopBar
        tabs={tabsConfig}
        activeTab={activeTab}
        onTabChange={handleTabChange}
        activeSubTab={activeSubTab}
        onSubTabChange={handleSubTabChange}
      />
  <div style={{ display: 'flex', flexDirection: 'row', height: '100vh', marginTop: '136px', position: 'relative' }}>
        {hasSidebar && (
          <Sidebar
            items={sidebarItems}
            onOpenModal={handleOpenModal}
            onAction={handleSidebarAction}
          />
        )}
        <div style={{ flex: 1, marginLeft: '14rem' }}>
          {renderContent()}
        </div>
      </div>
    </div>
  );
}

export default App;