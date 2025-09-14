import React, { useState, useRef, useEffect } from 'react';
   import StatusBar from './components/StatusBar';
   import TopBar from './components/TopBar';
   import Sidebar from './components/Sidebar';
   import ParametersModal from './components/ParametersModal';
   import Dashboard from './components/Dashboard';
   import { tabsConfig } from './config/tabsConfig';
   import './styles/GlobalStyles.css';

   function App() {
     const [activeTab, setActiveTab] = useState('dashboard');
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

     const currentTabConfig = tabsConfig.find(tab => tab.id === activeTab);
     const currentSubTabConfig = activeSubTab ? currentTabConfig.subTabs?.find(sub => sub.id === activeSubTab) : null;
     const sidebarItems = currentSubTabConfig?.sidebar || currentTabConfig?.sidebar || [];

     return (
       <div className="app">
         <StatusBar />
         <div ref={topBarRef}>
           <TopBar
             tabs={tabsConfig}
             activeTab={activeTab}
             onTabChange={handleTabChange}
             activeSubTab={activeSubTab}
             onSubTabChange={handleSubTabChange}
           />
         </div>
         <div className="main-content">
           {sidebarItems.length > 0 && (
             <Sidebar items={sidebarItems} onOpenModal={handleOpenModal} />
           )}
           <main className="content-area">
             {activeTab === 'dashboard' && <Dashboard />}
           </main>
         </div>
         {showModal && (
           <ParametersModal
             strategy={currentStrategy}
             onClose={() => setShowModal(false)}
           />
         )}
       </div>
     );
   }

   export default App;