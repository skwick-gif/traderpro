export const tabsConfig = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    sidebar: [
      { id: 'refresh', label: 'Refresh Data', action: 'refresh' },
      { id: 'charts', label: 'View Charts', action: 'showCharts' }
    ]
  },
  {
    id: 'scanners',
    label: 'Scanners',
    subTabs: [
      { id: 'stock-scanner', label: 'Stock Scanner', sidebar: [
        { id: 'scan', label: 'Start Scan', action: 'startScan' },
        { id: 'save', label: 'Save Filters', action: 'saveFilters' }
      ]},
      { id: 'option-scanner', label: 'Option Scanner', sidebar: [
        { id: 'scan', label: 'Start Scan', action: 'startScan' }
      ]},
      { id: 'watchlist', label: 'Watchlist', sidebar: [
        { id: 'add', label: 'Add Symbol', action: 'addSymbol' }
      ]}
    ]
  },
  {
    id: 'strategies',
    label: 'Strategies',
    subTabs: [
      { id: 'strangle', label: 'Short Strangle', sidebar: [
        { id: 'params', label: 'Update Parameters', action: 'updateParameters', strategy: 'strangle' }
      ], parameters: ['delta_target', 'dte_entry'] },
      { id: 'condor', label: 'Iron Condor', sidebar: [
        { id: 'params', label: 'Update Parameters', action: 'updateParameters', strategy: 'condor' }
      ], parameters: ['strike_width', 'profit_target'] },
      { id: 'spread', label: 'Credit Spread', sidebar: [
        { id: 'params', label: 'Update Parameters', action: 'updateParameters', strategy: 'spread' }
      ], parameters: ['strike_distance', 'premium'] }
    ]
  },
  {
    id: 'tools',
    label: 'Tools',
    subTabs: [
      { id: 'settings', label: 'Settings', sidebar: [
        { id: 'update', label: 'Update Settings', action: 'updateSettings' }
      ]},
      { id: 'analysis', label: 'Analysis', sidebar: [] },
      { id: 'backtest', label: 'Backtest', sidebar: [
        { id: 'run', label: 'Run Backtest', action: 'runBacktest' }
      ]}
    ]
  }
];