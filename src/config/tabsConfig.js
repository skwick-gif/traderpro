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
    sidebar: [
      { id: 'scan', label: 'Start Scan', action: 'startScan' },
      { id: 'save', label: 'Save Filters', action: 'saveFilters' },
      { id: 'add', label: 'Add Symbol', action: 'addSymbol' }
    ]
  },
  {
    id: 'options',
    label: 'Options',
    sidebar: [
      { id: 'thetagang', label: 'ThetaGang', action: 'showThetaGang' },
      { id: 'strangle', label: 'Short Strangle', action: 'updateParameters' },
      { id: 'condor', label: 'Iron Condor', action: 'updateParameters' },
      { id: 'spread', label: 'Credit Spread', action: 'updateParameters' },
      { id: 'stop', label: 'Stop ThetaGang', action: 'stopThetaGang' },
      { id: 'refresh', label: 'Refresh Data', action: 'refreshThetaGang' }
    ]
  },
  {
    id: 'data',
    label: 'Data',
    sidebar: [
      { id: 'download', label: 'Download', action: 'download' }
    ]
  }
];