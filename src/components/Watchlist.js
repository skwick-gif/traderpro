import React, { useState, useEffect } from 'react';
import { 
  Plus, Trash2, Star, StarOff, Search, Download, RefreshCw, X
} from 'lucide-react';
import styles from './Watchlist.module.css';

const AVAILABLE_COLUMNS = [
  { id: 'symbol', label: 'Symbol', type: 'text', sortable: true, width: 100, defaultVisible: true },
  { id: 'price', label: 'Price', type: 'currency', sortable: true, width: 100, defaultVisible: true },
  { id: 'change', label: 'Change', type: 'percentage', sortable: true, width: 80, defaultVisible: true },
  { id: 'changePercent', label: 'Change %', type: 'percentage', sortable: true, width: 80, defaultVisible: true },
  { id: 'volume', label: 'Volume', type: 'number', sortable: true, width: 100, defaultVisible: true },
  { id: 'score', label: 'Score', type: 'score', sortable: true, width: 80, defaultVisible: true }
];

const DEFAULT_VISIBLE_COLUMNS = AVAILABLE_COLUMNS
  .filter(col => col.defaultVisible)
  .map(col => col.id);

function Watchlist() {
  // All state variables MUST be defined
  const [watchlistItems, setWatchlistItems] = useState([]);
  const [filteredItems, setFilteredItems] = useState([]);
  const [visibleColumns, setVisibleColumns] = useState(DEFAULT_VISIBLE_COLUMNS);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: 'symbol', direction: 'asc' });
  const [showAddModal, setShowAddModal] = useState(false);
  const [newSymbol, setNewSymbol] = useState('');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [filterBy, setFilterBy] = useState('all');
  const [errorMessage, setErrorMessage] = useState(''); // THIS WAS MISSING

  // Filter and search effect
  useEffect(() => {
    let filtered = watchlistItems;

    if (searchTerm) {
      filtered = filtered.filter(item =>
        item.symbol.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (filterBy !== 'all') {
      filtered = filtered.filter(item => {
        switch (filterBy) {
          case 'gainers':
            return item.changePercent > 0;
          case 'losers':
            return item.changePercent < 0;
          case 'favorites':
            return item.isFavorite;
          default:
            return true;
        }
      });
    }

    if (sortConfig.key) {
      filtered.sort((a, b) => {
        const aVal = a[sortConfig.key];
        const bVal = b[sortConfig.key];
        
        if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1;
        if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1;
        return 0;
      });
    }

    setFilteredItems(filtered);
  }, [watchlistItems, searchTerm, filterBy, sortConfig]);

  // Listen for external events
  useEffect(() => {
    const handleExternalAdd = (event) => {
      if (event.detail.action === 'addSymbol') {
        setShowAddModal(true);
      }
    };
    
    window.addEventListener('watchlistAction', handleExternalAdd);
    
    return () => {
      window.removeEventListener('watchlistAction', handleExternalAdd);
    };
  }, []);

  const handleSort = (columnId) => {
    setSortConfig(prev => ({
      key: columnId,
      direction: prev.key === columnId && prev.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const handleAddSymbol = async () => {
    if (!newSymbol.trim()) {
      setErrorMessage('Please enter a symbol');
      return;
    }

    const symbolUpper = newSymbol.toUpperCase();
    
    if (watchlistItems.find(item => item.symbol === symbolUpper)) {
      setErrorMessage('Symbol already exists in watchlist');
      return;
    }

    try {
      setErrorMessage('');
      console.log('Fetching data for:', symbolUpper);
      
      const response = await fetch(`http://localhost:8000/api/stock/${symbolUpper}`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: Failed to fetch stock data`);
      }
      
      const stockData = await response.json();
      
      if (!stockData.data_available) {
        throw new Error('No data available for this symbol');
      }

      const newItem = {
        symbol: symbolUpper,
        price: stockData.current_price || 0,
        change: 0,
        changePercent: 0,
        volume: stockData.volume || 0,
        score: 0,
        isFavorite: false,
        addedDate: new Date().toISOString().split('T')[0]
      };

      setWatchlistItems(prev => [...prev, newItem]);
      setNewSymbol('');
      setShowAddModal(false);
      setErrorMessage('');
      
    } catch (error) {
      console.error('Error adding symbol:', error);
      setErrorMessage(`Failed to add ${symbolUpper}: ${error.message}`);
    }
  };

  const handleRemoveSymbol = (symbol) => {
    setWatchlistItems(prev => prev.filter(item => item.symbol !== symbol));
    setErrorMessage('');
  };

  const handleToggleFavorite = (symbol) => {
    setWatchlistItems(prev => 
      prev.map(item =>
        item.symbol === symbol 
          ? { ...item, isFavorite: !item.isFavorite }
          : item
      )
    );
  };

  const refreshWatchlist = async () => {
    setIsRefreshing(true);
    setErrorMessage('');
    
    try {
      // Refresh each symbol
      for (const item of watchlistItems) {
        try {
          const response = await fetch(`http://localhost:8000/api/stock/${item.symbol}`);
          if (response.ok) {
            const data = await response.json();
            setWatchlistItems(prev => 
              prev.map(prevItem =>
                prevItem.symbol === item.symbol
                  ? { ...prevItem, price: data.current_price, volume: data.volume }
                  : prevItem
              )
            );
          }
        } catch (error) {
          console.error(`Error refreshing ${item.symbol}:`, error);
        }
      }
    } catch (error) {
      console.error('Error refreshing watchlist:', error);
      setErrorMessage('Failed to refresh some symbols');
    } finally {
      setTimeout(() => setIsRefreshing(false), 1000);
    }
  };

  const exportWatchlist = () => {
    if (watchlistItems.length === 0) {
      setErrorMessage('No symbols to export');
      return;
    }
    
    const csvContent = [
      ['Symbol', 'Price', 'Volume', 'Added Date'].join(','),
      ...watchlistItems.map(item => [
        item.symbol,
        item.price,
        item.volume,
        item.addedDate
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `watchlist_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const formatValue = (value, type) => {
    if (value === null || value === undefined) return '-';

    switch (type) {
      case 'currency':
        return `$${Number(value).toFixed(2)}`;
      case 'percentage':
        return `${Number(value).toFixed(2)}%`;
      case 'number':
        if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
        if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
        if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
        return Number(value).toFixed(0);
      case 'score':
        return `${Number(value).toFixed(1)}`;
      default:
        return value;
    }
  };

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <h2 className={styles.title}>
            Watchlist ({filteredItems.length})
          </h2>
        </div>
        <div className={styles.headerRight}>
          <div className={styles.searchContainer}>
            <Search className={styles.searchIcon} />
            <input
              type="text"
              placeholder="Search symbols..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={styles.searchInput}
            />
          </div>
          
          <select
            value={filterBy}
            onChange={(e) => setFilterBy(e.target.value)}
            className={styles.filterSelect}
          >
            <option value="all">All Stocks</option>
            <option value="gainers">Gainers</option>
            <option value="losers">Losers</option>
            <option value="favorites">Favorites</option>
          </select>

          <button
            onClick={() => setShowAddModal(true)}
            className={`${styles.actionBtn} ${styles.addBtn}`}
          >
            <Plus size={16} />
            Add Symbol
          </button>

          <button
            onClick={refreshWatchlist}
            disabled={isRefreshing}
            className={`${styles.actionBtn} ${styles.refreshBtn}`}
          >
            <RefreshCw className={isRefreshing ? styles.spinning : ''} size={16} />
            Refresh
          </button>

          <button
            onClick={exportWatchlist}
            className={`${styles.actionBtn} ${styles.exportBtn}`}
          >
            <Download size={16} />
            Export
          </button>
        </div>
      </div>

      {/* Error Message */}
      {errorMessage && (
        <div className={styles.errorMessage}>
          <span>{errorMessage}</span>
          <button onClick={() => setErrorMessage('')} className={styles.closeError}>
            <X size={14} />
          </button>
        </div>
      )}

      {/* Watchlist Table */}
      <div className={styles.tableContainer}>
        <div className={styles.tableHeader}>
          {visibleColumns.map(columnId => {
            const column = AVAILABLE_COLUMNS.find(col => col.id === columnId);
            return (
              <div
                key={columnId}
                className={`${styles.headerCell} ${column.sortable ? styles.sortable : ''}`}
                onClick={() => column.sortable && handleSort(columnId)}
              >
                <span>{column.label}</span>
                {column.sortable && sortConfig.key === columnId && (
                  <span className={styles.sortIcon}>
                    {sortConfig.direction === 'asc' ? '↑' : '↓'}
                  </span>
                )}
              </div>
            );
          })}
          <div className={styles.headerCell}>Actions</div>
        </div>

        <div className={styles.tableBody}>
          {filteredItems.map((item) => (
            <div key={item.symbol} className={styles.tableRow}>
              {visibleColumns.map(columnId => {
                const column = AVAILABLE_COLUMNS.find(col => col.id === columnId);
                let cellClass = styles.cell;
                let value = formatValue(item[columnId], column.type);
                
                if (columnId === 'change' || columnId === 'changePercent') {
                  cellClass += item[columnId] >= 0 ? ` ${styles.positive}` : ` ${styles.negative}`;
                }
                
                return (
                  <div key={columnId} className={cellClass}>
                    {columnId === 'symbol' ? (
                      <div className={styles.symbolCell}>
                        <span className={styles.symbolText}>{value}</span>
                        {item.isFavorite && <Star className={styles.favoriteIcon} size={12} />}
                      </div>
                    ) : (
                      value
                    )}
                  </div>
                );
              })}
              
              <div className={styles.cell}>
                <div className={styles.actions}>
                  <button
                    onClick={() => handleToggleFavorite(item.symbol)}
                    className={`${styles.actionIcon} ${item.isFavorite ? styles.favorited : ''}`}
                  >
                    {item.isFavorite ? <Star size={14} /> : <StarOff size={14} />}
                  </button>
                  
                  <button
                    onClick={() => handleRemoveSymbol(item.symbol)}
                    className={`${styles.actionIcon} ${styles.remove}`}
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Add Symbol Modal */}
      {showAddModal && (
        <div className={styles.modal}>
          <div className={styles.modalContent}>
            <div className={styles.modalHeader}>
              <h3>Add Symbol to Watchlist</h3>
              <button
                onClick={() => {
                  setShowAddModal(false);
                  setErrorMessage('');
                }}
                className={styles.closeButton}
              >
                <X size={16} />
              </button>
            </div>
            <div className={styles.modalBody}>
              <input
                type="text"
                placeholder="Enter symbol (e.g., AAPL)"
                value={newSymbol}
                onChange={(e) => setNewSymbol(e.target.value)}
                className={styles.symbolInput}
                onKeyPress={(e) => e.key === 'Enter' && handleAddSymbol()}
                autoFocus
              />
            </div>
            <div className={styles.modalFooter}>
              <button
                onClick={() => {
                  setShowAddModal(false);
                  setErrorMessage('');
                }}
                className={`${styles.button} ${styles.cancel}`}
              >
                Cancel
              </button>
              <button
                onClick={handleAddSymbol}
                className={`${styles.button} ${styles.add}`}
                disabled={!newSymbol.trim()}
              >
                Add Symbol
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Empty State */}
      {filteredItems.length === 0 && !isRefreshing && (
        <div className={styles.emptyState}>
          <h3>No stocks in watchlist</h3>
          <p>Add some symbols to get started</p>
          <button
            onClick={() => setShowAddModal(true)}
            className={`${styles.button} ${styles.add}`}
          >
            <Plus size={16} />
            Add Your First Symbol
          </button>
        </div>
      )}
    </div>
  );
}

export default Watchlist;