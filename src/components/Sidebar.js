import React from 'react';
import styles from './Sidebar.module.css';

function Sidebar({ items, onOpenModal, onAction }) {
  const handleClick = (item) => {
    if (item.action === 'updateParameters') {
      onOpenModal(item.strategy);
    } else if (onAction) {
      onAction(item.action);
    } else {
      console.log(item.action);
    }
  };

  return (
    <aside className={styles.sidebar}>
      {items.map(item => (
        <button
          key={item.id}
          className={styles.sidebarButton}
          onClick={() => handleClick(item)}
        >
          {item.label}
        </button>
      ))}
    </aside>
  );
}

export default Sidebar;