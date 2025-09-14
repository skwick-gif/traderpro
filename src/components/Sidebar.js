import React from 'react';
   import styles from './Sidebar.module.css';

   function Sidebar({ items, onOpenModal }) {
     return (
       <aside className={styles.sidebar}>
         {items.map(item => (
           <button
             key={item.id}
             className={styles.sidebarButton}
             onClick={() => item.action === 'updateParameters' ? onOpenModal(item.strategy) : console.log(item.action)}
           >
             {item.label}
           </button>
         ))}
       </aside>
     );
   }

   export default Sidebar;