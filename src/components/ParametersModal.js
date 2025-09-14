import React, { useState } from 'react';
     import styles from './ParametersModal.module.css';

     function ParametersModal({ strategy, onClose }) {
       const [formData, setFormData] = useState({});

       const handleChange = (key, value) => {
         setFormData(prev => ({ ...prev, [key]: value }));
       };

       const handleSubmit = (e) => {
         e.preventDefault();
         console.log('Updated parameters:', formData);
         onClose();
       };

       const parameters = strategy === 'strangle' ? [
         { key: 'delta_target', label: 'Delta Target', type: 'number', step: '0.01', default: 0.16 },
         { key: 'dte_entry', label: 'DTE Entry', type: 'number', default: 45 }
       ] : [];

       return (
         <div className={styles.modalOverlay} onClick={onClose}>
           <div className={styles.modal} onClick={e => e.stopPropagation()}>
             <h2 className={styles.title}>Update {strategy} Parameters</h2>
             <form onSubmit={handleSubmit} className={styles.form}>
               {parameters.map(param => (
                 <div key={param.key} className={styles.field}>
                   <label className={styles.label}>{param.label}</label>
                   <input
                     type={param.type}
                     step={param.step}
                     value={formData[param.key] || param.default}
                     onChange={(e) => handleChange(param.key, param.type === 'number' ? parseFloat(e.target.value) : e.target.value)}
                     className={styles.input}
                   />
                 </div>
               ))}
               <div className={styles.buttons}>
                 <button type="button" onClick={onClose} className={styles.cancelButton}>
                   Cancel
                 </button>
                 <button type="submit" className={styles.saveButton}>
                   Save
                 </button>
               </div>
             </form>
           </div>
         </div>
       );
     }

     export default ParametersModal;