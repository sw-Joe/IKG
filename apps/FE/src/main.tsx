import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx'; // Your main component
import './index.css'; // Global styles

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
