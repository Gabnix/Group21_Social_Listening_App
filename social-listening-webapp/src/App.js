// App.js
import React, { useState } from 'react';
import Dashboard from './Dashboard';
import AdvancedSearch from './AdvancedSearch';
import Influencers from './Influencers';

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');

  // Function to switch between pages
  const navigateTo = (page) => {
    setCurrentPage(page);
  };

  return (
    <div className="App">
      {currentPage === 'dashboard' && <Dashboard navigateTo={navigateTo} />}
      {currentPage === 'advancedSearch' && <AdvancedSearch navigateTo={navigateTo} />}
      {currentPage === 'influencers' && <Influencers navigateTo={navigateTo} />}
    </div>
  );
}

export default App;