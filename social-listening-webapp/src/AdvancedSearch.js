// AdvancedSearch.js
import React, { useState } from 'react';
import { 
  Search, ChevronLeft, User, BarChart2, 
  Briefcase, PieChart, AlertCircle, Settings, HelpCircle, 
  MessageSquare, LogOut, Menu, X, Mail, Grid, FileText
} from 'lucide-react';
import './AdvancedSearch.css';

const AdvancedSearch = ({ navigateTo }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [searchQuery, setSearchQuery] = useState('"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR\n"Colletotrichum dematium" OR');
  const [activeTab, setActiveTab] = useState('advanced');
  
  // Operators with their symbols
  const operators = [
    { name: 'AND', symbol: '+' },
    { name: 'OR', symbol: '+' },
    { name: 'AND NOT', symbol: '+' },
    { name: 'PREF', symbol: '+' },
    { name: 'W/', symbol: '+' },
  ];

  return (
    <div className="dashboard-container">
      {/* Sidebar */}
      <div className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
        <div className="sidebar-header">
          <h2>Social Listening 21</h2>
          <button className="toggle-button" onClick={() => setSidebarOpen(!sidebarOpen)}>
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
        
        <div className="sidebar-content">
          <div className="sidebar-section">
            <div className="sidebar-item">
              <FileText size={20} />
              <span>Overview</span>
            </div>
          </div>
          
          <div className="sidebar-section">
            <div 
              className="sidebar-item"
              onClick={() => navigateTo('dashboard')}
              style={{ cursor: 'pointer' }}
            >
              <Grid size={20} />
              <span>Dashboard</span>
            </div>
          </div>
          
          <div className="sidebar-section">
            <div className="sidebar-item active">
              <Search size={20} />
              <span>Advanced Search</span>
            </div>
          </div>
          
          <div className="sidebar-section">
            <div className="sidebar-item">
              <Mail size={20} />
              <span>Messages</span>
            </div>
          </div>
          
          <div className="sidebar-section">
            <div className="sidebar-item">
              <PieChart size={20} />
              <span>Analysis</span>
            </div>
          </div>
          
          <div className="sidebar-section">
            <div className="sidebar-item">
              <BarChart2 size={20} />
              <span>Reports</span>
            </div>
          </div>
          
          <div className="sidebar-section bottom-section">
            <div className="sidebar-item">
              <HelpCircle size={20} />
              <span>Help</span>
            </div>
            <div className="sidebar-item">
              <Settings size={20} />
              <span>Settings</span>
            </div>
            <div className="sidebar-item">
              <LogOut size={20} />
              <span>Log out</span>
            </div>
          </div>
        </div>
      </div>
      
      {/* Main Content */}
      <div className={`main-content ${!sidebarOpen ? 'expanded' : ''}`}>
        {/* Top Navigation */}
        <div className="top-nav">
          <div className="search-title">
            <h2>Advanced Search</h2>
          </div>
          
          <div className="nav-actions">
            <div className="search-bar">
              <Search size={20} />
              <input type="text" placeholder='Type "/" to search' />
            </div>
            
            <div className="notification-icon">
              <AlertCircle size={20} />
            </div>
            
            <div className="user-profile">
              <div className="user-avatar">
                <User size={20} />
              </div>
              <span>Student</span>
            </div>
          </div>
        </div>
        
        {/* Advanced Search Content */}
        <div className="search-container">
          <div className="search-card">
            <div className="search-tabs">
              <div 
                className={`search-tab ${activeTab === 'basic' ? 'active' : ''}`}
                onClick={() => navigateTo('dashboard')}
              >
                <ChevronLeft size={16} />
                <span>Basic Search</span>
              </div>
              <div 
                className={`search-tab ${activeTab === 'advanced' ? 'active' : ''}`}
                onClick={() => setActiveTab('advanced')}
              >
                <span>Advanced</span>
              </div>
            </div>
            
            <div className="search-content">
              <div className="search-form">
                <div className="form-group">
                  <label htmlFor="queryString">Enter query string</label>
                  <textarea
                    id="queryString"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    rows={12}
                    className="query-textarea"
                  />
                </div>
                
                <div className="search-actions">
                  <div className="search-options">
                    <button className="option-button">Outline query</button>
                    <button className="option-button">Add influencer name / Source</button>
                  </div>
                  <button className="search-button">Search</button>
                </div>
              </div>
              
              <div className="search-operators">
                <h3>Operators</h3>
                <ul className="operators-list">
                  {operators.map((operator, index) => (
                    <li key={index} className="operator-item">
                      <span className="operator-name">{operator.name}</span>
                      <span className="operator-symbol">{operator.symbol}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdvancedSearch;