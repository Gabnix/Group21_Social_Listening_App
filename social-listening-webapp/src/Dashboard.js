// Dashboard.js - Fixing the Top Influencer navigation
import React, { useState } from 'react';
import { TrendLineChart, DiseasePieChart } from './DashboardCharts';
import { 
  Search, ChevronRight, User, BarChart2, Clock, 
  Briefcase, PieChart, AlertCircle, Settings, HelpCircle, 
  MessageSquare, LogOut, Menu, X
} from 'lucide-react';
import './Dashboard.css';

// Sample data for the dashboard (unchanged)
const keywordsData = [
  { id: 1, name: 'Diseases #1', trend: '+15.6%', status: 'up' },
  { id: 2, name: 'Diseases #2', trend: '+14%', status: 'up' },
  { id: 3, name: 'Diseases #3', trend: null, status: null },
  { id: 4, name: 'Diseases #4', trend: null, status: null },
];

const sourcesData = [
  { id: 1, name: 'X (Twitter)', trend: '+15.6%', engagement: '+14k', period: 'this week' },
  { id: 2, name: 'Facebook', trend: '+9.2%', engagement: '+8.7k', period: 'this week' },
  { id: 3, name: 'Gov. Websites', trend: '+5.7%', engagement: '-3.6k', period: 'this week' },
];

const diseases = [
  { id: 1, name: 'Diseases #1', trend: '-6.7%', engagement: '-3.6k' },
  { id: 2, name: 'Diseases #2', trend: null, engagement: null },
];

const chartData = [
  { name: '0', western: 100, rest: 150 },
  { name: '1', western: 200, rest: 220 },
  { name: '2', western: 250, rest: 340 },
  { name: '3', western: 180, rest: 100 },
  { name: '4', western: 50, rest: 30 },
  { name: '5', western: 100, rest: 80 },
  { name: '6', western: 250, rest: 150 },
  { name: '7', western: 180, rest: 100 },
  { name: '8', western: 150, rest: 110 },
  { name: '9', western: 170, rest: 130 },
];

const topDiseasesData = [
  { id: 1, name: 'Crop Disease #1', color: '#000000' },
  { id: 2, name: 'Crop Disease #2', color: '#808080' },
  { id: 3, name: 'Crop Disease #3', color: '#E0E0E0' },
];

// Main Dashboard Component
const Dashboard = ({ navigateTo }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedTimeRange, setSelectedTimeRange] = useState('Last 14 Days');
  
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
            <div className="sidebar-item active">
              <BarChart2 size={20} />
              <span>Overview</span>
            </div>
            <div className="sidebar-subitem">
              <span>Summary</span>
            </div>
          </div>
          
          <div className="sidebar-section">
            <div className="sidebar-item">
              <Briefcase size={20} />
              <span>Custom View</span>
            </div>
            <div 
              className="sidebar-subitem"
              onClick={() => navigateTo('influencers')} 
              style={{ cursor: 'pointer' }}
            >
              <span>• Top Influencer</span>
            </div>
            <div className="sidebar-subitem">
              <span>• Top Theme</span>
            </div>
            <div className="sidebar-subitem">
              <span>• Most Engagement</span>
            </div>
          </div>
          
          <div className="sidebar-section">
            <div className="sidebar-item">
              <PieChart size={20} />
              <span>Analysis</span>
            </div>
            <div className="sidebar-subitem">
              <span>• Sentiment</span>
            </div>
            <div className="sidebar-subitem">
              <span>• Seasonal</span>
            </div>
          </div>
          
          <div className="sidebar-section bottom-section">
            <div className="sidebar-item">
              <HelpCircle size={20} />
              <span>Help</span>
            </div>
            <div className="sidebar-item">
              <MessageSquare size={20} />
              <span>Contact Us</span>
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
          <div className="welcome-message">
            <h2>Welcome back</h2>
            <p>Here are today's stats !</p>
          </div>
          
          <div className="nav-actions">
            <div className="search-bar">
              <Search size={20} />
              <input 
                type="text" 
                placeholder="Type '/' to search" 
                onClick={() => navigateTo('advancedSearch')}
              />
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
        
        {/* Dashboard Cards */}
        <div className="dashboard-cards">
          {/* Keywords Card */}
          <div className="dashboard-card">
            <div className="card-header">
              <div className="card-icon keyword-icon">
                <Briefcase size={20} color="white" />
              </div>
              <h3>Keywords</h3>
              <div className="card-count">753 matches</div>
              <ChevronRight className="card-chevron" size={20} />
            </div>
            
            <div className="card-content">
              <div className="disease-grid">
                {keywordsData.map(disease => (
                  <div key={disease.id} className="disease-item">
                    <span className="disease-name">{disease.name}</span>
                    {disease.trend && (
                      <div className={`trend-indicator ${disease.status}`}>
                        <span>{disease.trend}</span>
                        <span className="trend-period">this week</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* Sources Card */}
          <div className="dashboard-card">
            <div className="card-header">
              <div className="card-icon sources-icon">
                <User size={20} color="white" />
              </div>
              <h3>Sources</h3>
              <ChevronRight className="card-chevron" size={20} />
            </div>
            
            <div className="card-content">
              <div className="sources-list">
                {sourcesData.map(source => (
                  <div key={source.id} className="source-item">
                    <span className="source-name">{source.name}</span>
                    <div className="source-stats">
                      <div className="stat-item">
                        <span className={`stat-value ${source.trend.startsWith('+') ? 'up' : 'down'}`}>
                          {source.trend}
                        </span>
                      </div>
                      <div className="stat-item">
                        <span className={`stat-value ${source.engagement.startsWith('+') ? 'up' : 'down'}`}>
                          {source.engagement}
                        </span>
                        <span className="stat-period">{source.period}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* Search and Analysis Card */}
          <div className="dashboard-card">
            <div className="card-header">
              <div className="card-icon search-icon">
                <Search size={20} color="white" />
              </div>
              <h3>Search and Analysis</h3>
              <ChevronRight className="card-chevron" size={20} />
            </div>
            
            <div className="card-content">
              <div className="diseases-list">
                {diseases.map(disease => (
                  <div key={disease.id} className="disease-item">
                    <span className="disease-name">{disease.name}</span>
                    {disease.trend && (
                      <div className="disease-stats">
                        <div className="stat-item">
                          <span className={`stat-value ${disease.trend.startsWith('+') ? 'up' : 'down'}`}>
                            {disease.trend}
                          </span>
                        </div>
                        <div className="stat-item">
                          <span className={`stat-value ${disease.engagement.startsWith('+') ? 'up' : 'down'}`}>
                            {disease.engagement}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
                <div className="disease-item">
                  <span className="disease-name">Diseases #2</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Dashboard Charts */}
        <div className="dashboard-charts">
          {/* Trends Chart */}
          <div className="chart-card">
            <div className="chart-header">
              <h3>Latest Trends</h3>
              <div className="chart-actions">
                <div className="search-mini">
                  <Search size={16} />
                  <span>Search</span>
                </div>
                
                <div className="time-selector">
                  <Clock size={16} />
                  <span>{selectedTimeRange}</span>
                </div>
                
                <div className="export-button">
                  <span>Export Data</span>
                </div>
              </div>
            </div>
            
            <div className="chart-content">
              <div className="chart-legend">
                <div className="legend-item">
                  <div className="legend-color" style={{ backgroundColor: '#4CAF50' }}></div>
                  <span>Western Australia</span>
                </div>
                <div className="legend-item">
                  <div className="legend-color" style={{ backgroundColor: '#9C27B0' }}></div>
                  <span>Rest of Australia</span>
                </div>
              </div>
              
              <div className="chart-container">
                <TrendLineChart />
              </div>
            </div>
          </div>
          
          {/* Top Diseases Pie Chart */}
          <div className="chart-card pie-chart-card">
            <div className="chart-header">
              <h3>Top Diseases</h3>
            </div>
            
            <div className="chart-content">
              <div className="pie-chart-container">
                <div className="disease-stat-container">
                  <div className="pie-stat">
                    <span className="pie-value">6.5k</span>
                    <span className="pie-label">matches</span>
                  </div>
                </div>
                <DiseasePieChart />
              </div>
              
              <div className="pie-legend">
                {topDiseasesData.map(disease => (
                  <div key={disease.id} className="legend-item">
                    <div className="legend-color" style={{ backgroundColor: disease.color }}></div>
                    <span>{disease.name}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;