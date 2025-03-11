// Influencers.js
import React, { useState } from 'react';
import { 
  Search, User, BarChart2, 
  Briefcase, PieChart, AlertCircle, Settings, HelpCircle, 
  MessageSquare, LogOut, Menu, X, Mail, Grid, FileText,
  Download, RefreshCw, LayoutGrid, Printer, HelpCircleIcon, MoreVertical, Plus
} from 'lucide-react';
import MediaTypeChart from './MediaChart';
import './Influencers.css';

// Sample data for the influencers table
const influencersData = [
  { 
    id: 1, 
    name: 'grain control', 
    url: 'http//www.graincentral.com', 
    network: 'twitter',
    posts: 5,
    reach: '113.2k',
    reachPerMention: '43.5k',
    engagement: 5,
    engagementPerMention: 1.4
  },
  { 
    id: 2, 
    name: 'dr sue insights', 
    url: 'http//www.groundcover.grdc.com.au', 
    network: 'twitter',
    posts: 3,
    reach: '100.0k',
    reachPerMention: '40.5k',
    engagement: 2,
    engagementPerMention: 2.3
  },
  { 
    id: 3, 
    name: 'lachstock consulting', 
    url: 'http//www.com', 
    network: 'twitter',
    posts: 3,
    reach: '1.2M',
    reachPerMention: '43.5k',
    engagement: 5,
    engagementPerMention: 1.4
  },
  { 
    id: 4, 
    name: 'Catherine norwood', 
    url: 'http//www.graincentral.com', 
    network: 'twitter',
    posts: 2,
    reach: '113.2k',
    reachPerMention: '43.5k',
    engagement: 5,
    engagementPerMention: 1.4
  },
  { 
    id: 5, 
    name: 'pr newswire (us)', 
    url: 'http//www.graincentral.com', 
    network: 'twitter',
    posts: 2,
    reach: '113.2k',
    reachPerMention: '43.5k',
    engagement: 5,
    engagementPerMention: 1.4
  },
  { 
    id: 6, 
    name: 'brantinus biotherapeutics', 
    url: 'http//www.graincentral.com', 
    network: 'twitter',
    posts: 1,
    reach: '113.2k',
    reachPerMention: '43.5k',
    engagement: 5,
    engagementPerMention: 1.4
  },
  { 
    id: 7, 
    name: 'collac canada', 
    url: 'http//www.graincentral.com', 
    network: 'twitter',
    posts: 2,
    reach: '113.2k',
    reachPerMention: '43.5k',
    engagement: 5,
    engagementPerMention: 1.4
  },
  { 
    id: 8, 
    name: 'research and markets', 
    url: 'http//www.graincentral.com', 
    network: 'twitter',
    posts: 6,
    reach: '113.2k',
    reachPerMention: '43.5k',
    engagement: 5,
    engagementPerMention: 1.4
  },
  { 
    id: 9, 
    name: 'sultan khalid', 
    url: 'http//www.graincentral.com', 
    network: 'twitter',
    posts: 3,
    reach: '113.2k',
    reachPerMention: '43.5k',
    engagement: 5,
    engagementPerMention: 1.4
  },
  { 
    id: 10, 
    name: 'grain control', 
    url: 'http//www.graincentral.com', 
    network: 'twitter',
    posts: 5,
    reach: '113.2k',
    reachPerMention: '43.5k',
    engagement: 5,
    engagementPerMention: 1.4
  }
];

// Media types data for the pie chart
const mediaTypeData = [
  { name: 'Twitter', value: 40, color: '#1DA1F2' },
  { name: 'Blog', value: 25, color: '#2ECC71' },
  { name: 'Facebook', value: 15, color: '#E67E22' },
  { name: 'Reddit', value: 12, color: '#474747' },
  { name: 'Others', value: 8, color: '#FDC84A' }
];

const Influencers = ({ navigateTo }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [searchQuery, setSearchQuery] = useState('Barley AND disease');
  const [activeTab, setActiveTab] = useState('top');
  
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
            <div 
              className="sidebar-item"
              onClick={() => navigateTo('dashboard')}
              style={{ cursor: 'pointer' }}
            >
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
            <div 
              className="sidebar-item"
              onClick={() => navigateTo('advancedSearch')}
              style={{ cursor: 'pointer' }}
            >
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
            <div className="sidebar-item active">
              <PieChart size={20} />
              <span>Influencers</span>
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
          <div className="influencer-title">
            <h2>Influencers</h2>
          </div>
          
          <div className="nav-actions">
            <div className="search-bar">
              <Search size={20} />
              <input
                type="text" 
                placeholder='Type "/" to search'
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
        
        {/* Influencers Content */}
        <div className="influencers-container">
          {/* Action Bar */}
          <div className="action-bar">
            <div className="search-field">
              <Search size={18} />
              <input 
                type="text" 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search for influencers..."
              />
            </div>
            
            <div className="action-buttons">
              <button className="action-button new-search">New Search</button>
              <button className="action-button saved">Saved Searches</button>
              <button className="action-icon-button"><RefreshCw size={18} /></button>
              <button className="action-icon-button"><LayoutGrid size={18} /></button>
              <button className="action-icon-button"><Download size={18} /></button>
              <button className="action-icon-button"><Printer size={18} /></button>
              <button className="action-icon-button"><HelpCircleIcon size={18} /></button>
            </div>
          </div>
          
          {/* Main Content Area */}
          <div className="influencer-content">
            {/* Search Tag */}
            <div className="search-tag-container">
              <div className="search-tag">
                <span>{searchQuery}</span>
                <button className="more-options"><MoreVertical size={16} /></button>
              </div>
              
              <div className="compare-button">
                <Plus size={18} />
                <span>Compare</span>
              </div>
            </div>
            
            {/* Table and Stats Container */}
            <div className="table-stats-container">
              {/* Influencers Table Section */}
              <div className="influencers-table-section">
                {/* Tabs */}
                <div className="influencer-tabs">
                  <div 
                    className={`influencer-tab ${activeTab === 'top' ? 'active' : ''}`}
                    onClick={() => setActiveTab('top')}
                  >
                    <span>TOP INFLUENCERS</span>
                  </div>
                  <div 
                    className={`influencer-tab ${activeTab === 'unique' ? 'active' : ''}`}
                    onClick={() => setActiveTab('unique')}
                  >
                    <span>UNIQUE AUTHORS</span>
                  </div>
                </div>
                
                {/* Table */}
                <div className="influencers-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Influencer</th>
                        <th>Network</th>
                        <th>Posts</th>
                        <th>Reach</th>
                        <th>Reach per mention</th>
                        <th>Engagement</th>
                        <th>Engagement per mention</th>
                      </tr>
                    </thead>
                    <tbody>
                      {influencersData.map(influencer => (
                        <tr key={influencer.id}>
                          <td className="influencer-cell">
                            <div className="influencer-info">
                              <div className="influencer-avatar">
                                <User size={18} />
                              </div>
                              <div className="influencer-details">
                                <div className="influencer-name">{influencer.name}</div>
                                <div className="influencer-url">{influencer.url}</div>
                              </div>
                            </div>
                          </td>
                          <td className="network-cell">
                            <div className="network-icon">
                              <User size={16} />
                            </div>
                          </td>
                          <td>{influencer.posts}</td>
                          <td>{influencer.reach}</td>
                          <td>{influencer.reachPerMention}</td>
                          <td>{influencer.engagement}</td>
                          <td>{influencer.engagementPerMention}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              
              {/* Stats Section */}
              <div className="stats-section">
                <div className="authors-count">
                  <h2>28</h2>
                </div>
                
                <div className="media-types">
                  <h3>SHARE OF MEDIA TYPES</h3>
                  <div className="pie-chart">
                    <MediaTypeChart data={mediaTypeData} />
                  </div>
                  
                  <div className="media-legend">
                    {mediaTypeData.map((media, index) => (
                      <div key={index} className="legend-item">
                        <div className="legend-color" style={{ backgroundColor: media.color }}></div>
                        <div className="legend-text">{media.name}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Influencers;