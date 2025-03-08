import React, { useState } from 'react';
import './SearchBar.css';

const SearchBar = ({ onSearch }) => {
  const [keyword, setKeyword] = useState('');
  const [subreddits, setSubreddits] = useState('');
  const [timeframe, setTimeframe] = useState('all');
  const [sort, setSort] = useState('relevance');
  const [showHelp, setShowHelp] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (keyword.trim()) {
      onSearch({
        keyword: keyword.trim(),
        subreddits: subreddits.trim(),
        timeframe,
        sort,
        includeNsfw: false
      });
    }
  };

  return (
    <div className="search-bar">
      <form onSubmit={handleSubmit}>
        <div className="search-main">
          <div className="search-input-container">
            <input
              type="text"
              placeholder="Example: pesticide AND fungicide"
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
              className="search-input"
              required
            />
            <button 
              type="button" 
              className="help-button"
              onClick={() => setShowHelp(!showHelp)}
            >
              ?
            </button>
          </div>
          <button type="submit" className="search-button">
            Search Reddit
          </button>
        </div>

        {showHelp && (
          <div className="search-help">
            <h4>Search Tips:</h4>
            <ul>
              <li>Use quotes for exact phrases: "crop diseases"</li>
              <li>Use AND to require all terms: "crop diseases" AND pesticide</li>
              <li>Use OR for alternatives: pesticide OR herbicide</li>
              <li>Combine operators: "organic farming" AND (pesticide OR herbicide)</li>
              <li>Default operator is AND if none specified</li>
            </ul>
          </div>
        )}
        
        <div className="search-options">
          <div className="option-group">
            <label>Subreddits (optional, comma-separated):</label>
            <input
              type="text"
              placeholder="e.g., science,technology,news"
              value={subreddits}
              onChange={(e) => setSubreddits(e.target.value)}
              className="subreddit-input"
            />
          </div>
          
          <div className="option-group">
            <label>Time Range:</label>
            <select 
              value={timeframe} 
              onChange={(e) => setTimeframe(e.target.value)}
              className="select-input"
            >
              <option value="hour">Past Hour</option>
              <option value="day">Past 24 Hours</option>
              <option value="week">Past Week</option>
              <option value="month">Past Month</option>
              <option value="year">Past Year</option>
              <option value="all">All Time</option>
            </select>
          </div>
          
          <div className="option-group">
            <label>Sort By:</label>
            <select 
              value={sort} 
              onChange={(e) => setSort(e.target.value)}
              className="select-input"
            >
              <option value="relevance">Relevance</option>
              <option value="hot">Hot</option>
              <option value="top">Top</option>
              <option value="new">New</option>
              <option value="comments">Most Comments</option>
            </select>
          </div>
        </div>
      </form>
    </div>
  );
};

export default SearchBar; 