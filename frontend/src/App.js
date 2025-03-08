import React, { useState } from 'react';
import './App.css';
import SearchBar from './components/SearchBar';
import PostList from './components/PostList';

function App() {
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searched, setSearched] = useState(false);
  const [searchMetadata, setSearchMetadata] = useState(null);
  
  const searchReddit = async (searchParams) => {
    console.log('Searching with params:', searchParams);
    setLoading(true);
    setError(null);
    setSearched(true);
    
    try {
      // Convert search params to query string
      const queryParams = new URLSearchParams({
        keyword: searchParams.keyword,
        subreddits: searchParams.subreddits,
        timeframe: searchParams.timeframe,
        sort: searchParams.sort,
        includeNsfw: searchParams.includeNsfw
      }).toString();

      const response = await fetch(`http://localhost:9000/api/search?${queryParams}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch data from Reddit');
      }
      
      const data = await response.json();
      setPosts(data.posts);
      setSearchMetadata(data.metadata);
    } catch (err) {
      setError(err.message);
      setPosts([]);
      setSearchMetadata(null);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="app">
      <header className="app-header">
        <div className="container">
          <h1>Reddit Social Listening</h1>
          <p>Monitor Reddit posts for keywords of interest</p>
        </div>
      </header>
      
      <main className="container">
        <SearchBar onSearch={searchReddit} />
        
        {loading && (
          <div className="loading">
            <p>Searching Reddit...</p>
          </div>
        )}
        
        {error && (
          <div className="error">
            <p>{error}</p>
          </div>
        )}
        
        {searchMetadata && !loading && !error && (
          <div className="search-metadata">
            <p>
              Found {searchMetadata.total} results 
              {searchMetadata.timeframe !== 'all' && ` from the past ${searchMetadata.timeframe}`}
              {searchMetadata.sort !== 'relevance' && `, sorted by ${searchMetadata.sort}`}
            </p>
            <p className="search-query">Search query: {searchMetadata.query}</p>
          </div>
        )}
        
        {!loading && !error && searched && posts.length === 0 && (
          <div className="no-results">
            <p>No posts found. Try different keywords or search options.</p>
          </div>
        )}
        
        {!loading && !error && posts.length > 0 && (
          <PostList posts={posts} />
        )}
      </main>
      
      <footer className="app-footer">
        <div className="container">
          <p>&copy; {new Date().getFullYear()} Reddit Social Listening App</p>
        </div>
      </footer>
    </div>
  );
}

export default App; 