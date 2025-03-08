import React from 'react';
import './PostItem.css';

const PostItem = ({ post }) => {
  const formatDate = (timestamp) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };

  const truncateText = (text, maxLength = 200) => {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.slice(0, maxLength) + '...';
  };

  return (
    <div className="post-item">
      {post.thumbnail && (
        <div className="post-thumbnail">
          <img src={post.thumbnail} alt="Post thumbnail" />
        </div>
      )}
      
      <div className="post-content">
        <h3 className="post-title">
          <a href={post.url} target="_blank" rel="noopener noreferrer">
            {post.title}
          </a>
        </h3>
        
        <div className="post-meta">
          <span className="post-subreddit">{post.subreddit}</span>
          <span className="post-author">Posted by u/{post.author}</span>
          <span className="post-date">{formatDate(post.created)}</span>
        </div>
        
        {post.selftext && (
          <p className="post-text">{truncateText(post.selftext)}</p>
        )}
        
        <div className="post-stats">
          <span className="post-score">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 19V5M5 12l7-7 7 7"/>
            </svg>
            {post.score} points
          </span>
          
          <span className="post-comments">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            {post.numComments} comments
          </span>
          
          <a href={post.url} target="_blank" rel="noopener noreferrer" className="post-link">
            View on Reddit
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
              <polyline points="15 3 21 3 21 9"/>
              <line x1="10" y1="14" x2="21" y2="3"/>
            </svg>
          </a>
        </div>
      </div>
    </div>
  );
};

export default PostItem; 