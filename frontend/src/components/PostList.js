import React from 'react';
import './PostList.css';
import PostItem from './PostItem';

const PostList = ({ posts }) => {
  return (
    <div className="post-list">
      <h2 className="post-list-title">Found {posts.length} posts</h2>
      {posts.map(post => (
        <PostItem key={post.id} post={post} />
      ))}
    </div>
  );
};

export default PostList; 