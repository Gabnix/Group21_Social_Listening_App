'use client';

import React, { useState } from 'react';
import SearchBar from './components/SearchBar';
import SentimentChart from './components/SentimentChart';
import PostsList from './components/PostsList';
import { IPost } from '@/lib/db/models/Post';

interface SentimentData {
  sentiment: 'positive' | 'negative' | 'neutral';
  count: number;
}

export default function Home() { 
  const [loading, setLoading] = useState(false);
  const [posts, setPosts] = useState<IPost[]>([]);
  const [sentimentData, setSentimentData] = useState<SentimentData[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (keywords: string[]) => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/posts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ keywords }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }

      const data = await response.json();
      setPosts(data.posts);
      setSentimentData(data.sentimentDistribution);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <h1 className="text-3xl font-bold text-center mb-8">
          Crop Disease Social Media Monitor
        </h1>
        
        <SearchBar onSearch={handleSearch} />
        
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative my-4" role="alert">
            <span className="block sm:inline">{error}</span>
          </div>
        )}
        
        {loading ? (
          <div className="flex justify-center items-center my-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          </div>
        ) : (
          <>
            {posts.length > 0 && (
              <div className="grid md:grid-cols-2 gap-8 my-8">
                <div className="bg-white rounded-lg shadow-md">
                  <SentimentChart data={sentimentData} />
                </div>
                <div className="bg-white rounded-lg shadow-md">
                  <PostsList posts={posts} />
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </main>
  );
}
