import React from 'react';
import { IPost } from '@/lib/db/models/Post';

interface PostsListProps {
  posts: IPost[];
}

export default function PostsList({ posts }: PostsListProps) {
  const getSentimentColor = (label: string) => {
    switch (label) {
      case 'positive':
        return 'bg-green-100 text-green-800';
      case 'negative':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-yellow-100 text-yellow-800';
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      <h2 className="text-xl font-semibold mb-4">Most Engaging Posts</h2>
      <div className="space-y-4">
        {posts.map((post) => (
          <div
            key={post._id.toString()}
            className="border rounded-lg p-4 hover:shadow-md transition-shadow"
          >
            <div className="flex justify-between items-start mb-2">
              <div className="flex items-center gap-2">
                <span className="font-medium">{post.author}</span>
                <span className="text-gray-500 text-sm">
                  {new Date(post.timestamp).toLocaleDateString()}
                </span>
              </div>
              <span
                className={`px-2 py-1 rounded-full text-xs font-medium ${getSentimentColor(
                  post.sentiment.label
                )}`}
              >
                {post.sentiment.label}
              </span>
            </div>
            <p className="text-gray-700 mb-3">{post.content}</p>
            <div className="flex items-center gap-4 text-sm text-gray-500">
              <span>❤️ {post.engagement.likes}</span>
              <span>💬 {post.engagement.replies}</span>
              <span>🔄 {post.engagement.shares}</span>
            </div>
            <div className="mt-2">
              {post.keywords.map((keyword, index) => (
                <span
                  key={index}
                  className="inline-block bg-gray-100 rounded-full px-3 py-1 text-sm font-medium text-gray-700 mr-2 mb-2"
                >
                  #{keyword}
                </span>
              ))}
            </div>
            <a
              href={post.metadata.postUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline text-sm mt-2 inline-block"
            >
              View original post
            </a>
          </div>
        ))}
      </div>
    </div>
  );
} 