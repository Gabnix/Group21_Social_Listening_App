import { NextResponse } from 'next/server';
import dbConnect from '@/lib/db/connection';
import Post from '@/lib/db/models/Post';
import { analyzeSentiment } from '@/lib/utils/sentiment';
import { fetchSocialMediaPosts } from '@/lib/utils/socialMedia';

export async function POST(request: Request) {
  try {
    const { keywords } = await request.json();
    
    if (!Array.isArray(keywords) || keywords.length === 0) {
      return NextResponse.json(
        { error: 'Keywords must be a non-empty array' },
        { status: 400 }
      );
    }
    
    // Connect to database
    await dbConnect();
    
    // Fetch social media posts from Reddit and Facebook
    const posts = await fetchSocialMediaPosts(keywords);
    
    if (posts.length === 0) {
      return NextResponse.json({
        posts: [],
        sentimentDistribution: []
      });
    }
    
    // Analyze sentiment and store in database
    const analyzedPosts = await Promise.all(
      posts.map(async (post) => {
        const sentiment = analyzeSentiment(post.content);
        
        const newPost = new Post({
          ...post,
          keywords,
          sentiment
        });
        
        await newPost.save();
        return newPost;
      })
    );
    
    // Calculate sentiment distribution
    const sentimentCounts = analyzedPosts.reduce((acc, post) => {
      const label = post.sentiment.label;
      acc[label] = (acc[label] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    const sentimentDistribution = Object.entries(sentimentCounts).map(
      ([sentiment, count]) => ({
        sentiment,
        count
      })
    );
    
    // Sort posts by engagement
    const sortedPosts = analyzedPosts.sort(
      (a, b) => 
        (b.engagement.likes + b.engagement.shares + b.engagement.replies) -
        (a.engagement.likes + a.engagement.shares + a.engagement.replies)
    );
    
    return NextResponse.json({
      posts: sortedPosts,
      sentimentDistribution
    });
    
  } catch (error) {
    console.error('Error processing request:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
} 