const axios = require('axios');
const Post = require('../models/Post');
require('dotenv').config();

// Function to get Reddit access token
const getRedditAccessToken = async () => {
  try {
    const response = await axios.post(
      'https://www.reddit.com/api/v1/access_token',
      `grant_type=client_credentials`,
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'Authorization': `Basic ${Buffer.from(
            `${process.env.REDDIT_CLIENT_ID}:${process.env.REDDIT_CLIENT_SECRET}`
          ).toString('base64')}`
        }
      }
    );
    
    return response.data.access_token;
  } catch (error) {
    console.error('Reddit authentication error:', error);
    throw error;
  }
};

const fetchRedditPosts = async (subreddits, keywords) => {
  try {
    // Get access token
    const accessToken = await getRedditAccessToken();
    const results = [];
    
    for (const subreddit of subreddits) {
      const response = await axios.get(
        `https://oauth.reddit.com/r/${subreddit}/search.json`,
        {
          params: {
            q: keywords.join(' OR '),
            sort: 'new',
            limit: 100
          },
          headers: {
            'Authorization': `Bearer ${accessToken}`,
            'User-Agent': 'CropDiseaseTrends/1.0.0'
          }
        }
      );

      const posts = response.data.data.children;
      const savedPosts = [];
      
      for (const postData of posts) {
        const post = postData.data;
        
        // Check if post already exists
        const existingPost = await Post.findOne({ 
          platform: 'reddit', 
          postId: post.id 
        });

        if (!existingPost) {
          // Extract location if available (this is simplified)
          let coordinates = [0, 0];  // Default
          
          // Simplified location extraction logic
          const locationRegex = /\b(Australia|Perth|Sydney|Melbourne)\b/i;
          const locationMatch = post.selftext.match(locationRegex) || 
                               post.title.match(locationRegex);
          
          if (locationMatch) {
            // In a real app, you would use a geocoding service
            if (locationMatch[1].toLowerCase() === 'perth') {
              coordinates = [115.8613, -31.9523];
            } else if (locationMatch[1].toLowerCase() === 'sydney') {
              coordinates = [151.2093, -33.8688];
            }
            // Add more location mappings as needed
          }
          
          const newPost = await Post.create({
            platform: 'reddit',
            postId: post.id,
            content: post.selftext || post.title,
            author: post.author,
            timestamp: new Date(post.created_utc * 1000),
            location: {
              type: 'Point',
              coordinates: coordinates
            },
            keywords: keywords.filter(keyword => 
              (post.selftext + post.title).toLowerCase().includes(keyword.toLowerCase())
            ),
            metadata: post
          });
          
          savedPosts.push(newPost);
        }
      }
      
      results.push({ 
        subreddit, 
        count: savedPosts.length, 
        posts: savedPosts 
      });
    }

    return { success: true, results };
  } catch (error) {
    console.error('Reddit API error:', error);
    return { success: false, error: error.message };
  }
};

module.exports = { fetchRedditPosts };