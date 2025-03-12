// server/services/dataCollector.js
const mongoose = require('mongoose');
const axios = require('axios');
const Post = require('../models/Post');
require('dotenv').config();

// Reddit API configuration
const REDDIT_SUBREDDITS = ['technology', 'news', 'programming']; // Add subreddits relevant to your project
const POSTS_LIMIT = 25;

/**
 * Fetch data from Reddit for a specific subreddit
 * @param {string} subreddit - The subreddit to fetch data from
 */
async function fetchRedditData(subreddit) {
  try {
    console.log(`Fetching data from r/${subreddit}...`);
    
    const userAgent = 'SocialListeningApp/1.0.0';
    const response = await axios.get(`https://www.reddit.com/r/${subreddit}/hot.json?limit=${POSTS_LIMIT}`, {
      headers: {
        'User-Agent': userAgent
      }
    });
    
    if (!response.data || !response.data.data || !response.data.data.children) {
      console.log(`No data found for r/${subreddit}`);
      return 0;
    }
    
    const posts = response.data.data.children;
    let savedCount = 0;
    
    for (const post of posts) {
      // Skip ads and pinned posts
      if (post.data.promoted || post.data.pinned) continue;
      
      // Check if post already exists to avoid duplicates
      const existingPost = await Post.findOne({ 
        source_id: post.data.id,
        source: 'reddit' 
      });
      
      if (existingPost) continue;
      
      const redditPost = new Post({
        source: 'reddit',
        source_id: post.data.id,
        title: post.data.title,
        content: post.data.selftext || '',
        author: post.data.author,
        url: `https://reddit.com${post.data.permalink}`,
        created: new Date(post.data.created_utc * 1000),
        subreddit: post.data.subreddit,
        metadata: {
          upvotes: post.data.ups,
          downvotes: post.data.downs,
          score: post.data.score,
          comments: post.data.num_comments,
          awards: post.data.total_awards_received,
          nsfw: post.data.over_18
        }
      });
      
      await redditPost.save();
      savedCount++;
    }
    
    console.log(`Successfully saved ${savedCount} new posts from r/${subreddit}`);
    return savedCount;
  } catch (error) {
    console.error(`Error collecting data from r/${subreddit}:`, error.message);
    return 0;
  }
}

/**
 * Main function to run the data collection process
 */
async function collectData() {
  console.log('Starting Reddit data collection process...');
  let totalSaved = 0;
  
  try {
    // Create MongoDB connection if not already connected
    if (mongoose.connection.readyState !== 1) {
      await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/social-listening', {
        useNewUrlParser: true,
        useUnifiedTopology: true
      });
      console.log('MongoDB connected');
    }
    
    // Fetch data from each subreddit
    for (const subreddit of REDDIT_SUBREDDITS) {
      const savedCount = await fetchRedditData(subreddit);
      totalSaved += savedCount;
    }
    
    console.log(`Data collection process completed. Total new posts saved: ${totalSaved}`);
  } catch (error) {
    console.error('Error in data collection process:', error);
  } 
}

// Run once when called directly
if (require.main === module) {
  collectData()
    .then(() => {
      console.log('Collection task completed, exiting...');
      setTimeout(() => process.exit(0), 2000); // Give time for operations to complete
    })
    .catch(err => {
      console.error('Fatal error in collection task:', err);
      process.exit(1);
    });
}

// Export for use in scheduled tasks
module.exports = { collectData, fetchRedditData };