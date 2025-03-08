const express = require('express');
const cors = require('cors');
const axios = require('axios');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config();

const app = express();
const PORT = 9000;

// Middleware
app.use(cors());
app.use(express.json());

// Reddit API credentials
const REDDIT_CLIENT_ID = process.env.REDDIT_CLIENT_ID;
const REDDIT_CLIENT_SECRET = process.env.REDDIT_CLIENT_SECRET;

// Get Reddit access token
async function getRedditAccessToken() {
  try {
    const response = await axios.post(
      'https://www.reddit.com/api/v1/access_token',
      'grant_type=client_credentials',
      {
        auth: {
          username: REDDIT_CLIENT_ID,
          password: REDDIT_CLIENT_SECRET
        },
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      }
    );
    return response.data.access_token;
  } catch (error) {
    console.error('Error getting Reddit access token:', error);
    throw error;
  }
}

// Helper function to build Reddit search query
function buildSearchQuery(params) {
  const {
    keyword,
    subreddits = [],
    timeframe = 'all',
    sort = 'relevance',
    includeNsfw = false
  } = params;

  // Split the input into terms while preserving quoted phrases
  const terms = [];
  let currentTerm = '';
  let inQuotes = false;

  for (let i = 0; i < keyword.length; i++) {
    const char = keyword[i];
    
    if (char === '"') {
      inQuotes = !inQuotes;
      currentTerm += char;
    } else if (char === ' ' && !inQuotes) {
      if (currentTerm) {
        terms.push(currentTerm);
        currentTerm = '';
      }
    } else {
      currentTerm += char;
    }
  }
  if (currentTerm) {
    terms.push(currentTerm);
  }

  // Process terms and build query
  let query = terms
    .filter(term => term.trim()) // Remove empty terms
    .map(term => {
      term = term.trim();
      // Keep existing quotes
      if (term.startsWith('"') && term.endsWith('"')) {
        return term;
      }
      // Handle operators
      if (term.toUpperCase() === 'AND' || term.toUpperCase() === 'OR') {
        return term.toUpperCase();
      }
      // Add quotes if term contains spaces
      if (term.includes(' ')) {
        return `"${term}"`;
      }
      return term;
    })
    .join(' '); // Join with spaces

  // Add subreddit filtering
  if (subreddits.length > 0) {
    const subredditQuery = subreddits.map(sub => `subreddit:${sub}`).join(' OR ');
    query = `${query} (${subredditQuery})`;
  }

  // Add NSFW filtering
  if (!includeNsfw) {
    query = `${query} NOT nsfw:1`;
  }

  console.log('Search query:', query); // Debug log

  return {
    q: query,
    t: timeframe,
    sort: sort,
    limit: 100,
    restrict_sr: false
  };
}

// Search Reddit posts by keyword
app.get('/api/search', async (req, res) => {
  const { 
    keyword,
    subreddits,
    timeframe,
    sort,
    includeNsfw
  } = req.query;
  
  if (!keyword) {
    return res.status(400).json({ error: 'Keyword is required' });
  }
  
  try {
    const accessToken = await getRedditAccessToken();
    const searchParams = buildSearchQuery({
      keyword,
      subreddits: subreddits ? subreddits.split(',') : [],
      timeframe,
      sort,
      includeNsfw: includeNsfw === 'true'
    });
    
    const queryString = new URLSearchParams(searchParams).toString();
    const response = await axios.get(
      `https://oauth.reddit.com/search?${queryString}`,
      {
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'User-Agent': 'social-listening-app/1.0.0'
        }
      }
    );
    
    const posts = response.data.data.children.map(child => {
      const post = child.data;
      return {
        id: post.id,
        title: post.title,
        subreddit: post.subreddit_name_prefixed,
        author: post.author,
        url: `https://reddit.com${post.permalink}`,
        created: post.created_utc,
        score: post.score,
        numComments: post.num_comments,
        selftext: post.selftext,
        thumbnail: post.thumbnail !== 'self' && post.thumbnail !== 'default' ? post.thumbnail : null,
        // Additional fields
        upvoteRatio: post.upvote_ratio,
        isNsfw: post.over_18,
        flair: post.link_flair_text,
        domain: post.domain,
        isVideo: post.is_video
      };
    });

    // Sort results by relevance score (combination of score and comments)
    posts.sort((a, b) => {
      const scoreA = a.score + (a.numComments * 2); // Weight comments more
      const scoreB = b.score + (b.numComments * 2);
      return scoreB - scoreA;
    });
    
    res.json({ 
      posts,
      metadata: {
        total: posts.length,
        timeframe,
        sort,
        query: searchParams.q
      }
    });
  } catch (error) {
    console.error('Error searching Reddit:', error);
    res.status(500).json({ error: 'Failed to search Reddit' });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
}); 