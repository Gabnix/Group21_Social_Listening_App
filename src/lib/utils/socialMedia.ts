import axios from 'axios';

// Reddit API configuration
const REDDIT_CLIENT_ID = process.env.REDDIT_CLIENT_ID;
const REDDIT_CLIENT_SECRET = process.env.REDDIT_CLIENT_SECRET;
const REDDIT_USER_AGENT = 'web:crop-disease-monitor:v1.0.0';

// Facebook Graph API configuration
const FB_ACCESS_TOKEN = process.env.FB_ACCESS_TOKEN;

interface RedditPost {
  title: string;
  selftext: string;
  author: string;
  created_utc: number;
  score: number;
  num_comments: number;
  permalink: string;
  subreddit: string;
}

interface FacebookPost {
  message: string;
  from: {
    name: string;
    id: string;
  };
  created_time: string;
  likes: {
    summary: {
      total_count: number;
    };
  };
  comments: {
    summary: {
      total_count: number;
    };
  };
  shares?: {
    count: number;
  };
}

async function getRedditPosts(keywords: string[]) {
  try {
    console.log('Fetching Reddit posts for keywords:', keywords);
    
    // Get Reddit OAuth token
    const tokenResponse = await axios.post(
      'https://www.reddit.com/api/v1/access_token',
      `grant_type=client_credentials`,
      {
        auth: {
          username: REDDIT_CLIENT_ID!,
          password: REDDIT_CLIENT_SECRET!,
        },
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'User-Agent': REDDIT_USER_AGENT,
        },
      }
    );

    const token = tokenResponse.data.access_token;
    console.log('Successfully obtained Reddit token');

    // Search posts for each keyword
    const allPosts = await Promise.all(
      keywords.map(async (keyword) => {
        console.log(`Searching Reddit for keyword: ${keyword}`);
        const searchResponse = await axios.get(
          `https://oauth.reddit.com/r/farming+agriculture+crops+plantdisease/search?q=${encodeURIComponent(
            keyword
          )}&sort=new&limit=10&restrict_sr=true`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
              'User-Agent': REDDIT_USER_AGENT,
            },
          }
        );

        if (!searchResponse.data.data?.children) {
          console.log(`No results found for keyword: ${keyword}`);
          return [];
        }

        return searchResponse.data.data.children.map((child: { data: RedditPost }) => ({
          content: `${child.data.title} ${child.data.selftext}`.trim(),
          platform: 'reddit',
          author: child.data.author,
          timestamp: new Date(child.data.created_utc * 1000),
          engagement: {
            likes: child.data.score,
            replies: child.data.num_comments,
            shares: 0,
          },
          metadata: {
            language: 'en',
            postUrl: `https://reddit.com${child.data.permalink}`,
            location: child.data.subreddit,
          },
        }));
      })
    );

    const flattenedPosts = allPosts.flat();
    console.log(`Found ${flattenedPosts.length} total Reddit posts`);
    return flattenedPosts;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error('Reddit API Error:', {
        status: error.response?.status,
        message: error.response?.data,
        url: error.config?.url
      });
    } else {
      console.error('Error fetching Reddit posts:', error);
    }
    return [];
  }
}

async function getFacebookPosts(keywords: string[]) {
  try {
    const allPosts = await Promise.all(
      keywords.map(async (keyword) => {
        const searchResponse = await axios.get(
          `https://graph.facebook.com/v19.0/search?q=${encodeURIComponent(
            keyword
          )}&type=post&fields=message,from,created_time,likes.summary(true),comments.summary(true),shares&access_token=${FB_ACCESS_TOKEN}`
        );

        return searchResponse.data.data.map((post: FacebookPost) => ({
          content: post.message,
          platform: 'facebook',
          author: post.from.name,
          timestamp: new Date(post.created_time),
          engagement: {
            likes: post.likes?.summary?.total_count || 0,
            replies: post.comments?.summary?.total_count || 0,
            shares: post.shares?.count || 0,
          },
          metadata: {
            language: 'en', // Facebook usually returns posts in the user's language
            postUrl: `https://facebook.com/${post.from.id}`,
          },
        }));
      })
    );

    return allPosts.flat();
  } catch (error) {
    console.error('Error fetching Facebook posts:', error);
    return [];
  }
}

export async function fetchSocialMediaPosts(keywords: string[]) {
  // For now, we'll only fetch Reddit posts while we set up Facebook properly
  const redditPosts = await getRedditPosts(keywords);
  return redditPosts;
} 