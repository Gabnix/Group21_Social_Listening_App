const nock = require('nock');
const request = require('supertest');
const { app, getRedditAccessToken } = require('../server');

// Configure axios for nock
const axios = require('axios');
axios.defaults.adapter = 'http';

describe('Reddit API Integration', () => {
  beforeEach(() => {
    // Clear all nock interceptors
    nock.cleanAll();
  });

  afterAll(() => {
    nock.restore();
  });

  describe('getRedditAccessToken', () => {
    test('successfully retrieves access token', async () => {
      const mockToken = 'mock-access-token';
      
      nock('https://www.reddit.com')
        .post('/api/v1/access_token')
        .reply(200, {
          access_token: mockToken,
          token_type: 'bearer',
          expires_in: 3600,
        });

      const token = await getRedditAccessToken();
      expect(token).toBe(mockToken);
    });

    test('handles API error', async () => {
      nock('https://www.reddit.com')
        .post('/api/v1/access_token')
        .reply(401, {
          error: 'invalid_client',
          error_description: 'Invalid client credentials',
        });

      await expect(getRedditAccessToken()).rejects.toThrow();
    });
  });

  describe('Search API Endpoint', () => {
    const mockAccessToken = 'mock-access-token';
    
    beforeEach(() => {
      // Mock the token retrieval for all search tests
      nock('https://www.reddit.com')
        .post('/api/v1/access_token')
        .reply(200, { access_token: mockAccessToken });
    });

    test('successfully searches Reddit posts', async () => {
      const mockSearchResults = {
        data: {
          children: [
            {
              data: {
                id: '123',
                title: 'Test Post',
                subreddit_name_prefixed: 'r/test',
                author: 'testuser',
                permalink: '/r/test/comments/123/test_post',
                created_utc: 1615480800,
                score: 100,
                num_comments: 10,
                selftext: 'Test content',
                thumbnail: 'https://example.com/thumb.jpg',
                upvote_ratio: 0.95,
                over_18: false,
                link_flair_text: 'Test',
                domain: 'self.test',
                is_video: false
              }
            }
          ]
        }
      };

      nock('https://oauth.reddit.com')
        .get('/search')
        .query(true)
        .reply(200, mockSearchResults);

      const response = await request(app)
        .get('/api/search')
        .query({
          keyword: 'test',
          subreddits: 'test',
          timeframe: 'all',
          sort: 'relevance'
        });

      expect(response.status).toBe(200);
      expect(response.body.posts).toHaveLength(1);
      expect(response.body.posts[0]).toHaveProperty('title', 'Test Post');
    });

    test('handles missing keyword parameter', async () => {
      const response = await request(app)
        .get('/api/search')
        .query({
          subreddits: 'test',
          timeframe: 'all',
          sort: 'relevance'
        });

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('error', 'Keyword is required');
    });

    test('handles Reddit API errors', async () => {
      nock('https://oauth.reddit.com')
        .get('/search')
        .query(true)
        .reply(500, { error: 'Internal Server Error' });

      const response = await request(app)
        .get('/api/search')
        .query({ keyword: 'test' });

      expect(response.status).toBe(500);
      expect(response.body).toHaveProperty('error', 'Failed to search Reddit');
    });

    test('handles rate limiting', async () => {
      nock('https://oauth.reddit.com')
        .get('/search')
        .query(true)
        .reply(429, { error: 'Too Many Requests' });

      const response = await request(app)
        .get('/api/search')
        .query({ keyword: 'test' });

      expect(response.status).toBe(500);
      expect(response.body).toHaveProperty('error', 'Failed to search Reddit');
    });
  });
}); 