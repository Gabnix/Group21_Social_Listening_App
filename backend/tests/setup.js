// Mock environment variables
process.env.REDDIT_CLIENT_ID = 'test_client_id';
process.env.REDDIT_CLIENT_SECRET = 'test_client_secret';

// Reset rate limiting between tests
beforeEach(() => {
  global.requestCount = 0;
  global.lastReset = Date.now();
});

// Clean up after tests
afterEach(() => {
  jest.useRealTimers();
}); 