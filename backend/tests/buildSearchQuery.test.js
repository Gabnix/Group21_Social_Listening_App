const { buildSearchQuery } = require('../server');

describe('buildSearchQuery', () => {
  test('handles simple keyword search', () => {
    const params = {
      keyword: 'pesticide',
    };
    const result = buildSearchQuery(params);
    expect(result.q).toBe('pesticide NOT nsfw:1');
  });

  test('handles AND operator correctly', () => {
    const params = {
      keyword: 'pesticide AND fungicide',
    };
    const result = buildSearchQuery(params);
    expect(result.q).toBe('pesticide AND fungicide NOT nsfw:1');
  });

  test('handles quoted phrases correctly', () => {
    const params = {
      keyword: '"crop diseases" AND pesticide',
    };
    const result = buildSearchQuery(params);
    expect(result.q).toBe('"crop diseases" AND pesticide NOT nsfw:1');
  });

  test('handles subreddit filtering', () => {
    const params = {
      keyword: 'pesticide',
      subreddits: ['farming', 'agriculture'],
    };
    const result = buildSearchQuery(params);
    expect(result.q).toBe('pesticide (subreddit:farming OR subreddit:agriculture) NOT nsfw:1');
  });

  test('handles timeframe and sort parameters', () => {
    const params = {
      keyword: 'pesticide',
      timeframe: 'month',
      sort: 'top',
    };
    const result = buildSearchQuery(params);
    expect(result.q).toBe('pesticide NOT nsfw:1');
    expect(result.t).toBe('month');
    expect(result.sort).toBe('top');
  });

  test('handles complex queries with multiple operators', () => {
    const params = {
      keyword: '"organic farming" AND (pesticide OR herbicide)',
    };
    const result = buildSearchQuery(params);
    expect(result.q).toBe('"organic farming" AND (pesticide OR herbicide) NOT nsfw:1');
  });

  test('handles NSFW content inclusion', () => {
    const params = {
      keyword: 'pesticide',
      includeNsfw: true,
    };
    const result = buildSearchQuery(params);
    expect(result.q).toBe('pesticide');
  });

  test('handles empty subreddits array', () => {
    const params = {
      keyword: 'pesticide',
      subreddits: [],
    };
    const result = buildSearchQuery(params);
    expect(result.q).toBe('pesticide NOT nsfw:1');
  });

  test('handles whitespace in keywords', () => {
    const params = {
      keyword: '  pesticide  AND   fungicide  ',
    };
    const result = buildSearchQuery(params);
    expect(result.q).toBe('pesticide AND fungicide NOT nsfw:1');
  });
}); 