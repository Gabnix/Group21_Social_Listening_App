module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/tests/**/*.test.js'],
  collectCoverage: true,
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov'],
  setupFilesAfterEnv: ['./tests/setup.js'],
  testTimeout: 10000, // Increased timeout for rate limit tests
  clearMocks: true,
  resetMocks: true,
  restoreMocks: true,
}; 