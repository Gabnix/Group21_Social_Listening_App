const request = require("supertest");
const { app } = require("../server");

describe("Rate Limiting", () => {
  test("allows requests within rate limit", async () => {
    const response = await request(app)
      .get("/api/search")
      .query({ keyword: "test" });

    expect(response.status).not.toBe(429);
  });

  test("blocks requests when rate limit exceeded", async () => {
    // Make many requests to exceed rate limit
    const requests = Array(601)
      .fill()
      .map(() => request(app).get("/api/search").query({ keyword: "test" }));

    const responses = await Promise.all(requests);
    const rateLimitedResponses = responses.filter((r) => r.status === 429);

    expect(rateLimitedResponses.length).toBeGreaterThan(0);
    expect(rateLimitedResponses[0].body).toHaveProperty("error");
    expect(rateLimitedResponses[0].body).toHaveProperty("retryAfter");
  });

  test("resets rate limit after window", async () => {
    // Force window reset by manipulating time
    jest.useFakeTimers();
    jest.setSystemTime(Date.now() + 600000); // Advance time by 10 minutes

    const response = await request(app)
      .get("/api/search")
      .query({ keyword: "test" });

    expect(response.status).not.toBe(429);

    jest.useRealTimers();
  });
});
