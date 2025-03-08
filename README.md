# Reddit Social Listening App

A web application for monitoring Reddit posts based on keywords. This app allows users to search for specific keywords and displays relevant posts from Reddit.

## Features

- Search for Reddit posts by keyword
- Display post details including title, author, subreddit, score, and comments
- Modern and responsive UI
- Real-time data from Reddit API

## Tech Stack

- **Frontend**: React.js
- **Backend**: Node.js with Express
- **API**: Reddit API

## Setup Instructions

### Prerequisites

- Node.js (v14 or later)
- npm (v6 or later)
- Reddit API credentials (Client ID and Client Secret)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd social-listening-web-scrapper
   ```

2. Install backend dependencies:
   ```
   cd backend
   npm install
   ```

3. Configure environment variables:
   Create a `.env` file in the backend directory and add your Reddit API credentials:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   PORT=5000
   ```

4. Install frontend dependencies:
   ```
   cd ../frontend
   npm install
   ```

### Running the Application

1. Start the backend server:
   ```
   cd backend
   npm start
   === OR ===
   cd backend && node server.js
   ```

2. Start the frontend development server:
   ```
   cd frontend
   npm start
   ```

3. Open your browser and navigate to `http://localhost:3000`

## Getting Reddit API Credentials

1. Go to [Reddit Developer Portal](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Fill in the required information:
   - Name: Social Listening App
   - App type: Script
   - Description: An app to search Reddit posts
   - About URL: (Leave blank for now)
   - Redirect URI: http://localhost:3000
4. Click "Create app"
5. You'll receive a Client ID (below the app name) and Client Secret

## License

MIT 