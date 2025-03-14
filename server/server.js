const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const path = require('path');
const connectDB = require('./config/db');
const apiRoutes = require('./routes/api');
const users = require('./routes/users');
require('dotenv').config();

// Import routes
const posts = require('./routes/posts');

// Import scheduled collection
const { setupScheduledCollection } = require('./services/scheduledCollection');

// Initialize Express
const app = express();

// Express middleware
app.use(cors());
app.use(express.json());

// Connect to MongoDB
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/social-listening', {
  useNewUrlParser: true,
  useUnifiedTopology: true
})
  .then(() => console.log('MongoDB connected'))
  .catch(err => {
    console.error('MongoDB connection error:', err);
    process.exit(1);
  });

// Define routes
app.use('/api/posts', posts);

// Set up a simple health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', time: new Date().toISOString() });
});

// Serve static assets in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../client/build')));
  
  app.get('*', (req, res) => {
    res.sendFile(path.resolve(__dirname, '../client/build', 'index.html'));
  });
}

// Start data collection scheduler if enabled
if (process.env.ENABLE_SCHEDULER === 'true') {
  setupScheduledCollection(process.env.COLLECTION_SCHEDULE || '0 */2 * * *');
}

// Define port
const PORT = process.env.PORT || 5000;

// Start server
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));