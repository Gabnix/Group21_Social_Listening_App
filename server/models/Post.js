// server/models/Post.js
const mongoose = require('mongoose');

const PostSchema = new mongoose.Schema({
  source: {
    type: String,
    required: true,
    enum: ['reddit', 'twitter', 'facebook', 'other'],
    index: true
  },
  source_id: {
    type: String,
    required: true
  },
  title: {
    type: String,
    index: 'text'
  },
  content: {
    type: String,
    index: 'text'
  },
  author: {
    type: String,
    index: true
  },
  url: {
    type: String
  },
  created: {
    type: Date,
    default: Date.now,
    index: true
  },
  subreddit: {
    type: String,
    index: true
  },
  metadata: {
    upvotes: Number,
    downvotes: Number,
    score: Number,
    comments: Number,
    awards: Number,
    nsfw: Boolean
  },
  sentiment: {
    type: Number
  },
  keywords: [{
    type: String
  }],
  collected_at: {
    type: Date,
    default: Date.now
  }
}, {
  timestamps: true
});

// Compound index to ensure uniqueness
PostSchema.index({ source: 1, source_id: 1 }, { unique: true });

module.exports = mongoose.model('Post', PostSchema);