const mongoose = require('mongoose');

const PostSchema = new mongoose.Schema({
  platform: {
    type: String,
    required: true,
    enum: ['facebook', 'reddit', 'twitter', 'government']
  },
  postId: {
    type: String,
    required: true
  },
  content: {
    type: String,
    required: true
  },
  author: String,
  timestamp: {
    type: Date,
    default: Date.now
  },
  location: {
    type: {
      type: String,
      enum: ['Point'],
      default: 'Point'
    },
    coordinates: {
      type: [Number], // [longitude, latitude]
      required: true
    }
  },
  keywords: [String],
  sentiment: {
    score: Number, // -1 to 1 scale
    label: {
      type: String,
      enum: ['negative', 'neutral', 'positive']
    }
  },
  metadata: mongoose.Schema.Types.Mixed
}, { timestamps: true });

// Create compound index for platform and postId
PostSchema.index({ platform: 1, postId: 1 }, { unique: true });
// Create geospatial index for location queries
PostSchema.index({ location: '2dsphere' });

module.exports = mongoose.model('Post', PostSchema);