// server/routes/api/posts.js
const express = require('express');
const router = express.Router();
const Post = require('../models/Post');

/**
 * @route   GET api/posts
 * @desc    Get all posts with pagination
 * @access  Public
 */
router.get('/', async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const skip = (page - 1) * limit;
    
    const posts = await Post.find()
      .sort({ created: -1 })
      .skip(skip)
      .limit(limit);
    
    const total = await Post.countDocuments();
    
    res.json({
      posts,
      currentPage: page,
      totalPages: Math.ceil(total / limit),
      totalPosts: total
    });
  } catch (err) {
    console.error('Error fetching posts:', err.message);
    res.status(500).send('Server Error');
  }
});

/**
 * @route   GET api/posts/source/:source
 * @desc    Get posts by source (reddit, twitter, etc.)
 * @access  Public
 */
router.get('/source/:source', async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const skip = (page - 1) * limit;
    
    const posts = await Post.find({ source: req.params.source })
      .sort({ created: -1 })
      .skip(skip)
      .limit(limit);
    
    const total = await Post.countDocuments({ source: req.params.source });
    
    res.json({
      posts,
      currentPage: page,
      totalPages: Math.ceil(total / limit),
      totalPosts: total
    });
  } catch (err) {
    console.error(`Error fetching ${req.params.source} posts:`, err.message);
    res.status(500).send('Server Error');
  }
});

/**
 * @route   GET api/posts/subreddit/:subreddit
 * @desc    Get posts by subreddit
 * @access  Public
 */
router.get('/subreddit/:subreddit', async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const skip = (page - 1) * limit;
    
    const posts = await Post.find({ 
      source: 'reddit',
      subreddit: req.params.subreddit 
    })
      .sort({ created: -1 })
      .skip(skip)
      .limit(limit);
    
    const total = await Post.countDocuments({ 
      source: 'reddit',
      subreddit: req.params.subreddit 
    });
    
    res.json({
      posts,
      currentPage: page,
      totalPages: Math.ceil(total / limit),
      totalPosts: total
    });
  } catch (err) {
    console.error(`Error fetching posts from r/${req.params.subreddit}:`, err.message);
    res.status(500).send('Server Error');
  }
});

/**
 * @route   GET api/posts/search
 * @desc    Search posts by keyword
 * @access  Public
 */
router.get('/search', async (req, res) => {
  const { q } = req.query;
  
  if (!q) {
    return res.status(400).json({ msg: 'Search query is required' });
  }

  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const skip = (page - 1) * limit;
    
    // Text search on title and content
    const posts = await Post.find({
      $or: [
        { title: { $regex: q, $options: 'i' } },
        { content: { $regex: q, $options: 'i' } }
      ]
    })
      .sort({ created: -1 })
      .skip(skip)
      .limit(limit);
    
    const total = await Post.countDocuments({
      $or: [
        { title: { $regex: q, $options: 'i' } },
        { content: { $regex: q, $options: 'i' } }
      ]
    });
    
    res.json({
      posts,
      currentPage: page,
      totalPages: Math.ceil(total / limit),
      totalPosts: total,
      query: q
    });
  } catch (err) {
    console.error(`Error searching posts for "${q}":`, err.message);
    res.status(500).send('Server Error');
  }
});

/**
 * @route   GET api/posts/analytics
 * @desc    Get analytics data about posts
 * @access  Public
 */
router.get('/analytics', async (req, res) => {
  try {
    // Get counts by source
    const sourceStats = await Post.aggregate([
      { $group: { _id: '$source', count: { $sum: 1 } } },
      { $sort: { count: -1 } }
    ]);
    
    // Get counts by subreddit
    const subredditStats = await Post.aggregate([
      { $match: { source: 'reddit' } },
      { $group: { _id: '$subreddit', count: { $sum: 1 } } },
      { $sort: { count: -1 } },
      { $limit: 10 }
    ]);
    
    // Get post count by day for the last 7 days
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
    
    const postsByDay = await Post.aggregate([
      { $match: { created: { $gte: sevenDaysAgo } } },
      {
        $group: {
          _id: {
            year: { $year: '$created' },
            month: { $month: '$created' },
            day: { $dayOfMonth: '$created' }
          },
          count: { $sum: 1 }
        }
      },
      { $sort: { '_id.year': 1, '_id.month': 1, '_id.day': 1 } }
    ]);
    
    // Format the results
    const dailyData = postsByDay.map(day => ({
      date: `${day._id.year}-${day._id.month.toString().padStart(2, '0')}-${day._id.day.toString().padStart(2, '0')}`,
      count: day.count
    }));
    
    res.json({
      totalPosts: await Post.countDocuments(),
      sourceBreakdown: sourceStats,
      topSubreddits: subredditStats,
      postsOverTime: dailyData
    });
  } catch (err) {
    console.error('Error fetching analytics:', err.message);
    res.status(500).send('Server Error');
  }
});

module.exports = router;