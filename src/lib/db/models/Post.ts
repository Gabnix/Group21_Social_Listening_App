import mongoose, { Schema, Document } from 'mongoose';

export interface IPost extends Document {
  content: string;
  platform: string;
  author: string;
  timestamp: Date;
  engagement: {
    likes: number;
    replies: number;
    shares: number;
  };
  sentiment: {
    score: number;
    label: 'positive' | 'negative' | 'neutral';
    confidence: number;
  };
  keywords: string[];
  metadata: {
    location?: string;
    language: string;
    postUrl: string;
  };
  createdAt: Date;
  updatedAt: Date;
}

const PostSchema = new Schema<IPost>(
  {
    content: { type: String, required: true },
    platform: { type: String, required: true },
    author: { type: String, required: true },
    timestamp: { type: Date, required: true },
    engagement: {
      likes: { type: Number, default: 0 },
      replies: { type: Number, default: 0 },
      shares: { type: Number, default: 0 },
    },
    sentiment: {
      score: { type: Number, required: true },
      label: { 
        type: String, 
        required: true,
        enum: ['positive', 'negative', 'neutral']
      },
      confidence: { type: Number, required: true },
    },
    keywords: [{ type: String }],
    metadata: {
      location: String,
      language: { type: String, required: true },
      postUrl: { type: String, required: true },
    },
  },
  { timestamps: true }
);

export default mongoose.models.Post || mongoose.model<IPost>('Post', PostSchema); 