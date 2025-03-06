import { WordTokenizer, SentimentAnalyzer, PorterStemmer } from 'natural';

interface SentimentResult {
  score: number;
  label: 'positive' | 'negative' | 'neutral';
  confidence: number;
}

export function analyzeSentiment(text: string): SentimentResult {
  const tokenizer = new WordTokenizer();
  const analyzer = new SentimentAnalyzer('English', PorterStemmer, 'afinn');
  
  // Tokenize and analyze
  const tokens = tokenizer.tokenize(text);
  if (!tokens) return { score: 0, label: 'neutral', confidence: 0 };
  
  const score = analyzer.getSentiment(tokens);
  
  // Normalize score to range [-1, 1]
  const normalizedScore = Math.max(-1, Math.min(1, score / 5));
  
  // Calculate confidence based on the absolute value of the score
  const confidence = Math.abs(normalizedScore);
  
  // Determine sentiment label
  let label: 'positive' | 'negative' | 'neutral';
  if (normalizedScore > 0.1) {
    label = 'positive';
  } else if (normalizedScore < -0.1) {
    label = 'negative';
  } else {
    label = 'neutral';
  }
  
  return {
    score: normalizedScore,
    label,
    confidence
  };
} 