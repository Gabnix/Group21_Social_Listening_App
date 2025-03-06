import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

interface SentimentData {
  sentiment: 'positive' | 'negative' | 'neutral';
  count: number;
}

interface SentimentChartProps {
  data: SentimentData[];
}

const COLORS = {
  positive: '#4CAF50',
  neutral: '#FFC107',
  negative: '#F44336'
};

export default function SentimentChart({ data }: SentimentChartProps) {
  const total = data.reduce((acc, item) => acc + item.count, 0);

  return (
    <div className="w-full h-[400px] p-4">
      <h2 className="text-xl font-semibold mb-4 text-center">Sentiment Distribution</h2>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            outerRadius={120}
            fill="#8884d8"
            dataKey="count"
            nameKey="sentiment"
          >
            {data.map((entry, index) => (
              <Cell 
                key={`cell-${index}`} 
                fill={COLORS[entry.sentiment]} 
              />
            ))}
          </Pie>
          <Tooltip 
            formatter={(value: number) => [`${value} posts (${((value/total) * 100).toFixed(1)}%)`, 'Count']}
          />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
} 