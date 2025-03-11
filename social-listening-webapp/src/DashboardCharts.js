// DashboardCharts.js
import React from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell 
} from 'recharts';

// Line chart data
const trendData = [
  { name: '0', western: 100, rest: 150 },
  { name: '1', western: 200, rest: 220 },
  { name: '2', western: 250, rest: 340 },
  { name: '3', western: 180, rest: 100 },
  { name: '4', western: 50, rest: 30 },
  { name: '5', western: 100, rest: 80 },
  { name: '6', western: 250, rest: 150 },
  { name: '7', western: 180, rest: 100 },
  { name: '8', western: 150, rest: 110 },
  { name: '9', western: 170, rest: 130 },
];

// Pie chart data
const diseaseData = [
  { name: 'Crop Disease #1', value: 2500, color: '#000000' },
  { name: 'Crop Disease #2', value: 2000, color: '#808080' },
  { name: 'Crop Disease #3', value: 2000, color: '#E0E0E0' },
];

// Trend Line Chart Component
export const TrendLineChart = () => {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart
        data={trendData}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" />
        <YAxis hide={true} />
        <Tooltip />
        <Line 
          type="monotone" 
          dataKey="western" 
          stroke="#4CAF50" 
          strokeWidth={2}
          dot={{ r: 4 }}
          activeDot={{ r: 8 }}
        />
        <Line 
          type="monotone" 
          dataKey="rest" 
          stroke="#9C27B0" 
          strokeWidth={2}
          dot={{ r: 4 }}
          activeDot={{ r: 8 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

// Disease Pie Chart Component
export const DiseasePieChart = () => {
  return (
    <ResponsiveContainer width="100%" height={200}>
      <PieChart>
        <Pie
          data={diseaseData}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={80}
          paddingAngle={0}
          dataKey="value"
        >
          {diseaseData.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.color} />
          ))}
        </Pie>
      </PieChart>
    </ResponsiveContainer>
  );
};