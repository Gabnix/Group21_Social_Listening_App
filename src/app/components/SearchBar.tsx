import React, { useState } from 'react';

interface SearchBarProps {
  onSearch: (keywords: string[]) => void;
}

export default function SearchBar({ onSearch }: SearchBarProps) {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const keywords = input
      .split(',')
      .map(keyword => keyword.trim())
      .filter(keyword => keyword.length > 0);
    onSearch(keywords);
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto p-4">
      <div className="flex flex-col gap-2">
        <label htmlFor="keywords" className="text-sm font-medium text-gray-700">
          Enter keywords (comma-separated)
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            id="keywords"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="e.g., wheat rust, fungicide resistance, crop disease"
            className="flex-1 p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <button
            type="submit"
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            Search
          </button>
        </div>
        <p className="text-xs text-gray-500">
          Add multiple keywords separated by commas for broader results
        </p>
      </div>
    </form>
  );
} 