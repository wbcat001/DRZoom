import React, { useState, useEffect } from 'react';
import { useSelection, useData } from '../../store/useAppStore';
import './SearchBar.css';

const SearchBar: React.FC = () => {
  const { selection, setSearchQuery, setSearchResults } = useSelection();
  const { data } = useData();
  const [inputValue, setInputValue] = useState<string>('');
  const [matchCount, setMatchCount] = useState<number>(0);

  // Handle search input change
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const query = e.target.value.toLowerCase();
    setInputValue(e.target.value);

    if (query.trim() === '') {
      // Clear search results
      setSearchResults([]);
      setSearchQuery('');
      setMatchCount(0);
      return;
    }

    // Search for matching labels (words)
    const matchingPointIds = data.points
      .filter(point => point.l.toLowerCase().includes(query))
      .map(point => point.i);

    setSearchQuery(query);
    setSearchResults(matchingPointIds);
    setMatchCount(matchingPointIds.length);
  };

  // Clear search
  const handleClearSearch = () => {
    setInputValue('');
    setSearchResults([]);
    setSearchQuery('');
    setMatchCount(0);
  };

  return (
    <div className="search-bar">
      <div className="search-input-wrapper">
        <input
          type="text"
          className="search-input"
          placeholder="Search labels (e.g., 'apple', 'word')..."
          value={inputValue}
          onChange={handleSearchChange}
        />
        {inputValue && (
          <button className="search-clear-btn" onClick={handleClearSearch} title="Clear search">
            ✕
          </button>
        )}
      </div>
      {inputValue && (
        <div className="search-results-info">
          Found <span className="match-count">{matchCount}</span> match{matchCount !== 1 ? 'es' : ''}
        </div>
      )}
    </div>
  );
};

export default SearchBar;
