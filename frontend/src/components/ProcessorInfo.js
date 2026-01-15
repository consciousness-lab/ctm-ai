// src/components/ProcessorInfo.js

import React, { useState } from 'react';

const PROCESSOR_DESCRIPTIONS = {
    'VisionProcessor': {
        name: 'Vision Processor',
        description: 'Analyzes images and visual content. Can detect objects, scenes, text in images, and perform visual question answering.',
        category: 'Multimodal',
        icon: '🖼️'
    },
    'LanguageProcessor': {
        name: 'Language Processor',
        description: 'Processes natural language text. Handles text understanding, generation, translation, summarization, and semantic analysis.',
        category: 'Text',
        icon: '📝'
    },
    'SearchProcessor': {
        name: 'Search Processor',
        description: 'Searches the web for information using Google Search. Finds relevant articles, facts, and current information from the internet.',
        category: 'Web',
        icon: '🔍'
    },
    'CodeProcessor': {
        name: 'Code Processor',
        description: 'Analyzes and generates code. Can understand programming languages, debug code, explain code logic, and generate code snippets.',
        category: 'Technical',
        icon: '💻'
    },
    'AudioProcessor': {
        name: 'Audio Processor',
        description: 'Processes audio files and speech. Can transcribe audio, analyze speech patterns, detect emotions, and extract audio features.',
        category: 'Multimodal',
        icon: '🎵'
    },
    'VideoProcessor': {
        name: 'Video Processor',
        description: 'Analyzes video content frame by frame. Can understand video scenes, actions, objects, and temporal relationships in video sequences.',
        category: 'Multimodal',
        icon: '🎬'
    },
    'FinanceProcessor': {
        name: 'Finance Processor',
        description: 'Retrieves real-time financial data including stock prices, market trends, cryptocurrency prices, forex rates, and financial news from Google Finance.',
        category: 'Data',
        icon: '💰'
    },
    'GeoDBProcessor': {
        name: 'GeoDB Processor',
        description: 'Accesses geographic and location data. Provides city information, population data, geographic coordinates, and location-based insights.',
        category: 'Data',
        icon: '🌍'
    },
    'TwitterProcessor': {
        name: 'Twitter Processor',
        description: 'Searches Twitter/X for tweets, trends, and social media content. Can find recent tweets, hashtags, and user activity.',
        category: 'Social',
        icon: '🐦'
    },
    'WeatherProcessor': {
        name: 'Weather Processor',
        description: 'Gets real-time weather data for any location worldwide. Provides current conditions, forecasts, and weather analytics.',
        category: 'Data',
        icon: '☁️'
    },
    'YouTubeProcessor': {
        name: 'YouTube Processor',
        description: 'Searches YouTube for videos, channels, and content. Can find video metadata, channel information, and video recommendations.',
        category: 'Social',
        icon: '📺'
    },
    'MathProcessor': {
        name: 'Math Processor',
        description: 'Solves mathematical problems using Wolfram|Alpha. Handles calculations, equations, mathematical queries, and computational problems.',
        category: 'Technical',
        icon: '🔢'
    },
    'NewsProcessor': {
        name: 'News Processor',
        description: 'Retrieves news articles. Can search for top headlines, topic-specific news, and local news from around the world.',
        category: 'Data',
        icon: '📰'
    },
    'SocialProcessor': {
        name: 'Social Processor',
        description: 'Searches for social media profile links across platforms including Facebook, Instagram, TikTok, Twitter, LinkedIn, YouTube, and more.',
        category: 'Social',
        icon: '👥'
    },
    'ExerciseProcessor': {
        name: 'Exercise Processor',
        description: 'Provides comprehensive exercise database with over 1,300 exercises. Includes workout plans, muscle targeting, equipment requirements, and exercise demonstrations.',
        category: 'Health',
        icon: '💪'
    },
    'MusicProcessor': {
        name: 'Music Processor',
        description: 'Retrieves music information, artist data, song details, and music recommendations. Can search for songs, albums, and music metadata.',
        category: 'Media',
        icon: '🎶'
    },
};

const CATEGORY_COLORS = {
    'Multimodal': '#667eea',
    'Text': '#764ba2',
    'Web': '#f093fb',
    'Technical': '#4facfe',
    'Data': '#00f2fe',
    'Social': '#fa709a',
    'Health': '#fee140',
    'Media': '#30cfd0'
};

function ProcessorInfo() {
    const [selectedCategory, setSelectedCategory] = useState(null);
    const [searchTerm, setSearchTerm] = useState('');

    const categories = [...new Set(Object.values(PROCESSOR_DESCRIPTIONS).map(p => p.category))];
    
    const filteredProcessors = Object.entries(PROCESSOR_DESCRIPTIONS).filter(([key, info]) => {
        const matchesCategory = !selectedCategory || info.category === selectedCategory;
        const matchesSearch = searchTerm === '' || 
            info.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            info.description.toLowerCase().includes(searchTerm.toLowerCase());
        return matchesCategory && matchesSearch;
    });

    return (
        <div className="processor-info-container">
            <div className="processor-info-header">
                <h2>Processor Documentation</h2>
                <p className="processor-info-subtitle">
                    Learn about each processor's capabilities and use cases
                </p>
            </div>

            <div className="processor-info-controls">
                <div className="search-box">
                    <input
                        type="text"
                        placeholder="Search processors..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="processor-search-input"
                    />
                </div>
                <div className="category-filters">
                    <button
                        className={`category-filter ${selectedCategory === null ? 'active' : ''}`}
                        onClick={() => setSelectedCategory(null)}
                    >
                        All
                    </button>
                    {categories.map(category => (
                        <button
                            key={category}
                            className={`category-filter ${selectedCategory === category ? 'active' : ''}`}
                            onClick={() => setSelectedCategory(category)}
                            style={{ 
                                borderColor: CATEGORY_COLORS[category],
                                color: selectedCategory === category ? CATEGORY_COLORS[category] : 'inherit'
                            }}
                        >
                            {category}
                        </button>
                    ))}
                </div>
            </div>

            <div className="processor-info-grid">
                {filteredProcessors.map(([key, info]) => (
                    <div key={key} className="processor-info-card">
                        <div className="processor-info-card-header">
                            <span className="processor-icon">{info.icon}</span>
                            <div>
                                <h3 className="processor-info-name">{info.name}</h3>
                                <span 
                                    className="processor-category-badge"
                                    style={{ 
                                        backgroundColor: CATEGORY_COLORS[info.category] + '20',
                                        color: CATEGORY_COLORS[info.category]
                                    }}
                                >
                                    {info.category}
                                </span>
                            </div>
                        </div>
                        <p className="processor-info-description">{info.description}</p>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default ProcessorInfo;

