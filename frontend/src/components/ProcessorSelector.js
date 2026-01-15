// src/components/ProcessorSelector.js

import React, { useState } from 'react';

const PROCESSOR_DESCRIPTIONS = {
    'VisionProcessor': {
        name: 'Vision Processor',
        description: 'Analyzes images and visual content. Can detect objects, scenes, text in images, and perform visual question answering.',
        category: 'Multimodal',
        icon: '️'
    },
    'LanguageProcessor': {
        name: 'Language Processor',
        description: 'Processes natural language text. Handles text understanding, generation, translation, summarization, and semantic analysis.',
        category: 'Text',
        icon: ''
    },
    'SearchProcessor': {
        name: 'Search Processor',
        description: 'Searches the web for information using Google Search. Finds relevant articles, facts, and current information from the internet.',
        category: 'Web',
        icon: ''
    },
    'CodeProcessor': {
        name: 'Code Processor',
        description: 'Analyzes and generates code. Can understand programming languages, debug code, explain code logic, and generate code snippets.',
        category: 'Technical',
        icon: ''
    },
    'AudioProcessor': {
        name: 'Audio Processor',
        description: 'Processes audio files and speech. Can transcribe audio, analyze speech patterns, detect emotions, and extract audio features.',
        category: 'Multimodal',
        icon: ''
    },
    'VideoProcessor': {
        name: 'Video Processor',
        description: 'Analyzes video content frame by frame. Can understand video scenes, actions, objects, and temporal relationships in video sequences.',
        category: 'Multimodal',
        icon: ''
    },
    'FinanceProcessor': {
        name: 'Finance Processor',
        description: 'Retrieves real-time financial data including stock prices, market trends, cryptocurrency prices, forex rates, and financial news from Google Finance.',
        category: 'Data',
        icon: ''
    },
    'GeoDBProcessor': {
        name: 'GeoDB Processor',
        description: 'Accesses geographic and location data. Provides city information, population data, geographic coordinates, and location-based insights.',
        category: 'Data',
        icon: ''
    },
    'TwitterProcessor': {
        name: 'Twitter Processor',
        description: 'Searches Twitter/X for tweets, trends, and social media content. Can find recent tweets, hashtags, and user activity.',
        category: 'Social',
        icon: ''
    },
    'WeatherProcessor': {
        name: 'Weather Processor',
        description: 'Gets real-time weather data for any location worldwide. Provides current conditions, forecasts, and weather analytics.',
        category: 'Data',
        icon: '️'
    },
    'YouTubeProcessor': {
        name: 'YouTube Processor',
        description: 'Searches YouTube for videos, channels, and content. Can find video metadata, channel information, and video recommendations.',
        category: 'Social',
        icon: ''
    },
    'MathProcessor': {
        name: 'Math Processor',
        description: 'Solves mathematical problems using Wolfram|Alpha. Handles calculations, equations, mathematical queries, and computational problems.',
        category: 'Technical',
        icon: ''
    },
    'NewsProcessor': {
        name: 'News Processor',
        description: 'Retrieves news articles. Can search for top headlines, topic-specific news, and local news from around the world.',
        category: 'Data',
        icon: ''
    },
    'SocialProcessor': {
        name: 'Social Processor',
        description: 'Searches for social media profile links across platforms including Facebook, Instagram, TikTok, Twitter, LinkedIn, YouTube, and more.',
        category: 'Social',
        icon: ''
    },
    'ExerciseProcessor': {
        name: 'Exercise Processor',
        description: 'Provides comprehensive exercise database with over 1,300 exercises. Includes workout plans, muscle targeting, equipment requirements, and exercise demonstrations.',
        category: 'Health',
        icon: ''
    },
    'MusicProcessor': {
        name: 'Music Processor',
        description: 'Retrieves music information, artist data, song details, and music recommendations. Can search for songs, albums, and music metadata.',
        category: 'Media',
        icon: ''
    },
    'APIProcessor': {
        name: 'API Processor',
        description: 'Generic API processor for custom integrations and external service calls.',
        category: 'Technical',
        icon: ''
    },
    'ToolProcessor': {
        name: 'Tool Processor',
        description: 'Executes various tools and utilities for specialized tasks and operations.',
        category: 'Technical',
        icon: '️'
    },
};

function ProcessorSelector({ allProcessors, selectedProcessors, onChange }) {
    const [hoveredProcessor, setHoveredProcessor] = useState(null);
    const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });

    const handleToggle = (processor) => {
        let newList;
        if (selectedProcessors.includes(processor)) {
            newList = selectedProcessors.filter((p) => p !== processor);
        } else {
            newList = [...selectedProcessors, processor];
        }
        onChange(newList);
    };

    const handleMouseEnter = (proc, event) => {
        const rect = event.target.getBoundingClientRect();
        setTooltipPosition({
            x: rect.left + rect.width / 2,
            y: rect.top
        });
        setHoveredProcessor(proc);
    };

    const handleMouseLeave = () => {
        setHoveredProcessor(null);
    };

    // Format processor name for display
    const formatProcessorName = (name) => {
        return name.replace('Processor', '');
    };

    const getProcessorInfo = (proc) => {
        return PROCESSOR_DESCRIPTIONS[proc] || {
            name: formatProcessorName(proc),
            description: 'No description available.',
            category: 'Other',
            icon: '⚙️'
        };
    };

    return (
        <div className="processor-selector">
            <label className="processor-selector-label">Select Processors</label>
            <div className="processor-buttons">
                {allProcessors.map((proc) => {
                    const isSelected = selectedProcessors.includes(proc);
                    const info = getProcessorInfo(proc);
                    return (
                        <button
                            key={proc}
                            type="button"
                            onClick={() => handleToggle(proc)}
                            onMouseEnter={(e) => handleMouseEnter(proc, e)}
                            onMouseLeave={handleMouseLeave}
                            className={`processor-btn ${isSelected ? 'selected' : ''}`}
                        >
                            <span className="processor-icon">{info.icon}</span>
                            {formatProcessorName(proc)}
                        </button>
                    );
                })}
            </div>
            <div className="selected-processors">
                <strong>Active:</strong> {selectedProcessors.length > 0 
                    ? selectedProcessors.map(p => formatProcessorName(p)).join(', ') 
                    : 'None selected'}
            </div>
            
            {/* Tooltip */}
            {hoveredProcessor && (
                <div 
                    className="processor-tooltip"
                    style={{
                        left: tooltipPosition.x,
                        top: tooltipPosition.y
                    }}
                >
                    <div className="tooltip-header">
                        <span className="tooltip-icon">{getProcessorInfo(hoveredProcessor).icon}</span>
                        <span className="tooltip-name">{getProcessorInfo(hoveredProcessor).name}</span>
                    </div>
                    <div className="tooltip-category">
                        {getProcessorInfo(hoveredProcessor).category}
                    </div>
                    <div className="tooltip-description">
                        {getProcessorInfo(hoveredProcessor).description}
                    </div>
                </div>
            )}
        </div>
    );
}

export default ProcessorSelector;
