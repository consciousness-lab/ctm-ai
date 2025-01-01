// components/ProcessorSelector.js
import React from 'react';

const ProcessorSelector = ({ processors, counts, onCountChange }) => {
    const processorTypes = {
        GPT4VProcessor: 'GPT-4V Processor',
        GPT4Processor: 'GPT-4 Processor',
        SearchEngineProcessor: 'Search Engine',
        WolframAlphaProcessor: 'Wolfram Alpha',
        BaseProcessor: 'Base Processor'
    };

    return (
        <div className="processor-selector">
            <h3 className="text-lg font-medium mb-4">Select Processors</h3>
            <div className="grid grid-cols-1 gap-4">
                {Object.entries(processorTypes).map(([type, label]) => (
                    <div key={type} className="flex items-center justify-between p-4 border rounded">
                        <div>
                            <span className="font-medium">{label}</span>
                            <span className="ml-2 text-gray-600">Count: {counts[type] || 0}</span>
                        </div>
                        <div className="space-x-2">
                            <button
                                onClick={() => onCountChange(type, Math.max(0, (counts[type] || 0) - 1))}
                                className="px-3 py-1 bg-red-100 text-red-600 rounded hover:bg-red-200"
                            >
                                -
                            </button>
                            <button
                                onClick={() => onCountChange(type, (counts[type] || 0) + 1)}
                                className="px-3 py-1 bg-green-100 text-green-600 rounded hover:bg-green-200"
                            >
                                +
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ProcessorSelector;