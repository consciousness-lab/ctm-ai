// src/components/ProcessorSelector.js

import React from 'react';

function ProcessorSelector({ allProcessors, selectedProcessors, onChange }) {
    const handleToggle = (processor) => {
        let newList;
        if (selectedProcessors.includes(processor)) {
            newList = selectedProcessors.filter((p) => p !== processor);
        } else {
            newList = [...selectedProcessors, processor];
        }
        onChange(newList);
    };

    // Format processor name for display
    const formatProcessorName = (name) => {
        return name.replace('Processor', '');
    };

    return (
        <div className="processor-selector">
            <label className="processor-selector-label">Select Processors</label>
            <div className="processor-buttons">
                {allProcessors.map((proc) => {
                    const isSelected = selectedProcessors.includes(proc);
                    return (
                        <button
                            key={proc}
                            type="button"
                            onClick={() => handleToggle(proc)}
                            className={`processor-btn ${isSelected ? 'selected' : ''}`}
                        >
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
        </div>
    );
}

export default ProcessorSelector;
