// src/components/ProcessorSelector.js

import React, { useState } from 'react';
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

    return (
        <div style={{marginTop: '10px'}}>
            <strong>Processors:</strong>
            <div style={{display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '5px', marginBottom: '5px'}}>
                {allProcessors.map((proc) => {
                    const isSelected = selectedProcessors.includes(proc);
                    return (
                        <button
                            key={proc}
                            onClick={() => handleToggle(proc)}
                            style={{
                                backgroundColor: isSelected ? '#4caf50' : '#f0f0f0',
                                color: isSelected ? '#fff' : '#000',
                                border: '1px solid #ccc',
                                padding: '5px 8px',
                                cursor: 'pointer',
                            }}
                        >
                            {proc}
                        </button>
                    );
                })}
            </div>
            <div style={{marginTop: '15px', marginBottom: '15px'}}>
                <strong>Selected Processors:</strong> {selectedProcessors.join(', ') || 'None'}
            </div>
        </div>
    );
}

export default ProcessorSelector;
