import React from 'react';

export function parseDetailString(detailString) {
    if (!detailString) return null;

    const lines = detailString.split('\n');

    return lines.map((line, index) => {
        const colonIndex = line.indexOf(':');
        if (colonIndex === -1) {
            return (
                <div key={index}>
                    <span>{line}</span>
                </div>
            );
        }
        const title = line.substring(0, colonIndex).trim();
        const content = line.substring(colonIndex + 1).trim();

        return (
            <div key={index}>
                <span className="detail-title">{title}:</span>
                &nbsp;
                <span className="detail-content">{content}</span>
            </div>
        );
    });
}
