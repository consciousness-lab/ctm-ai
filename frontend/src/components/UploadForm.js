// src/components/UploadForm.js

import React, { useState, useRef } from 'react';
import { uploadFiles } from '../utils/api';

const UploadForm = () => {
    const [query, setQuery] = useState('');
    const [text, setText] = useState('');
    const [files, setFiles] = useState([]);
    const [isDragging, setIsDragging] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');
    const [uploadProgress, setUploadProgress] = useState(0);
    const [serverResponse, setServerResponse] = useState(null);
    const fileInputRef = useRef(null);

    const MAX_FILE_SIZE = 1000 * 1024 * 1024;

    const getFileType = (file) => {
        if (file.type.startsWith('image/')) return 'image';
        if (file.type.startsWith('audio/')) return 'audio';
        if (file.type.startsWith('video/')) return 'video';
        return 'other';
    };

    const getFileIcon = (type) => {
        switch (type) {
            case 'image': return '🖼️';
            case 'audio': return '🎵';
            case 'video': return '🎬';
            default: return '📄';
        }
    };

    const isFileAlreadyAdded = (file) => {
        return files.some(
            (existingFile) =>
                existingFile.name === file.name && existingFile.size === file.size
        );
    };

    const processFiles = (fileList) => {
        const validFiles = Array.from(fileList).filter((file) => {
            const isValid = file.size <= MAX_FILE_SIZE && !isFileAlreadyAdded(file);
            const isAcceptedType = file.type.startsWith('image/') || 
                                   file.type.startsWith('audio/') || 
                                   file.type.startsWith('video/');
            return isValid && isAcceptedType;
        });

        if (validFiles.length !== fileList.length) {
            setErrorMessage('Some files were skipped (duplicate, too large, or unsupported type)');
            setTimeout(() => setErrorMessage(''), 3000);
        }

        if (validFiles.length > 0) {
            setFiles((prev) => [...prev, ...validFiles]);
        }
    };

    const handleFileChange = (e) => {
        if (e.target.files) {
            processFiles(e.target.files);
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files) {
            processFiles(e.dataTransfer.files);
        }
    };

    const removeFile = (fileToRemove) => {
        setFiles(files.filter((file) => file !== fileToRemove));
    };

    const handleDropZoneClick = () => {
        fileInputRef.current?.click();
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setErrorMessage('');
        setServerResponse(null);
        setUploadProgress(0);

        if (files.length === 0 && !query.trim() && !text.trim()) {
            setErrorMessage('Please select at least one file or enter a query and text.');
            return;
        }

        try {
            const formData = new FormData();
            formData.append('query', query);
            formData.append('text', text);

            files.forEach((file) => {
                const type = getFileType(file);
                if (type === 'image') formData.append('images', file);
                else if (type === 'audio') formData.append('audios', file);
                else if (type === 'video') formData.append('videos', file);
            });

            const result = await uploadFiles(formData, (progressEvent) => {
                const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                setUploadProgress(percent);
            });

            setServerResponse(result);
        } catch (error) {
            console.error('Error uploading data:', error);
            setServerResponse({ error: error.message });
        }
    };

    const imageCount = files.filter(f => getFileType(f) === 'image').length;
    const audioCount = files.filter(f => getFileType(f) === 'audio').length;
    const videoCount = files.filter(f => getFileType(f) === 'video').length;

    return (
        <div className="upload-form-container">
            <form onSubmit={handleSubmit} className="upload-form">
                <div className="form-grid">
                    <div className="form-group">
                        <label className="form-label"><strong>Query</strong></label>
                        <textarea
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            rows={1}
                            className="form-input"
                            placeholder="Enter your query"
                        />
                    </div>

                    <div className="form-group">
                        <label className="form-label"><strong>Text</strong></label>
                        <textarea
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            rows={3}
                            className="form-input"
                            placeholder="Enter additional text"
                        />
                    </div>

                    <div className="form-group">
                        <label className="form-label"><strong>Files</strong></label>
                        <div
                            className={`drop-zone ${isDragging ? 'dragging' : ''}`}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onDrop={handleDrop}
                            onClick={handleDropZoneClick}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/*,audio/*,video/*"
                                multiple
                                onChange={handleFileChange}
                                style={{ display: 'none' }}
                            />
                            <div className="drop-zone-content">
                                <span className="drop-zone-icon">📁</span>
                                <span className="drop-zone-text">
                                    {isDragging 
                                        ? 'Drop files here' 
                                        : 'Drag & drop files here, or click to browse'}
                                </span>
                                <span className="drop-zone-hint">
                                    Supports images, audio, and video files
                                </span>
                            </div>
                        </div>

                        {files.length > 0 && (
                            <div className="file-summary">
                                {imageCount > 0 && <span className="file-badge">🖼️ {imageCount}</span>}
                                {audioCount > 0 && <span className="file-badge">🎵 {audioCount}</span>}
                                {videoCount > 0 && <span className="file-badge">🎬 {videoCount}</span>}
                            </div>
                        )}

                        {files.length > 0 && (
                            <ul className="file-list">
                                {files.map((file, idx) => (
                                    <li key={idx} className="file-item">
                                        <span className="file-type-icon">{getFileIcon(getFileType(file))}</span>
                                        <span className="file-name">{file.name}</span>
                                        <button
                                            type="button"
                                            onClick={() => removeFile(file)}
                                            className="remove-button"
                                        >
                                            ×
                                        </button>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>
                </div>

                {uploadProgress > 0 && uploadProgress < 100 && (
                    <div className="progress-bar-container">
                        <div
                            className="progress-bar"
                            style={{ width: `${uploadProgress}%` }}
                        />
                    </div>
                )}

                {errorMessage && <div className="error-message">{errorMessage}</div>}
                {serverResponse?.error && <div className="error-message">{serverResponse.error}</div>}
                {serverResponse && !serverResponse.error && (
                    <div className="success-message">Upload Success!</div>
                )}

                <button type="submit" className="control-button start">
                    Upload
                </button>
            </form>
        </div>
    );
};

export default UploadForm;
