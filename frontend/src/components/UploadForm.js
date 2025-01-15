// src/components/UploadForm.js

import React, { useState } from 'react';
import { uploadFiles } from '../utils/api';

const UploadForm = () => {
    const [query, setQuery] = useState('');
    const [text, setText] = useState('');
    const [imageFiles, setImageFiles] = useState([]);
    const [audioFiles, setAudioFiles] = useState([]);
    const [videoFiles, setVideoFiles] = useState([]);
    const [errorMessage, setErrorMessage] = useState('');
    const [uploadProgress, setUploadProgress] = useState(0);
    const [serverResponse, setServerResponse] = useState(null);

    const MAX_FILE_SIZE = 1000 * 1024 * 1024;

    const isFileAlreadyAdded = (file, fileList) => {
        return fileList.some(
            (existingFile) =>
                existingFile.name === file.name && existingFile.size === file.size
        );
    };

    const handleImagesChange = (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            const newFiles = Array.from(files).filter(
                (file) => !isFileAlreadyAdded(file, imageFiles) && file.size <= MAX_FILE_SIZE
            );

            if (newFiles.length !== files.length) {
                setErrorMessage('File size exceeds the limit');
            }

            if (newFiles.length > 0) {
                setImageFiles((prevFiles) => [...prevFiles, ...newFiles]);
            }
        }
    };

    const handleAudioChange = (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            const newFiles = Array.from(files).filter(
                (file) => !isFileAlreadyAdded(file, audioFiles) && file.size <= MAX_FILE_SIZE
            );

            if (newFiles.length !== files.length) {
                setErrorMessage('File size exceeds the limit.');
            }

            if (newFiles.length > 0) {
                setAudioFiles((prevFiles) => [...prevFiles, ...newFiles]);
            }
        }
    };

    const handleVideoChange = (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            const newFiles = Array.from(files).filter(
                (file) => !isFileAlreadyAdded(file, videoFiles) && file.size <= MAX_FILE_SIZE
            );

            if (newFiles.length !== files.length) {
                setErrorMessage('File size exceeds the limit.');
            }

            if (newFiles.length > 0) {
                setVideoFiles((prevFiles) => [...prevFiles, ...newFiles]);
            }
        }
    };

    const removeFile = (fileToRemove, fileList, setFileList) => {
        setFileList(fileList.filter((file) => file !== fileToRemove));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setErrorMessage('');
        setServerResponse(null);
        setUploadProgress(0);

        if (imageFiles.length === 0 && audioFiles.length === 0 && videoFiles.length === 0 && !query.trim() && !text.trim()) {
            setErrorMessage('Please select at least one file or enter a query and text.');
            return;
        }

        try {
            const formData = new FormData();

            formData.append('query', query);
            formData.append('text', text);

            imageFiles.forEach((file) => {
                formData.append('images', file);
            });

            audioFiles.forEach((file) => {
                formData.append('audios', file);
            });

            videoFiles.forEach((file) => {
                formData.append('videos', file);
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

                    <div className="file-upload-section">
                        <div className="file-group">
                            <label className="form-label"><strong>Image</strong></label>
                            <input
                                type="file"
                                accept="image/*"
                                multiple
                                onChange={handleImagesChange}
                                className="file-input"
                            />
                            {imageFiles.length > 0 && (
                                <ul className="file-list">
                                    {imageFiles.map((file, idx) => (
                                        <li key={idx} className="file-item">
                                            <span className="file-name">{file.name}</span>
                                            <button
                                                type="button"
                                                onClick={() => removeFile(file, imageFiles, setImageFiles)}
                                                className="remove-button"
                                            >
                                                ×
                                            </button>
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </div>

                        <div className="file-group">
                            <label className="form-label"><strong>Audio</strong></label>
                            <input
                                type="file"
                                accept="audio/*"
                                multiple
                                onChange={handleAudioChange}
                                className="file-input"
                            />
                            {audioFiles.length > 0 && (
                                <ul className="file-list">
                                    {audioFiles.map((file, idx) => (
                                        <li key={idx} className="file-item">
                                            <span className="file-name">{file.name}</span>
                                            <button
                                                type="button"
                                                onClick={() => removeFile(file, audioFiles, setAudioFiles)}
                                                className="remove-button"
                                            >
                                                ×
                                            </button>
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </div>

                        <div className="file-group">
                            <label className="form-label"><strong>Video</strong></label>
                            <input
                                type="file"
                                accept="video/*"
                                multiple
                                onChange={handleVideoChange}
                                className="file-input"
                            />
                            {videoFiles.length > 0 && (
                                <ul className="file-list">
                                    {videoFiles.map((file, idx) => (
                                        <li key={idx} className="file-item">
                                            <span className="file-name">{file.name}</span>
                                            <button
                                                type="button"
                                                onClick={() => removeFile(file, videoFiles, setVideoFiles)}
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
                </div>

                {uploadProgress > 0 && uploadProgress < 100 && (
                    <div className="progress-bar-container">
                        <div
                            className="progress-bar"
                            style={{ width: `${uploadProgress}%` }}
                        >
                            {uploadProgress}%
                        </div>
                    </div>
                )}

                {errorMessage && <div className="error-message">{errorMessage}</div>}
                {serverResponse?.error && <div className="error-message">{serverResponse.error}</div>}
                {serverResponse && !serverResponse.error && (
                    <div className="success-message">Upload Success!</div>
                )}

                <button type="submit" className={`control-button start`}>
                    Submit
                </button>
            </form>
        </div>
    );
};

export default UploadForm;
