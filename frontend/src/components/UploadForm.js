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
                formData.append('video_frames', file);
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
        <div style={{ maxWidth: 700, margin: '0 auto', padding: '10px' }}>
            <h2>Upload Files</h2>

            <form onSubmit={handleSubmit}>
                <div style={{ marginBottom: 10 }}>
                    <label>Query:</label>
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        style={{ width: '100%', padding: '6px', marginTop: '4px' }}
                    />
                </div>

                <div style={{ marginBottom: 10 }}>
                    <label>Text:</label>
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        rows={3}
                        style={{ width: '100%', padding: '6px', marginTop: '4px' }}
                    />
                </div>

                <div style={{ marginBottom: 10 }}>
                    <label>Images:</label>
                    <input
                        type="file"
                        accept="image/*"
                        multiple
                        onChange={handleImagesChange}
                        style={{ display: 'block', marginTop: '4px' }}
                    />
                    {imageFiles.length > 0 && (
                        <ul>
                            {imageFiles.map((file, idx) => (
                                <li key={idx}>
                                    {file.name}
                                    <button
                                        type="button"
                                        onClick={() => removeFile(file, imageFiles, setImageFiles)}
                                        style={{
                                            marginLeft: '10px',
                                            color: 'red',
                                            border: 'none',
                                            background: 'none',
                                            cursor: 'pointer'
                                        }}
                                    >
                                        Remove
                                    </button>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>

                <div style={{marginBottom: 10}}>
                    <label>Audios:</label>
                    <input
                        type="file"
                        accept="audio/*"
                        multiple
                        onChange={handleAudioChange}
                        style={{display: 'block', marginTop: '4px'}}
                    />
                    {audioFiles.length > 0 && (
                        <ul>
                            {audioFiles.map((file, idx) => (
                                <li key={idx}>
                                    {file.name}
                                    <button
                                        type="button"
                                        onClick={() => removeFile(file, audioFiles, setAudioFiles)}
                                        style={{
                                            marginLeft: '10px',
                                            color: 'red',
                                            border: 'none',
                                            background: 'none',
                                            cursor: 'pointer'
                                        }}
                                    >
                                        Remove
                                    </button>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>

                <div style={{marginBottom: 10}}>
                    <label>Videos:</label>
                    <input
                        type="file"
                        accept="video/*"
                        multiple
                        onChange={handleVideoChange}
                        style={{display: 'block', marginTop: '4px'}}
                    />
                    {videoFiles.length > 0 && (
                        <ul>
                            {videoFiles.map((file, idx) => (
                                <li key={idx}>
                                    {file.name}
                                    <button
                                        type="button"
                                        onClick={() => removeFile(file, videoFiles, setVideoFiles)}
                                        style={{
                                            marginLeft: '10px',
                                            color: 'red',
                                            border: 'none',
                                            background: 'none',
                                            cursor: 'pointer'
                                        }}
                                    >
                                        Remove
                                    </button>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>

                {uploadProgress > 0 && uploadProgress < 100 && (
                    <div style={{ marginBottom: '10px', backgroundColor: '#f3f3f3', borderRadius: '4px' }}>
                        <div
                            style={{
                                width: `${uploadProgress}%`,
                                height: '20px',
                                backgroundColor: '#4caf50',
                                borderRadius: '4px',
                                textAlign: 'center',
                                color: 'white',
                                lineHeight: '20px'
                            }}
                        >
                            {uploadProgress}%
                        </div>
                    </div>
                )}

                {errorMessage && <div style={{ color: 'red', marginBottom: '10px' }}>{errorMessage}</div>}
                {serverResponse && serverResponse.error && (
                    <div style={{ color: 'red', marginBottom: '10px' }}>{serverResponse.error}</div>
                )}
                {serverResponse && !serverResponse.error && (
                    <div style={{ color: 'green', marginBottom: '10px' }}>Upload Success！</div>
                )}

                <button type="submit" style={{marginTop: '10px'}}>
                    Submit to Server
                </button>
            </form>

            {serverResponse && (
                <div style={{marginTop: 20}}>
                    <h3>Server Response:</h3>
                    <pre
                        style={{
                            backgroundColor: '#f8f8f8',
                            padding: '10px',
                            border: '1px solid #ccc',
                        }}
                    >
            {JSON.stringify(serverResponse, null, 2)}
          </pre>
                </div>
            )}
        </div>
    );
};

export default UploadForm;
