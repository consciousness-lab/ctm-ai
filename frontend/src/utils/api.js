// utils/api.js
import axios from 'axios';

const BASE_URL = 'http://localhost:5000/api';

const fetchWithError = async (url, options = {}) => {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

export const initializeProcessors = async (k) => {
  try {
    const response = await fetch('http://localhost:5000/api/init', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ k }),
    });
    
    const data = await response.json();
    return data.processorNames; // Assuming your API returns processor names
  } catch (error) {
    console.error('Error initializing processors:', error);
    throw error;
  }
};

export const outputGist = async (updates) => {
  return fetchWithError(`${BASE_URL}/output-gist`, {
    method: 'POST',
    body: JSON.stringify({ updates }),
  });
};

export const handleUptreeUpdate = async (layer, updates) => {
  return fetchWithError(`${BASE_URL}/uptree`, {
    method: 'POST',
    body: JSON.stringify({ layer, updates }),
  });
};

export const updateFinalNode = async (nodeId, parents, finalGist) => {
  return fetchWithError(`${BASE_URL}/final-node`, {
    method: 'POST',
    body: JSON.stringify({
      node_id: nodeId,
      parents,
      final_gist: finalGist,
    }),
  });
};

export const handleReverse = async (updates) => {
  return fetchWithError(`${BASE_URL}/reverse`, {
    method: 'POST',
    body: JSON.stringify({ updates }),
  });
};

export const updateProcessors = async (updates) => {
  return fetchWithError(`${BASE_URL}/update-processors`, {
    method: 'POST',
    body: JSON.stringify({ updates }),
  });
};

export const getNodeDetails = async (nodeId) => {
  return fetchWithError(`${BASE_URL}/nodes/${nodeId}`);
};

export const getCurrentState = async () => {
  return fetchWithError(`${BASE_URL}/state`);
};

export const uploadFiles = async (formData, onUploadProgress) => {
  try {
    const response = await axios.post(`${BASE_URL}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress,
    });

    return response.data;
  } catch (error) {
    if (error.response) {
      throw new Error(error.response.data.error || 'Upload failed');
    } else {
      throw new Error('Network error');
    }
  }
};
