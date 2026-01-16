// utils/api.js

import axios from 'axios';

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';


const fetchWithError = async (url, options = {}) => {
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
};

export const initializeProcessors = async (selectedProcessors) => {
  try {
    const response = await fetch(`${BASE_URL}/init`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({ selected_processors: selectedProcessors }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('API response:', data);  // Debug log
    return data;  // Return the full response data
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

export const uploadFiles = async (formData, onUploadProgress) => {
  try {
    const response = await axios.post(`${BASE_URL}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Accept': 'application/json',
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


export const fuseGist = async (updates) => {
  try {
    const response = await fetch(`${BASE_URL}/fuse-gist`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ updates }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error in fuse gist:', error);
    throw error;
  }
};


export const fetchProcessorNeighborhoods = async () => {
  try {
    const response = await fetch(`${BASE_URL}/fetch-neighborhood`);
    if (!response.ok) {
      throw new Error('Failed to fetch processor neighborhoods');
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching processor neighborhoods:', error);
    return null;
  }
};

// Downtree broadcast (uses /reverse endpoint which calls downtree_broadcast)
export const handleDowntree = async () => {
  return fetchWithError(`${BASE_URL}/reverse`, {
    method: 'POST',
  });
};

// Fuse processors
export const handleFuse = async () => {
  return fetchWithError(`${BASE_URL}/fuse`, {
    method: 'POST',
  });
};

// Load example files from server
export const loadExampleFiles = async (imagePath, audioPath, query, text) => {
  return fetchWithError(`${BASE_URL}/load-example`, {
    method: 'POST',
    body: JSON.stringify({
      image_path: imagePath,
      audio_path: audioPath,
      query: query,
      text: text,
    }),
  });
};
