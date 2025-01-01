// utils/api.js
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

export const getCurrentState = async () => {
  return fetchWithError(`${BASE_URL}/state`);
};

export const fuseGist = async (updates) => {
  try {
    const response = await fetch('http://localhost:5000/api/fuse-gist', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
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
    const response = await fetch('http://localhost:5000/api/fetch-neighborhood');
    if (!response.ok) {
      throw new Error('Failed to fetch processor neighborhoods');
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching processor neighborhoods:', error);
    return null;
  }
};
