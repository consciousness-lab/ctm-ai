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
  return fetchWithError(`${BASE_URL}/init`, {
    method: 'POST',
    body: JSON.stringify({ k }),
  });
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
