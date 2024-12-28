// utils/api.js
export const updateNodeParents = async (parentUpdates) => {
  try {
    const response = await fetch('http://localhost:5000/api/update-node-parents', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ updates: parentUpdates }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('Updated node parents:', data);
  } catch (error) {
    console.error('Error updating node parents:', error);
  }
};