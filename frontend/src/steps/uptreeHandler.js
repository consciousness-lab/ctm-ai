import { handleUptreeUpdate } from '../utils/api';

// Get pairing info for uptree competition
// Returns array of { node_id, parents } for each competition pair
// node_id uses the same format as frontend graph nodes: layer{uptreeStep}_{idx}
function getUptreePairings(activeNodes, uptreeStep) {
  const pairings = [];
  
  for (let i = 0; i < activeNodes.length; i += 2) {
    const node1 = activeNodes[i];
    const node2 = activeNodes[i + 1];
    
    // Use consistent ID with frontend graph nodes
    const nodeId = `layer${uptreeStep}_${Math.floor(i / 2)}`;
    
    if (node2) {
      // Pair of two nodes competing
      pairings.push({
        node_id: nodeId,
        parents: [node1, node2],
      });
    } else {
      // Single node, auto-advance
      pairings.push({
        node_id: nodeId,
        parents: [node1],
      });
    }
  }
  
  return pairings;
}

// handleUptreeStep now returns competition results
export const handleUptreeStep = async ({
  k,
  uptreeStep,
  activeNodes, // Current active nodes in this layer
}) => {
  try {
    console.log('handleUptreeStep:', k, uptreeStep, 'activeNodes:', activeNodes);
    
    // Generate pairings based on current active nodes (pass uptreeStep for consistent IDs)
    const updates = getUptreePairings(activeNodes, uptreeStep);
    
    console.log('Uptree pairings:', updates);

    const response = await handleUptreeUpdate(uptreeStep, updates);
    
    console.log('Uptree response:', response);
    
    // Return competition results for graph updates
    return response?.competition_results || [];
  } catch (error) {
    console.error('Error in uptree step:', error);
    return [];
  }
};
