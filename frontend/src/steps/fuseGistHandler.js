// steps/fuseGistHandler.js
import { PHASES } from '../constants';
import { fuseGist } from '../utils/api';

export const handleFuseGistStep = async ({
  k,
  setDisplayPhase
}) => {
  try {
    const updates = [];
    const fusedLayer = {
      nodes: [],
      edges: []
    };

    // Create fused nodes (one for each pair of processors)
    for (let i = 0; i < k; i++) {
      const fusedNodeId = `n${i + 1}`;
      const sourceNode1 = `g${i + 1}`;

      // Add fused node
      fusedLayer.nodes.push({
        data: {
          id: fusedNodeId,
          label: `n${i + 1}`
        }
      });

      // Add edges from source nodes to fused node
      fusedLayer.edges.push({
        data: {
          source: sourceNode1,
          target: fusedNodeId,
          id: `e${sourceNode1}-${fusedNodeId}`
        }
      });

      // Prepare update for backend
      updates.push({
        fused_node_id: fusedNodeId,
        source_nodes: [sourceNode1]
      });
    }

    // Send updates to backend
    await fuseGist(updates);
  } catch (error) {
    console.error('Error in fuse gist step:', error);
  }
};
