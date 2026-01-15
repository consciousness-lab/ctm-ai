// steps/fuseHandler.js
import { fuseGist } from '../utils/api';

export const handleFuseStep = async ({ k }) => {
  try {
    // Call backend to perform processor fusion (fuse_processor)
    const updates = [];
    for (let i = 0; i < k; i++) {
      const fusedNodeId = `n${i + 1}`;
      const sourceNode1 = `g${i + 1}`;
      updates.push({
        fused_node_id: fusedNodeId,
        source_nodes: [sourceNode1]
      });
    }
    await fuseGist(updates);
  } catch (error) {
    console.error('Error in fuse step:', error);
  }
};
