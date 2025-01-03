// steps/reverseHandler.js
import { PHASES } from '../constants';
import { handleReverse } from '../utils/api';

export const handleReverseStep = async ({
  pyramidLayers,
}) => {
  try {

    // Prepare updates for reverse broadcasting
    const updates = pyramidLayers.flatMap(layer =>
      layer.nodes.map(node => ({
        node_id: node.data.id,
        broadcast_value: `Broadcast from ${node.data.id}`
      }))
    );

    // Send updates to backend
    await handleReverse(updates);

  } catch (error) {
    console.error('Error in reverse step:', error);
  }
};
