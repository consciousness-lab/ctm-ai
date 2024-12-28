// steps/reverseHandler.js
import { PHASES } from '../constants';
import { handleReverse } from '../utils/api';

export const handleReverseStep = async ({
  pyramidLayers,
  setElements,
  setCurrentStep,
  setDisplayPhase
}) => {
  try {
    setDisplayPhase(PHASES.REVERSE);

    // Prepare updates for reverse broadcasting
    const updates = pyramidLayers.flatMap(layer =>
      layer.nodes.map(node => ({
        node_id: node.data.id,
        broadcast_value: `Broadcast from ${node.data.id}`
      }))
    );

    // Send updates to backend
    await handleReverse(updates);

    // Update visualization - reverse all edges
    setElements(prev => prev.map(el => {
      if (el.data.source && el.data.target) {
        return {
          ...el,
          data: {
            ...el.data,
            source: el.data.target,
            target: el.data.source,
            id: `e${el.data.target}-${el.data.source}`
          }
        };
      }
      return el;
    }));

    setCurrentStep(PHASES.UPDATE);
  } catch (error) {
    console.error('Error in reverse step:', error);
  }
};
