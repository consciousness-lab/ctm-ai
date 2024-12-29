// steps/uptreeHandler.js
import { PHASES } from '../constants';
import { handleUptreeUpdate } from '../utils/api';

export const handleUptreeStep = async ({
  k,
  pyramidLayers,
  currentLayerIndex,
  uptreeStep,
  setElements,
  setCurrentLayerIndex,
  setCurrentStep,
  setUptreeStep,
  setDisplayPhase
}) => {
  try {
    setDisplayPhase(PHASES.UPTREE);
    const nextLayer = pyramidLayers[currentLayerIndex];
    const updates = [];

    // Create updates for current layer
    nextLayer.edges.forEach(edge => {
      const { source, target } = edge.data;
      updates.push({
        node_id: target,
        parents: [source],
      });
    });

    // Send updates to backend
    await handleUptreeUpdate(uptreeStep, updates);

    // Update visualization
    setElements(prev => [...prev, ...nextLayer.nodes, ...nextLayer.edges]);
    setCurrentLayerIndex(prev => prev + 1);

    if (uptreeStep >= k - 1) {
      setCurrentStep(PHASES.FINAL_NODE);
      setUptreeStep(1);
    } else {
      setUptreeStep(prev => prev + 1);
    }
  } catch (error) {
    console.error('Error in uptree step:', error);
  }
};
