import { PHASES } from '../constants';
import { updateNodeParents } from '../utils/api';

// steps/uptreeHandler.js
export const handleUptreeStep = ({
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
  setDisplayPhase(PHASES.UPTREE);
  const nextLayer = pyramidLayers[currentLayerIndex];
  const parentUpdates = [];

  nextLayer.edges.forEach(edge => {
    const { source, target } = edge.data;
    parentUpdates.push({
      node_id: target,
      parents: [source],
    });
  });

  updateNodeParents(parentUpdates);
  setElements(prev => [...prev, ...nextLayer.nodes, ...nextLayer.edges]);
  setCurrentLayerIndex(prev => prev + 1);
  
  if (uptreeStep >= k - 1) {
    setCurrentStep(PHASES.FINAL_NODE);
    setUptreeStep(1);
  } else {
    setUptreeStep(prev => prev + 1);
  }
};