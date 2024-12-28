// steps/finalNodeHandler.js
import { PHASES } from '../constants';
import { updateFinalNode } from '../utils/api';

export const handleFinalNodeStep = async ({
  k,
  pyramidLayers,
  setElements,
  setCurrentStep,
  setDisplayPhase
}) => {
  try {
    setDisplayPhase(PHASES.FINAL_NODE);
    const finalLayer = pyramidLayers[k + 1];
    const finalNode = finalLayer.nodes[0].data.id;
    const parentNodes = pyramidLayers[k].nodes.map(node => node.data.id);

    // Send update to backend
    await updateFinalNode(finalNode, parentNodes, 'Final combined gist');

    // Update visualization
    setElements(prev => [...prev, ...finalLayer.nodes, ...finalLayer.edges]);
    setCurrentStep(PHASES.REVERSE);
  } catch (error) {
    console.error('Error in final node step:', error);
  }
};
