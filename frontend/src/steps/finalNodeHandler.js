import { PHASES } from '../constants';
import { updateNodeParents } from '../utils/api';

// steps/finalNodeHandler.js
export const handleFinalNodeStep = ({
  k,
  pyramidLayers,
  setElements,
  setCurrentStep,
  setDisplayPhase
}) => {
  setDisplayPhase(PHASES.FINAL_NODE);
  const finalLayer = pyramidLayers[k + 1];
  const finalNode = finalLayer.nodes[0].data.id;
  const parentNodes = pyramidLayers[k].nodes.map(node => node.data.id);

  updateNodeParents([{
    node_id: finalNode,
    parents: parentNodes,
  }]);

  setElements(prev => [...prev, ...finalLayer.nodes, ...finalLayer.edges]);
  setCurrentStep(PHASES.REVERSE);
};