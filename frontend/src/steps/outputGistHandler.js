import { PHASES } from '../constants';
import { updateNodeParents } from '../utils/api';

// steps/outputGistHandler.js
export const handleOutputGistStep = ({
  k,
  pyramidLayers,
  setElements,
  setCurrentLayerIndex,
  setCurrentStep,
  setDisplayPhase
}) => {
  setDisplayPhase(PHASES.OUTPUT_GIST);
  const nextLayer = pyramidLayers[1];
  const initToBottomEdges = [];
  const parentUpdates = [];

  for (let i = 0; i < k; i++) {
    const initNodeId = `init${i + 1}`;
    const bottomNodeId = `n${i + 1}`;
    
    initToBottomEdges.push({
      data: {
        source: initNodeId,
        target: bottomNodeId,
        id: `e${initNodeId}-${bottomNodeId}`,
      },
    });

    parentUpdates.push({
      node_id: bottomNodeId,
      parents: [initNodeId],
    });
  }

  updateNodeParents(parentUpdates);
  setElements(prev => [...prev, ...nextLayer.nodes, ...nextLayer.edges, ...initToBottomEdges]);
  setCurrentLayerIndex(2);
  setCurrentStep(PHASES.UPTREE);
};