// steps/outputGistHandler.js
import { PHASES } from '../constants';
import { outputGist } from '../utils/api';

export const handleOutputGistStep = async ({
  k,
  pyramidLayers,
  setElements,
  setCurrentLayerIndex,
  setCurrentStep,
  setDisplayPhase
}) => {
  try {
    setDisplayPhase(PHASES.OUTPUT_GIST);
    const nextLayer = pyramidLayers[1];
    const updates = [];

    // Create gist updates from processors to bottom layer
    for (let i = 0; i < k; i++) {
      const initNodeId = `init${i + 1}`;
      const bottomNodeId = `n${i + 1}`;

      updates.push({
        processor_id: initNodeId,
        target_id: bottomNodeId,
        gist: `Gist from processor ${i + 1}`
      });
    }

    // Send updates to backend
    await outputGist(updates);

    // Update visualization
    const initToBottomEdges = updates.map(update => ({
      data: {
        source: update.processor_id,
        target: update.target_id,
        id: `e${update.processor_id}-${update.target_id}`,
      },
    }));

    setElements(prev => [...prev, ...nextLayer.nodes, ...nextLayer.edges, ...initToBottomEdges]);
    setCurrentLayerIndex(2);
    setCurrentStep(PHASES.UPTREE);
  } catch (error) {
    console.error('Error in output gist step:', error);
  }
};
