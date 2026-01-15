// steps/outputGistHandler.js
import { outputGist } from '../utils/api';

export const handleOutputGistStep = async ({
    k,
    processorNames,
}) => {
    try {
        const updates = [];

        for (let i = 0; i < k; i++) {
            const initNodeId = processorNames[i];
            // Use n instead of g since we merged gist and fused layers
            const fusedNodeId = `n${i + 1}`;
            updates.push({
                processor_id: initNodeId,
                target_id: fusedNodeId,
            });
        }

        await outputGist(updates);
    } catch (error) {
        console.error('Error in output gist step:', error);
    }
};
