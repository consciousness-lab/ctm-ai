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
            const bottomNodeId = `g${i + 1}`;
            updates.push({
                processor_id: initNodeId,
                target_id: bottomNodeId,
            });
        }

        await outputGist(updates);
    } catch (error) {
        console.error('Error in output gist step:', error);
    }
};
