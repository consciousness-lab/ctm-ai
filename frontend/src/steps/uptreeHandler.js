import { handleUptreeUpdate } from '../utils/api';
import { addUptreeNodes, addUptreeEdges } from '../utils/graphBuilder';

export const handleUptreeStep = async ({
  k,
  uptreeStep,
}) => {
  try {
    const updates = [];

    console.log('handleUptreeStep:', k, uptreeStep);
    // Use uptreeStep directly (matches modifyGraph in App.js)
    const { nodes } = addUptreeNodes(k, uptreeStep);
    const { edges } = addUptreeEdges(k, uptreeStep);

    console.log('nodes:', nodes);
    console.log('edges:', edges);
    nodes.forEach((node) => {
      const nodeId = node.data.id;

      const parents = edges
        .filter((edge) => edge.data.target === nodeId)
        .map((edge) => edge.data.source);

      updates.push({
        node_id: nodeId,
        parents,
      });
    });

    await handleUptreeUpdate(uptreeStep, updates);
  } catch (error) {
    console.error('Error in uptree step:', error);
  }
};
