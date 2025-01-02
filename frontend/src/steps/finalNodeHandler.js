// steps/finalNodeHandler.js
import { updateFinalNode } from '../utils/api';
import { addFinalNode } from '../utils/graphBuilder';

export const handleFinalNodeStep = async ({
  k,
}) => {
  try {
    const { edges } = addFinalNode(k)[1];

    const parentNodes = Array.isArray(edges)
        ? edges.map((edge) => edge.data?.source).filter(Boolean) // Extract valid sources
        : [];

    await updateFinalNode('o', parentNodes, 'Final combined gist');
  } catch (error) {
    console.error('Error in final node step:', error);
  }
};
