// steps/generateHandler.js
import { updateFinalNode } from '../utils/api';
import { addFinalNode } from '../utils/graphBuilder';

export const handleGenerateStep = async ({
  k,
}) => {
  try {
    const { edges } = addFinalNode(k);

    const parentNodes = Array.isArray(edges)
        ? edges.map((edge) => edge.data?.source).filter(Boolean)
        : [];

    await updateFinalNode('o', parentNodes, 'Final combined gist');
  } catch (error) {
    console.error('Error in generate step:', error);
  }
};
