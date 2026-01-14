// steps/fuseHandler.js
import { handleFuse } from '../utils/api';

export const handleFuseStep = async () => {
  try {
    // Call backend to perform processor fusion
    await handleFuse();
  } catch (error) {
    console.error('Error in fuse step:', error);
  }
};
