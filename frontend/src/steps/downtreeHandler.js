// steps/downtreeHandler.js
import { handleDowntree } from '../utils/api';

export const handleDowntreeStep = async () => {
  try {
    // Call backend to perform downtree broadcast
    await handleDowntree();
  } catch (error) {
    console.error('Error in downtree step:', error);
  }
};
