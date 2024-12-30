// steps/updateHandler.js
import { PHASES } from '../constants';
import { updateProcessors } from '../utils/api';

export const handleUpdateStep = async ({
  k,
  setUptreeStep,
}) => {
  try {

    // Prepare processor updates
    const updates = Array.from({ length: k }, (_, i) => ({
      processor_id: `init${i + 1}`,
      new_state: 'READY'
    }));

    // Send updates to backend
    await updateProcessors(updates);

    setUptreeStep(1);
  } catch (error) {
    console.error('Error in update step:', error);
  }
};
