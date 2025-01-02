import { updateProcessors } from '../utils/api';

export const handleUpdateStep = async ({
  k,
  setUptreeStep,
}) => {
  try {

    const updates = Array.from({ length: k }, (_, i) => ({
      processor_id: `init${i + 1}`,
      new_state: 'READY'
    }));

    await updateProcessors(updates);

    setUptreeStep(1);
  } catch (error) {
    console.error('Error in update step:', error);
  }
};
