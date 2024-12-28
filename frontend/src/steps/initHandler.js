// steps/initHandler.js
import { PHASES } from '../constants';
import { initializeProcessors } from '../utils/api';

export const handleInitialStep = async ({ k, setDisplayPhase, setCurrentStep }) => {
  try {
    await initializeProcessors(k);
    setDisplayPhase(PHASES.INIT);
    setCurrentStep(PHASES.OUTPUT_GIST);
  } catch (error) {
    console.error('Error in initial step:', error);
  }
};
