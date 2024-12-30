// steps/initHandler.js
import { PHASES } from '../constants';
import { initializeProcessors } from '../utils/api';

export const handleInitialStep = async ({ k, setDisplayPhase, setCurrentStep, setProcessorNames }) => {
  try {
    console.log('Initializing processors...');
    // Assuming initializeProcessors now returns an array of processor names
    const processorNames = await initializeProcessors(k);
    
    // Store the processor names in state
    setProcessorNames(processorNames);
    
    setDisplayPhase(PHASES.INIT);
    setCurrentStep(PHASES.OUTPUT_GIST);
    
    return processorNames;
  } catch (error) {
    console.error('Error in initial step:', error);
    return null;
  }
};