import { PHASES } from '../constants';

// steps/initHandler.js
export const handleInitialStep = ({ setDisplayPhase, setCurrentStep }) => {
  setDisplayPhase(PHASES.INIT);
  setCurrentStep(PHASES.OUTPUT_GIST);
};