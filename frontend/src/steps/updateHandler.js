import { PHASES } from '../constants';

// steps/updateHandler.js
export const handleUpdateStep = ({
  setElements,
  setCurrentStep,
  setUptreeStep,
  setDisplayPhase
}) => {
  setDisplayPhase(PHASES.UPDATE);
  setElements(prev => prev.filter(el => el.data?.id?.startsWith('init')));
  setCurrentStep(PHASES.OUTPUT_GIST);
  setUptreeStep(1);
};