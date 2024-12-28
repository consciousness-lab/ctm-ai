import { PHASES } from '../constants';

// steps/reverseHandler.js
export const handleReverseStep = ({
  setElements,
  setCurrentStep,
  setDisplayPhase
}) => {
  setDisplayPhase(PHASES.REVERSE);
  setElements(prev => prev.map(el => {
    if (el.data.source && el.data.target) {
      return {
        ...el,
        data: {
          ...el.data,
          source: el.data.target,
          target: el.data.source,
          id: `e${el.data.target}-${el.data.source}`
        }
      };
    }
    return el;
  }));
  setCurrentStep(PHASES.UPDATE);
};