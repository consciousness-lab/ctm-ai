// steps/initHandler.js
import { PHASES } from '../constants';
import { initializeProcessors } from '../utils/api';

export const handleInitialStep = async ({
                                            selectedProcessors,
                                            setDisplayPhase,
                                            setCurrentStep,
                                            setProcessorNames,
                                        }) => {
    if (!Array.isArray(selectedProcessors) || selectedProcessors.length === 0) {
        console.error('No processors selected or invalid format for selectedProcessors.');
        return null;
    }

    try {
        setDisplayPhase(PHASES.INIT);

        const response = await initializeProcessors(selectedProcessors);

        console.log('Initialize response:', response);

        if (
            response &&
            Array.isArray(response.processorNames) &&
            response.processorNames.length > 0
        ) {
            setProcessorNames(response.processorNames);
            return response.processorNames;
        } else {
            console.error('Unexpected response format or empty processorNames:', response);
            return null;
        }
    } catch (error) {
        console.error('Error in handleInitialStep:', error.message || error);
        return null;
    }
};
