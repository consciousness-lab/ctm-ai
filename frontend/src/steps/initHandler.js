import { PHASES } from '../constants';
import { initializeProcessors } from '../utils/api';


export const handleInitialStep = async ({ k, setDisplayPhase, setProcessorNames }) => {
    if (typeof k !== 'number' || k <= 0) {
        console.error('Invalid input: k must be a positive number.');
        return null;
    }

    try {
        setDisplayPhase(PHASES.INIT);
        const response = await initializeProcessors(k);
        console.log('Initialize response:', response); // Debug log

        if (response && Array.isArray(response.processorNames) && response.processorNames.length > 0) {
            console.log('Processor names:', response.processorNames); // Debug log
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
