// steps/index.js

export const handleInitialStep = async (stepProps) => {
    const { k, selectedProcessors } = stepProps;

    try {
        const response = await fetch('http://localhost:5000/api/init', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                k: k,
                processors: selectedProcessors || []
            }),
        });

        if (!response.ok) {
            console.error('Server response not ok:', response.status);
            throw new Error('Failed to initialize processors');
        }

        const data = await response.json();
        console.log('Initialization response:', data);

        if (data.processorNames) {
            stepProps.setProcessorNames(data.processorNames);
            return data.processorNames;
        }

        console.error('Invalid response format:', data);
        return null;
    } catch (error) {
        console.error('Error in handleInitialStep:', error);
        return null;
    }
};