import React, { useState, useEffect } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import { PHASES, PHASE_DESCRIPTIONS } from './constants';
import { addProcessorNodes } from './utils/graphBuilder';
import { layout, stylesheet } from './config/cytoscapeConfig';
import {
    addGistNodes,
    addGistEdges,
    addFusedNodes,
    addFusedEdges,
    addUptreeNodes,
    addUptreeEdges,
    addFinalNode,
} from './utils/graphBuilder';

import {
    handleInitialStep,
    handleOutputGistStep,
    handleFuseGistStep,
    handleUptreeStep,
    handleFinalNodeStep,
    handleReverseStep,
    handleUpdateStep,
} from './steps/index';
import './App.css';
import UploadForm from "./components/UploadForm";
import ProcessorSelector from "./components/ProcessorSelector";

const App = () => {
    const [elements, setElements] = useState([]);
    const [processorNames, setProcessorNames] = useState([]);
    const [initialized, setInitialized] = useState(false);
    const [selectedNode, setSelectedNode] = useState(null);
    const [nodeDetailText, setNodeDetailText] = useState('');
    const [currentStep, setCurrentStep] = useState(PHASES.INIT);
    const [uptreeStep, setUptreeStep] = useState(1);
    const [displayPhase, setDisplayPhase] = useState(PHASES.INIT);
    const [processorCounts, setProcessorCounts] = useState({
        BaseProcessor: 0,
        GPT4VProcessor: 0,
        GPT4Processor: 0,
        SearchEngineProcessor: 0,
        WolframAlphaProcessor: 0
    });

    const processorTypes = {
        BaseProcessor: 'Base Processor',
        GPT4VProcessor: 'GPT-4V Processor',
        GPT4Processor: 'GPT-4 Processor',
        SearchEngineProcessor: 'Search Engine',
        WolframAlphaProcessor: 'Wolfram Alpha'
    };

    const handleProcessorCountChange = (type, count) => {
        setProcessorCounts(prev => ({
            ...prev,
            [type]: count
        }));
    };


    const modifyGraph = (totalProcessors) => {
        const updateElementsForPhase = (newElements) => {
            setElements((prevElements) => [
                ...prevElements,
                ...newElements.nodes,
                ...newElements.edges,
            ]);
        };

        switch (currentStep) {
            case PHASES.OUTPUT_GIST: {
                const newElements = {
                    nodes: addGistNodes(totalProcessors).nodes,
                    edges: addGistEdges(totalProcessors, processorNames).edges,
                };
                updateElementsForPhase(newElements);
                break;
            }

            case PHASES.FUSE_GIST: {
                const newElements = {
                    nodes: addFusedNodes(totalProcessors).nodes,
                    edges: addFusedEdges(totalProcessors).edges,
                };
                updateElementsForPhase(newElements);
                break;
            }

            case PHASES.UPTREE: {
                const newElements = {
                    nodes: addUptreeNodes(totalProcessors, uptreeStep + 1).nodes,
                    edges: addUptreeEdges(totalProcessors, uptreeStep + 1).edges,
                };
                updateElementsForPhase(newElements);
                break;
            }

            case PHASES.FINAL_NODE: {
                const newElements = addFinalNode(totalProcessors);
                updateElementsForPhase(newElements);
                break;
            }

            case PHASES.REVERSE: {
                setElements((prevElements) =>
                    prevElements.map((element) =>
                        element.data?.source && element.data?.target
                            ? {
                                ...element,
                                data: {
                                    ...element.data,
                                    source: element.data.target,
                                    target: element.data.source,
                                },
                            }
                            : element
                    )
                );
                break;
            }

            case PHASES.UPDATE: {
                setElements((prevElements) => {
                    const processorNodes = prevElements.filter((element) =>
                        element.data?.label?.toLowerCase().includes('processor')
                    );

                    const processorNodeIds = processorNodes.map((node) => node.data.id);
                    const validEdges = prevElements.filter(
                        (element) =>
                            element.data?.source &&
                            element.data?.target &&
                            processorNodeIds.includes(element.data.source) &&
                            processorNodeIds.includes(element.data.target)
                    );

                    return [...processorNodes, ...validEdges];
                });
                break;
            }

            default:
                console.error('Unknown phase for modifying the graph');
                break;
        }
    };


    const handleStep = async() => {
        const totalProcessors = Object.values(processorCounts).reduce((sum, count) => sum + count, 0);

        if (totalProcessors === 0) {
            alert("Please select at least one processor");
            return;
        }

        const selectedProcessors = Object.entries(processorCounts)
            .filter(([_, count]) => count > 0)
            .flatMap(([type, count]) => Array(count).fill(type));

        const stepProps = {
            k: totalProcessors,
            processorNames,
            uptreeStep,
            setCurrentStep,
            setUptreeStep,
            setDisplayPhase,
            selectedProcessors,
        };

        console.log('Current step:', currentStep);
        switch (currentStep) {
            case PHASES.OUTPUT_GIST:
                setDisplayPhase(PHASES.OUTPUT_GIST);
                await handleOutputGistStep(stepProps);
                modifyGraph(totalProcessors);
                setCurrentStep(PHASES.FUSE_GIST);
                break;

            case PHASES.FUSE_GIST:
                setDisplayPhase(PHASES.FUSE_GIST);
                await handleFuseGistStep(stepProps);
                modifyGraph(totalProcessors);
                setCurrentStep(PHASES.UPTREE);
                break

            case PHASES.UPTREE:
                setDisplayPhase(PHASES.UPTREE);
                await handleUptreeStep(stepProps);
                modifyGraph(totalProcessors);

                if (uptreeStep >= totalProcessors - 1) {
                    setCurrentStep(PHASES.FINAL_NODE);
                    setUptreeStep(1);
                } else {
                    setUptreeStep((prev) => prev + 1);
                }
                break;

            case PHASES.FINAL_NODE:
                setDisplayPhase(PHASES.FINAL_NODE);
                await handleFinalNodeStep(stepProps);
                modifyGraph(totalProcessors);
                setCurrentStep(PHASES.REVERSE);
                break;

            case PHASES.REVERSE:
                setDisplayPhase(PHASES.REVERSE);
                await handleReverseStep(stepProps);
                modifyGraph(totalProcessors);
                setCurrentStep(PHASES.UPDATE);
                break;

            case PHASES.UPDATE:
                setDisplayPhase(PHASES.UPDATE);
                await handleUpdateStep({
                    processorCounts,
                    ...stepProps,
                });
                modifyGraph(totalProcessors);
                setCurrentStep(PHASES.OUTPUT_GIST);
                break;

            default:
                console.error('Unknown step phase');
        }
    };


    const handleStart = async () => {
    const totalProcessors = Object.values(processorCounts).reduce((sum, count) => sum + count, 0);

    if (totalProcessors === 0) {
        alert("Please select at least one processor");
        return;
    }

    const selectedProcessors = [];
    Object.entries(processorCounts).forEach(([type, count]) => {
        for (let i = 0; i < count; i++) {
            selectedProcessors.push(type);
        }
    });

    const stepProps = {
        k: totalProcessors,
        setDisplayPhase,
        setCurrentStep,
        setProcessorNames,
        selectedProcessors,
    };

    try {
        console.log('Initializing with processors:', selectedProcessors);
        const processorNames = await handleInitialStep(stepProps);

        if (processorNames) {
            console.log('Received processor names:', processorNames);
            const initialElements = addProcessorNodes(totalProcessors, processorNames);
            setElements(initialElements.nodes);
            setInitialized(true);
            setCurrentStep(PHASES.OUTPUT_GIST);
        } else {
            throw new Error('Failed to initialize processor names');
        }
    } catch (error) {
        console.error('Initialization failed:', error);
        alert('Failed to initialize processors: ' + error.message);
        setInitialized(false);
    }
};


    useEffect(() => {
        if (!selectedNode) {
            setNodeDetailText('');
            return;
        }

        fetch(`http://localhost:5000/api/nodes/${selectedNode}`)
        .then((response) => response.json())
        .then((data) => {
            if (data.self) {
            let detailsText = `Node Details:\n${data.self}`;
            if (data.parents && Object.keys(data.parents).length > 0) {
                detailsText += `\n\nParent Details:\n`;
                Object.entries(data.parents).forEach(([parentId, parentDetails]) => {
                detailsText += `Parent ${parentId}: ${parentDetails}\n`;
                });
            } else {
                detailsText += `\n\nNo parent details available.`;
            }
            setNodeDetailText(detailsText);
            } else {
            setNodeDetailText('No details found for this node.');
            }
        })
        .catch((error) => {
            console.error('Error fetching node details:', error);
            setNodeDetailText('Error loading node details.');
        });
    }, [selectedNode]);

    return (
        <div className="app-container">
            <h1>CTM-AI</h1>
            
            {/* Upload Form */}
            <UploadForm />

            <div className="controls-container">
                <ProcessorSelector
                    processors={processorTypes}
                    counts={processorCounts}
                    onCountChange={handleProcessorCountChange}
                />
                <button onClick={handleStart}>Start</button>
            </div>

            {initialized ? (
                <div className="visualization-container">
                    <div className="cytoscape-container">
                        <CytoscapeComponent
                            elements={elements}
                            layout={layout}
                            stylesheet={stylesheet}
                            style={{ width: '100%', height: '100%' }}
                            cy={(cy) => {
                                cy.on('tap', 'node', (evt) => {
                                    setSelectedNode(evt.target.id());
                                });
                            }}
                        />
                    </div>
                    <div className="info-panel">
                        <h2>Node Information</h2>
                        {selectedNode ? (
                            <pre className="node-details">{nodeDetailText}</pre>
                        ) : (
                            <p>Click a node to see details.</p>
                        )}
                        <hr />
                        <button onClick={handleStep}>Step</button>
                        <p>Current Move: {PHASE_DESCRIPTIONS[displayPhase]}</p>
                    </div>
                </div>
            ) : (
                <p>Please enter k and click "Start" to begin.</p>
            )}
        </div>
    );

};

export default App;
