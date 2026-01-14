import React, { useState, useEffect } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import { PHASES, PHASE_DESCRIPTIONS } from './constants';
import { addProcessorNodes } from './utils/graphBuilder';
import { layout, stylesheet } from './config/cytoscapeConfig';
import {
    addProcessorEdges,
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
import { fetchProcessorNeighborhoods } from './utils/api';
import {parseDetailString} from './utils/parseDetailString'
import ProcessorSelector from "./components/ProcessorSelector";
import UploadForm from "./components/UploadForm";


const ProcessPhase = ({ phase, displayPhase, description }) => {
  const phaseNumber = Number(phase);
  const isActive = phaseNumber === displayPhase;
  return (
    <div className={`phase-item ${isActive ? 'active' : 'inactive'}`}>
      <div className="phase-indicator">
        {phaseNumber}
      </div>
      <div className="phase-content">
        <p className="phase-title">{PHASES[phase]}</p>
        <p className="phase-description">{description}</p>
      </div>
    </div>
  );
};

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const App = () => {
    const [availableProcessors] = useState([
        'VisionProcessor',
        'LanguageProcessor',
        'SearchProcessor',
        'CodeProcessor',
        'AudioProcessor',
        'VideoProcessor'
    ]);
    const [nodeDetailJSX, setNodeDetailJSX] = useState(null);
    const [k, setK] = useState(0);
    const [elements, setElements] = useState([]);
    const [processorNames, setProcessorNames] = useState([]);
    const [initialized, setInitialized] = useState(false);
    const [selectedNode, setSelectedNode] = useState(null);
    const [currentStep, setCurrentStep] = useState(PHASES.INIT);
    const [uptreeStep, setUptreeStep] = useState(1);
    const [displayPhase, setDisplayPhase] = useState(PHASES.INIT);
    const [selectedProcessors, setSelectedProcessors] = useState([]);
    const allProcessors = availableProcessors;
    const [neighborhoods, setNeighborhoods] = useState(null);
    const [uploadKey, setUploadKey] = useState(Date.now());
    const [cyInstance, setCyInstance] = useState(null);

    const modifyGraph = () => {
        const updateElementsForPhase = (newElements) => {
            setElements((prevElements) => [
                ...prevElements,
                ...newElements.nodes,
                ...newElements.edges,
            ]);
        };

        switch (currentStep) {
            case PHASES.OUTPUT_GIST:
            case PHASES.FUSE_GIST: {
                // Combined: create fused nodes with edges directly from processors
                const nodes = addFusedNodes(k).nodes;
                const edges = addFusedEdges(k, processorNames, neighborhoods);
                const newElements = {
                    nodes: nodes,
                    edges: edges?.edges || []
                };
                updateElementsForPhase(newElements);
                break;
            }

            case PHASES.UPTREE: {
                const newElements = {
                    nodes: addUptreeNodes(k, uptreeStep + 1).nodes,
                    edges: addUptreeEdges(k, uptreeStep + 1).edges,
                };
                updateElementsForPhase(newElements);
                break;
            }

            case PHASES.FINAL_NODE: {
                const newElements = addFinalNode(k);
                updateElementsForPhase(newElements);
                break;
            }

            case PHASES.REVERSE: {
                setElements((prevElements) => {
                    const processorNodes = prevElements.filter((element) =>
                        element.data && 
                        element.data.id && 
                        !element.data.source && 
                        processorNames.includes(element.data.id)
                    );
                    const processorIds = new Set(processorNodes.map(node => node.data.id));

                    return prevElements.map((element) => {
                        if (!element.data?.source || !element.data?.target) {
                            return element;
                        }

                        const isBothProcessors = processorIds.has(element.data.source) &&
                                            processorIds.has(element.data.target);

                        if (isBothProcessors) {
                            return element;
                        }

                        return {
                            ...element,
                            data: {
                                ...element.data,
                                source: element.data.target,
                                target: element.data.source,
                            },
                        };
                    });
                });
                break;
            }

            case PHASES.UPDATE: {
                const updateElements = async () => {
                    // Filter nodes that are processors (keep them, remove others like gists/fused nodes)
                    const processorNodes = elements.filter((element) => {
                        // Keep elements that are nodes and have an ID in the processor list
                        return element.data && 
                               element.data.id && 
                               !element.data.source && // ensure it's a node, not an edge
                               processorNames.includes(element.data.id);
                    });

                    const neighborhoods = await fetchProcessorNeighborhoods();
                    setNeighborhoods(neighborhoods);
                    if (neighborhoods) {
                        const newEdges = addProcessorEdges(neighborhoods, processorNames);
                        setElements([...processorNodes, ...newEdges]);
                    } else {
                        setElements(processorNodes);
                    }
                };

                // Execute the update
                updateElements();
                break;
            }

            default:
                console.error('Unknown phase for modifying the graph');
                break;
        }
    };


    const handleStep = async() => {
        const stepProps = {
            k,
            processorNames,
            uptreeStep,
            setCurrentStep,
            setUptreeStep,
            setDisplayPhase,
        };

        console.log('Current step:', currentStep);
        switch (currentStep) {
            case PHASES.OUTPUT_GIST:
            case PHASES.FUSE_GIST:
                // Combined: Output gist + Fuse gist in one step
                setDisplayPhase(PHASES.FUSE_GIST);
                await handleOutputGistStep(stepProps);
                await handleFuseGistStep(stepProps);
                modifyGraph();
                setCurrentStep(PHASES.UPTREE);
                break

            case PHASES.UPTREE:
                setDisplayPhase(PHASES.UPTREE);
                await handleUptreeStep(stepProps);
                modifyGraph();

                if (uptreeStep >= k - 1) {
                    setCurrentStep(PHASES.FINAL_NODE);
                    setUptreeStep(1);
                } else {
                    setUptreeStep((prev) => prev + 1);
                }
                break;

            case PHASES.FINAL_NODE:
                setDisplayPhase(PHASES.FINAL_NODE);
                await handleFinalNodeStep(stepProps);
                modifyGraph();
                setCurrentStep(PHASES.REVERSE);
                break;

            case PHASES.REVERSE:
                setDisplayPhase(PHASES.REVERSE);
                await handleReverseStep(stepProps);
                modifyGraph();
                setCurrentStep(PHASES.UPDATE);
                break;

            case PHASES.UPDATE:
                setDisplayPhase(PHASES.UPDATE);
                await handleUpdateStep(stepProps);
                modifyGraph();
                setCurrentStep(PHASES.OUTPUT_GIST);
                break;

            default:
                console.error('Unknown step phase');
        }
    };


    const handleStart = async () => {
        const dynamicK = selectedProcessors.length;
        setK(dynamicK);

        const stepProps = {
            k: dynamicK,
            setDisplayPhase,
            setCurrentStep,
            setProcessorNames,
            selectedProcessors,
        };
        const namesFromBackend = await handleInitialStep(stepProps);
        setCurrentStep(PHASES.OUTPUT_GIST);

        if (namesFromBackend) {
            console.log('Backend returned names:', namesFromBackend);
            const initialElements = addProcessorNodes(dynamicK, namesFromBackend);
            console.log('Created nodes:', initialElements.nodes);
            let allElements = initialElements.nodes;

            const neighborhoods = await fetchProcessorNeighborhoods();
            setNeighborhoods(neighborhoods);

            if (neighborhoods) {
                console.log('Neighborhoods:', neighborhoods);
                const edges = addProcessorEdges(neighborhoods, namesFromBackend);
                console.log('Created edges:', edges);
                allElements = [...allElements, ...edges];
            }
            
            console.log('All Elements sent to Cytoscape:', allElements);
            setElements(allElements);
            setInitialized(true);
            setProcessorNames(namesFromBackend);
        }
    };

    const handleRefresh = async () => {
        try {
            await fetch(`${BASE_URL}/refresh`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
              },
            });
        } catch (error) {
            console.error("Error calling /api/refresh:", error);
        }

        setSelectedProcessors([]);
        setInitialized(false);
        setElements([]);
        setProcessorNames([]);
        setSelectedNode(null);
        setNodeDetailJSX(null);
        setCurrentStep(PHASES.INIT);
        setDisplayPhase(PHASES.INIT);
        setUptreeStep(1);
        setK(0);
        setNeighborhoods(null);
        setCyInstance(null);

        setUploadKey(Date.now());
    };

    // 当elements变化时，自动居中并适应大小
    useEffect(() => {
        if (cyInstance && elements.length > 0) {
            // 延迟执行以确保布局完成
            setTimeout(() => {
                cyInstance.fit(undefined, 50); // 50px padding
                cyInstance.center();
            }, 100);
        }
    }, [cyInstance, elements]);

    useEffect(() => {
        if (!selectedNode) {
            setNodeDetailJSX(null);
            return;
        }

        fetch(`${BASE_URL}/nodes/${selectedNode}`)
            .then((response) => response.json())
            .then((data) => {
                if (!data.self) {
                    setNodeDetailJSX(<div>No details found for this node.</div>);
                    return;
                }

                // Check if this is a processor node
                if (data.processor_info) {
                    const info = data.processor_info;
                    const linkedCount = info.linked_processors?.length || 0;
                    const memoryCount = (info.memory?.fuse_history?.length || 0) + 
                                       (info.memory?.winner_answer?.length || 0) +
                                       (info.memory?.all_context_history?.length || 0);

                    const processorJSX = (
                        <div className="processor-details">
                            <h3>Processor: {info.type}</h3>
                            
                            <div className="detail-section">
                                <p><span className="detail-label">Model:</span> {info.model}</p>
                            </div>

                            <div className="detail-section">
                                <p className="detail-label">Linked Processors ({linkedCount}):</p>
                                {linkedCount > 0 ? (
                                    <div className="linked-list">
                                        {info.linked_processors.map((p, idx) => (
                                            <span key={idx} className="linked-badge">
                                                {p.split('_')[0].replace('Processor', '')}
                                            </span>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="no-data">No linked processors</p>
                                )}
                            </div>

                            <div className="detail-section">
                                <p className="detail-label">Memory ({memoryCount} entries):</p>
                                {info.memory?.all_context_history?.length > 0 ? (
                                    <div className="memory-list">
                                        {info.memory.all_context_history.slice(-3).map((item, idx) => (
                                            <div key={idx} className="memory-item">
                                                <p><strong>Q:</strong> {item.query?.substring(0, 100)}...</p>
                                                <p><strong>A:</strong> {item.answer?.substring(0, 100)}...</p>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="no-data">No memory entries yet</p>
                                )}
                            </div>
                        </div>
                    );
                    setNodeDetailJSX(processorJSX);
                    return;
                }

                // Regular node details
                const nodeSelfLines = parseDetailString(data.self);

                let parentLines = null;
                if (data.parents && Object.keys(data.parents).length > 0) {
                    parentLines = [];
                    parentLines.push(<h3 key="parents-header">Parent Details:</h3>);

                    Object.entries(data.parents).forEach(([parentId, parentText]) => {
                        parentLines.push(
                            <div key={`title-${parentId}`} className="parent-id">
                                <p><strong>Parent {parentId}:</strong></p>
                            </div>
                        );
                        const parsed = parseDetailString(parentText);
                        parentLines.push(
                            <div key={`content-${parentId}`}>
                                {parsed}
                            </div>
                        );
                    });
                }

                const finalJSX = (
                    <div>
                        <h3>Node Details:</h3>
                        {nodeSelfLines}

                        {parentLines ? parentLines : <p>No parent details available.</p>}
                    </div>
                );

                setNodeDetailJSX(finalJSX);
            })
            .catch((error) => {
                console.error('Error fetching node details:', error);
                setNodeDetailJSX(<div>Error loading node details.</div>);
            });
    }, [selectedNode]);



  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <h1 className="app-title">CTM-AI Visualization</h1>
        <p className="app-subtitle">Conscious Turing Machine - Interactive System Visualization</p>
      </header>

      {/* Instructions Panel */}
      <div className="instructions-panel">
        <div className="panel-card">
          <h2 className="panel-title">Quick Start Guide</h2>
          <div className="info-content">
            <p className="intro-text">
              Welcome to the CTM-AI System Visualization Tool. Follow these steps to explore how the system processes and analyzes data.
            </p>
            <ul className="instruction-list">
              <li>
                <span className="step-number">1</span>
                <span className="step-content">
                  <strong>Input your query</strong> — What you'd like the CTM-AI system to answer
                </span>
              </li>
              <li>
                <span className="step-number">2</span>
                <span className="step-content">
                  <strong>Upload your data</strong> — Support for images, audio, video, and text
                </span>
              </li>
              <li>
                <span className="step-number">3</span>
                <span className="step-content">
                  <strong>Select processors</strong> — Choose the analysis modules to use
                </span>
              </li>
              <li>
                <span className="step-number">4</span>
                <span className="step-content">
                  <strong>Start & Monitor</strong> — Begin visualization and track progress
                </span>
              </li>
              <li>
                <span className="step-number">5</span>
                <span className="step-content">
                  <strong>Explore the graph</strong> — Click nodes to view detailed information
                </span>
              </li>
              <li>
                <span className="step-number">6</span>
                <span className="step-content">
                  <strong>View results</strong> — Find your answer in the final output node
                </span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Upload Section */}
      <div className="upload-section">
        <div className="panel-card">
          <h2 className="panel-title">Data Input</h2>
          <UploadForm key={uploadKey} />
        </div>
      </div>

      <div className="main-grid">
        {/* Left Panel - Process Control */}
        <div className="control-panel">
            <div className="panel-card">
                <h2 className="panel-title">Process Control</h2>

                <div className="control-content">
                    <div className="input-group">
                        <div className="controls-container">
                            <ProcessorSelector
                                allProcessors={allProcessors}
                                selectedProcessors={selectedProcessors}
                                onChange={setSelectedProcessors}
                            />

                            <button
                                onClick={handleRefresh}
                                className="control-button refresh"
                            >
                                Refresh
                            </button>
                            <button onClick={handleStart} disabled={initialized}
                                    className={`control-button start ${initialized ? 'disabled' : ''}`}>
                                Start
                            </button>


                        </div>

                    </div>
                    {initialized && (
                        <button
                            onClick={handleStep}
                            className="control-button step"
                        >
                            Next Step
                        </button>
                    )}
                </div>
            </div>

            <div className="panel-card">
                <h2 className="panel-title">Process Phases</h2>
                <div className="phases-list">
                    {Object.entries(PHASE_DESCRIPTIONS).map(([phase, description]) => (
                <ProcessPhase
                    key={phase}
                    phase={phase}
                    displayPhase={displayPhase}
                    description={description}
                />
                ))}
            </div>
          </div>
        </div>

        {/* Main Visualization */}
        <div className="visualization-panel">
          <div className="panel-card">
            <h2 className="panel-title">CTM Visualization</h2>
            <div className="cytoscape-container">
              {initialized ? (
                <CytoscapeComponent
                  elements={elements}
                  layout={layout}
                  stylesheet={stylesheet}
                  style={{ width: '100%', height: '100%' }}
                  userZoomingEnabled={false}
                  userPanningEnabled={false}
                  boxSelectionEnabled={false}
                  autoungrabify={true}
                  cy={(cy) => {
                    // 保存cy实例以便后续操作
                    if (!cyInstance) {
                      setCyInstance(cy);
                    }
                    cy.on('tap', 'node', (evt) => {
                      setSelectedNode(evt.target.id());
                    });
                  }}
                />
              ) : (
                <div className="placeholder-text">
                  Please enter k and click "Start" to begin visualization
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel - Node Information */}
        <div className="info-panel">
          <div className="panel-card">
            <h2 className="panel-title">Node Information</h2>
            <div className="info-content">
                {nodeDetailJSX ? nodeDetailJSX : <p>Click a node to see details</p>}
            </div>
          </div>

          {initialized && (
            <div className="panel-card">
              <h2 className="panel-title">Current Status</h2>
              <div className="status-content">
                <p><strong>Current Phase:</strong> {displayPhase}</p>
                {currentStep === PHASES.UPTREE && (
                  <p><strong>Uptree Step:</strong> {uptreeStep} of {k-1}</p>
                )}
                <p><strong>Processors:</strong> {processorNames.map(p => p.split('_')[0].replace('Processor', '')).join(', ')}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
