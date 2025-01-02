import React, { useState, useEffect } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import { PHASES, PHASE_DESCRIPTIONS } from './constants';
import { layout, stylesheet } from './config/cytoscapeConfig';
import {
    addProcessorNodes,
    addProcessorEdges,
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
import { fetchProcessorNeighborhoods } from './utils/api';
import './App.css';
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

const App = () => {
    const [k, setK] = useState(3);
    const [elements, setElements] = useState([]);
    const [processorNames, setProcessorNames] = useState([]);
    const [initialized, setInitialized] = useState(false);
    const [selectedNode, setSelectedNode] = useState(null);
    const [nodeDetailText, setNodeDetailText] = useState('');
    const [currentStep, setCurrentStep] = useState(PHASES.INIT);
    const [uptreeStep, setUptreeStep] = useState(1);
    const [displayPhase, setDisplayPhase] = useState(PHASES.INIT);
    const [neighborhoods, setNeighborhoods] = useState(null);


    const modifyGraph = () => {
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
                    nodes: addGistNodes(k).nodes,
                    edges: addGistEdges(k, processorNames).edges,
                };
                updateElementsForPhase(newElements);
                break;
            }

            case PHASES.FUSE_GIST: {
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
                        element.data?.label?.toLowerCase().includes('processor')
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
                    const processorNodes = elements.filter((element) =>
                        element.data?.label?.toLowerCase().includes('processor')
                    );

                    const neighborhoods = await fetchProcessorNeighborhoods();
                    setNeighborhoods(neighborhoods);
                    if (neighborhoods) {
                        const newEdges = addProcessorEdges(neighborhoods);
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
                setDisplayPhase(PHASES.OUTPUT_GIST);
                await handleOutputGistStep(stepProps);
                modifyGraph();
                setCurrentStep(PHASES.FUSE_GIST);
                break;

            case PHASES.FUSE_GIST:
                setDisplayPhase(PHASES.FUSE_GIST);
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
        const stepProps = {
            k,
            setDisplayPhase,
            setProcessorNames,
        };
        const processorNames = await handleInitialStep(stepProps);
        setCurrentStep(PHASES.OUTPUT_GIST);

        if (processorNames) {
            const initialElements = addProcessorNodes(k, processorNames);
            setElements(initialElements.nodes);

            const neighborhoods = await fetchProcessorNeighborhoods();
            setNeighborhoods(neighborhoods);

            if (neighborhoods) {
                const edges = addProcessorEdges(neighborhoods);
                setElements(prev => [...prev, ...edges]);
            }
            setInitialized(true);
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
      <h1 className="app-title">CTM-AI Visualization</h1>

    <div className="upload-section">
        <div className="panel-header">
            <h2 className="panel-title">Upload Files</h2>
        </div>
        <div className="panel-card">
            <UploadForm />
        </div>
    </div>

      <div className="main-grid">
        {/* Left Panel - Process Control */}
        <div className="control-panel">
          <div className="panel-card">
            <h2 className="panel-title">Process Control</h2>
            <div className="control-content">
              <div className="input-group">
                <label>Processor number (k):</label>
                <input
                  type="number"
                  min="1"
                  value={k}
                  onChange={(e) => setK(parseInt(e.target.value, 10))}
                  className="k-input"
                />
              </div>
              <button
                onClick={handleStart}
                disabled={initialized}
                className={`control-button start ${initialized ? 'disabled' : ''}`}
              >
                Start Process
              </button>
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
                  cy={(cy) => {
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
              {selectedNode ? (
                <pre className="node-details">{nodeDetailText}</pre>
              ) : (
                <p className="placeholder-text">Click a node to see details</p>
              )}
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
                <p><strong>Processors:</strong> {processorNames.join(', ')}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;