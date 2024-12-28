import React, { useState, useEffect } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';
import { PHASES, PHASE_DESCRIPTIONS } from './constants';
import { buildAllLayers, buildInitialElements } from './utils/graphBuilder';
import { layout, stylesheet } from './config/cytoscapeConfig';
import {
  handleInitialStep,
  handleOutputGistStep,
  handleUptreeStep,
  handleFinalNodeStep,
  handleReverseStep,
  handleUpdateStep
} from './steps/index';
import './App.css';

const App = () => {
  // State declarations
  const [k, setK] = useState(3);
  const [pyramidLayers, setPyramidLayers] = useState([]);
  const [elements, setElements] = useState([]);
  const [currentLayerIndex, setCurrentLayerIndex] = useState(0);
  const [initialized, setInitialized] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeDetailText, setNodeDetailText] = useState('');
  const [currentStep, setCurrentStep] = useState(PHASES.INIT);
  const [uptreeStep, setUptreeStep] = useState(1);
  const [displayPhase, setDisplayPhase] = useState(PHASES.INIT);

  // State declarations

  // Step handler
  const handleStep = () => {
    const stepProps = {
      k,
      pyramidLayers,
      currentLayerIndex,
      uptreeStep,
      setElements,
      setCurrentLayerIndex,
      setCurrentStep,
      setUptreeStep,
      setDisplayPhase
    };

    switch (currentStep) {
      case PHASES.INIT:
        handleInitialStep(stepProps);
        break;
      case PHASES.OUTPUT_GIST:
        handleOutputGistStep(stepProps);
        break;
      case PHASES.UPTREE:
        handleUptreeStep(stepProps);
        break;
      case PHASES.FINAL_NODE:
        handleFinalNodeStep(stepProps);
        break;
      case PHASES.REVERSE:
        handleReverseStep(stepProps);
        break;
      case PHASES.UPDATE:
        handleUpdateStep(stepProps);
        break;
      default:
        console.error('Unknown step phase');
    }
  };

  // Initialization handler
  const handleInitialize = async () => {
    // First call handleInitialStep
    await handleInitialStep({
      k,
      setDisplayPhase,
      setCurrentStep
    });

    // Then proceed with the rest of initialization
    const allLayers = buildAllLayers(k);
    setPyramidLayers(allLayers);
    setElements([...allLayers[0].nodes]);
    setCurrentLayerIndex(1);
    setInitialized(true);
  };
  // Node details effect
  useEffect(() => {
    if (!selectedNode) {
      setNodeDetailText('');
      return;
    }

    fetch(`http://localhost:5000/api/nodes/${selectedNode}`)
      .then(response => response.json())
      .then(data => {
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
      .catch(error => {
        console.error('Error fetching node details:', error);
        setNodeDetailText('Error loading node details.');
      });
  }, [selectedNode]);

  return (
    <div className="app-container">
      <h1>CTM-AI</h1>
      <div className="controls-container">
        <label>
          Processor number (k):
          <input
            type="number"
            min="1"
            value={k}
            onChange={(e) => setK(parseInt(e.target.value, 10))}
          />
        </label>
        <button onClick={handleInitialize}>Start</button>
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
              <pre className="node-details">
                {nodeDetailText}
              </pre>
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