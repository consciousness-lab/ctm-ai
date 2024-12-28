import React, { useState, useEffect } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';

const PHASES = {
  INIT: 0,
  OUTPUT_GIST: 1,
  UPTREE: 2,
  FINAL_NODE: 3,
  REVERSE: 4,
  UPDATE: 5
};

const PHASE_DESCRIPTIONS = {
  [PHASES.INIT]: "Prepare processors...",
  [PHASES.OUTPUT_GIST]: "Output gist from processors...",
  [PHASES.UPTREE]: "Up-tree sampling...",
  [PHASES.FINAL_NODE]: "Generating final output...",
  [PHASES.REVERSE]: "Down-tree broadcasting...",
  [PHASES.UPDATE]: "Update processors..."
};

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

  // Styles and layouts
  const layout = { name: 'preset', directed: true, padding: 10 };
  const stylesheet = [
    {
      selector: 'node',
      style: {
        content: 'data(label)',
        'text-valign': 'center',
        'background-color': '#61bffc',
        width: 45,
        height: 45,
      }
    },
    {
      selector: 'node.rectangle',
      style: {
        shape: 'rectangle',
        width: 60,
        height: 40,
        'background-color': '#9c27b0',
      }
    },
    {
      selector: 'node.bottom-layer',
      style: {
        'background-color': '#4CAF50',
      }
    },
    {
      selector: 'node.final-node',
      style: {
        'background-color': '#FF5722',
      }
    },
    {
      selector: 'edge',
      style: {
        'curve-style': 'bezier',
        width: 3,
        'target-arrow-shape': 'triangle',
        'line-color': '#ddd',
        'target-arrow-color': '#ddd',
      }
    }
  ];

  // Build initial rectangular nodes
  const buildInitialElements = (kVal) => {
    const nodes = [];
    const spacing = 100;
    const startX = 400 - ((kVal - 1) * spacing) / 2;
    const startY = 500;

    for (let i = 0; i < kVal; i++) {
      const initNodeId = `init${i + 1}`;
      nodes.push({
        data: { id: initNodeId, label: `P${i + 1}` },
        position: { x: startX + i * spacing, y: startY },
        classes: 'rectangle'
      });
    }

    return { nodes, edges: [] };
  };

  // Build all layers of the pyramid
  const buildAllLayers = (kVal) => {
    let layersData = [];
    let layerNodeIds = [];
    let currentNodeId = 1;
    let nodePositions = new Map();

    // Generate node IDs for each layer
    for (let layerIndex = 0; layerIndex < kVal; layerIndex++) {
      const numNodes = kVal - layerIndex;
      let nodeIdsThisLayer = [];
      for (let j = 0; j < numNodes; j++) {
        nodeIdsThisLayer.push(`n${currentNodeId}`);
        currentNodeId++;
      }
      layerNodeIds.push(nodeIdsThisLayer);
    }

    const nodeSpacing = 100;

    // Build each layer
    for (let layerIndex = 0; layerIndex < kVal; layerIndex++) {
      const nodeIds = layerNodeIds[layerIndex];
      let layerNodes = [];
      let layerEdges = [];
      const yPos = 400 - layerIndex * 100;
      const nodesInThisLayer = nodeIds.length;
      const layerWidth = (nodesInThisLayer - 1) * nodeSpacing;
      const startX = 400 - (layerWidth / 2);

      // Create nodes for this layer
      nodeIds.forEach((nodeId, idx) => {
        const xPos = startX + idx * nodeSpacing;
        nodePositions.set(nodeId, { x: xPos, y: yPos });

        layerNodes.push({
          data: { id: nodeId, label: nodeId },
          position: { x: xPos, y: yPos },
          classes: layerIndex === 0 ? 'bottom-layer' : ''
        });
      });

      // Create edges between layers
      if (layerIndex > 0) {
        const belowIds = layerNodeIds[layerIndex - 1];
        nodeIds.forEach((targetId, targetIdx) => {
          if (targetIdx === 0) {
            // First node in layer
            layerEdges.push({
              data: { source: belowIds[0], target: targetId, id: `e${belowIds[0]}-${targetId}` }
            });
            layerEdges.push({
              data: { source: belowIds[1], target: targetId, id: `e${belowIds[1]}-${targetId}` }
            });
          } else if (targetIdx === nodeIds.length - 1) {
            // Last node in layer
            layerEdges.push({
              data: { source: belowIds[belowIds.length - 2], target: targetId, id: `e${belowIds[belowIds.length - 2]}-${targetId}` }
            });
            layerEdges.push({
              data: { source: belowIds[belowIds.length - 1], target: targetId, id: `e${belowIds[belowIds.length - 1]}-${targetId}` }
            });
          } else {
            // Middle nodes
            layerEdges.push({
              data: { source: belowIds[targetIdx], target: targetId, id: `e${belowIds[targetIdx]}-${targetId}` }
            });
            layerEdges.push({
              data: { source: belowIds[targetIdx + 1], target: targetId, id: `e${belowIds[targetIdx + 1]}-${targetId}` }
            });
          }
        });
      }
      
      layersData.push({ nodes: layerNodes, edges: layerEdges });
    }

    // Add final output node
    const topNodeId = layerNodeIds[kVal - 1][0];
    const topNodePos = nodePositions.get(topNodeId);
    const finalNodeId = `n${currentNodeId}`;

    const finalLayer = {
      nodes: [{
        data: { id: finalNodeId, label: finalNodeId },
        position: { x: topNodePos.x, y: topNodePos.y - 100 },
        classes: 'final-node'
      }],
      edges: [{
        data: { source: topNodeId, target: finalNodeId, id: `e${topNodeId}-${finalNodeId}` }
      }]
    };
    layersData.push(finalLayer);

    // Add initial rectangular nodes
    const initElements = buildInitialElements(kVal);
    layersData.unshift({ nodes: initElements.nodes, edges: initElements.edges });

    return layersData;
  };

  // Helper functions
  const updateNodeParents = async (parentUpdates) => {
    try {
      const response = await fetch('http://localhost:5000/api/update-node-parents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ updates: parentUpdates }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Updated node parents:', data);
    } catch (error) {
      console.error('Error updating node parents:', error);
    }
  };

  // Update all the step handlers to set the display phase
  const handleInitialStep = () => {
    setDisplayPhase(PHASES.INIT);
    setCurrentStep(PHASES.OUTPUT_GIST);
  };

  const handleOutputGistStep = () => {
    setDisplayPhase(PHASES.OUTPUT_GIST);
    const nextLayer = pyramidLayers[1];
    const initToBottomEdges = [];
    const parentUpdates = [];

    for (let i = 0; i < k; i++) {
      const initNodeId = `init${i + 1}`;
      const bottomNodeId = `n${i + 1}`;
      
      initToBottomEdges.push({
        data: {
          source: initNodeId,
          target: bottomNodeId,
          id: `e${initNodeId}-${bottomNodeId}`,
        },
      });

      parentUpdates.push({
        node_id: bottomNodeId,
        parents: [initNodeId],
      });
    }

    updateNodeParents(parentUpdates);
    setElements(prev => [...prev, ...nextLayer.nodes, ...nextLayer.edges, ...initToBottomEdges]);
    setCurrentLayerIndex(2);
    setCurrentStep(PHASES.UPTREE);
  };

  const handleUptreeStep = () => {
    setDisplayPhase(PHASES.UPTREE);
    const nextLayer = pyramidLayers[currentLayerIndex];
    const parentUpdates = [];

    nextLayer.edges.forEach(edge => {
      const { source, target } = edge.data;
      parentUpdates.push({
        node_id: target,
        parents: [source],
      });
    });

    updateNodeParents(parentUpdates);
    setElements(prev => [...prev, ...nextLayer.nodes, ...nextLayer.edges]);
    setCurrentLayerIndex(prev => prev + 1);
    
    if (uptreeStep >= k - 1) {
      setCurrentStep(PHASES.FINAL_NODE);
      setUptreeStep(1);
    } else {
      setUptreeStep(prev => prev + 1);
    }
  };

  const handleFinalNodeStep = () => {
    setDisplayPhase(PHASES.FINAL_NODE);
    const finalLayer = pyramidLayers[k + 1];
    const finalNode = finalLayer.nodes[0].data.id;
    const parentNodes = pyramidLayers[k].nodes.map(node => node.data.id);

    updateNodeParents([{
      node_id: finalNode,
      parents: parentNodes,
    }]);

    setElements(prev => [...prev, ...finalLayer.nodes, ...finalLayer.edges]);
    setCurrentStep(PHASES.REVERSE);
  };

  const handleReverseStep = () => {
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

  const handleUpdateStep = () => {
    setDisplayPhase(PHASES.UPDATE);
    setElements(prev => prev.filter(el => el.data?.id?.startsWith('init')));
    setCurrentStep(PHASES.OUTPUT_GIST);
    setUptreeStep(1);
  };

  // Update initialization
  const handleInitialize = () => {
    const allLayers = buildAllLayers(k);
    setPyramidLayers(allLayers);
    setElements([...allLayers[0].nodes]);
    setCurrentLayerIndex(1);
    setDisplayPhase(PHASES.INIT);
    setCurrentStep(PHASES.OUTPUT_GIST);
    setUptreeStep(1);
    setInitialized(true);
  };


  const handleStep = () => {
    switch (currentStep) {
      case PHASES.INIT:
        handleInitialStep();
        break;
      case PHASES.OUTPUT_GIST:
        handleOutputGistStep();
        break;
      case PHASES.UPTREE:
        handleUptreeStep();
        break;
      case PHASES.FINAL_NODE:
        handleFinalNodeStep();
        break;
      case PHASES.REVERSE:
        handleReverseStep();
        break;
      case PHASES.UPDATE:
        handleUpdateStep();
        break;
      default:
        console.error('Unknown step phase');
    }
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
    <div style={{ margin: '20px' }}>
      <h1>CTM-AI</h1>
      <div style={{ marginBottom: '10px' }}>
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
        <div style={{ display: 'flex' }}>
          <div style={{ width: '800px', height: '600px' }}>
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
          <div style={{ marginLeft: '20px', width: '300px' }}>
            <h2>Node Information</h2>
            {selectedNode ? (
              <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
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