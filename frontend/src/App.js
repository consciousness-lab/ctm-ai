import React, { useState, useEffect } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';

const App = () => {
  const [k, setK] = useState(3);
  const [pyramidLayers, setPyramidLayers] = useState([]);
  const [elements, setElements] = useState([]);
  const [currentLayerIndex, setCurrentLayerIndex] = useState(0);
  const [currentPhase, setCurrentPhase] = useState('building');
  const [initialized, setInitialized] = useState(false);
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeDetailText, setNodeDetailText] = useState('');
  const [totalSteps, setTotalSteps] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);

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
      },
    },
    {
      selector: 'node.rectangle',
      style: {
        shape: 'rectangle',
        width: 60,
        height: 40,
        'background-color': '#9c27b0',
      },
    },
    {
      selector: 'node.bottom-layer',
      style: {
        'background-color': '#4CAF50',
      },
    },
    {
      selector: 'node.final-node',
      style: {
        'background-color': '#FF5722',
      },
    },
    {
      selector: 'edge',
      style: {
        'curve-style': 'bezier',
        width: 3,
        'target-arrow-shape': 'triangle',
        'line-color': '#ddd',
        'target-arrow-color': '#ddd',
      },
    },
  ];

  const buildInitialElements = (kVal) => {
    const nodes = [];
    const edges = [];
    const spacing = 100;
    const startX = 400 - ((kVal - 1) * spacing) / 2;
    const startY = 500;

    // Create rectangular init nodes
    for (let i = 0; i < kVal; i++) {
      const initNodeId = `init${i + 1}`;
      nodes.push({
        data: { id: initNodeId, label: `Init ${i + 1}` },
        position: { x: startX + i * spacing, y: startY },
        classes: 'rectangle'
      });
    }

    return { nodes, edges };
  };

  const buildAllLayers = (kVal) => {
    let layersData = [];
    let layerNodeIds = [];
    let currentNodeId = 1;
    let nodePositions = new Map();

    // Build node IDs for each layer
    for (let layerIndex = 0; layerIndex < kVal; layerIndex++) {
      const numNodes = kVal - layerIndex;
      let nodeIdsThisLayer = [];
      for (let j = 0; j < numNodes; j++) {
        nodeIdsThisLayer.push(`n${currentNodeId}`);
        currentNodeId++;
      }
      layerNodeIds.push(nodeIdsThisLayer);
    }

    const maxNodesInLayer = kVal;
    const nodeSpacing = 100;
    const totalWidth = (maxNodesInLayer - 1) * nodeSpacing;
    const baseX = 400 - (totalWidth / 2);

    // Build layers
    for (let layerIndex = 0; layerIndex < kVal; layerIndex++) {
      const nodeIds = layerNodeIds[layerIndex];
      let layerNodes = [];
      let layerEdges = [];
      const yPos = 400 - layerIndex * 100;
      const nodesInThisLayer = nodeIds.length;
      const layerWidth = (nodesInThisLayer - 1) * nodeSpacing;
      const startX = 400 - (layerWidth / 2);

      nodeIds.forEach((nodeId, idx) => {
        const xPos = startX + idx * nodeSpacing;
        nodePositions.set(nodeId, { x: xPos, y: yPos });

        const classes = layerIndex === 0 ? 'node' : '';

        layerNodes.push({
          data: { id: nodeId, label: nodeId },
          position: { x: xPos, y: yPos },
          classes: classes
        });
      });

      if (layerIndex > 0) {
        const belowIds = layerNodeIds[layerIndex - 1];
        nodeIds.forEach((targetId, targetIdx) => {
          if (targetIdx === 0) {
            layerEdges.push({
              data: { source: belowIds[0], target: targetId, id: `e${belowIds[0]}-${targetId}` },
            });
            layerEdges.push({
              data: { source: belowIds[1], target: targetId, id: `e${belowIds[1]}-${targetId}` },
            });
          } else if (targetIdx === nodeIds.length - 1) {
            layerEdges.push({
              data: { source: belowIds[belowIds.length - 2], target: targetId, id: `e${belowIds[belowIds.length - 2]}-${targetId}` },
            });
            layerEdges.push({
              data: { source: belowIds[belowIds.length - 1], target: targetId, id: `e${belowIds[belowIds.length - 1]}-${targetId}` },
            });
          } else {
            layerEdges.push({
              data: { source: belowIds[targetIdx], target: targetId, id: `e${belowIds[targetIdx]}-${targetId}` },
            });
            layerEdges.push({
              data: { source: belowIds[targetIdx + 1], target: targetId, id: `e${belowIds[targetIdx + 1]}-${targetId}` },
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

  const handleInitialize = () => {
    const allLayers = buildAllLayers(k);
    setPyramidLayers(allLayers);
    setElements([...allLayers[0].nodes]);  // Start with init nodes
    setCurrentLayerIndex(1);
    setCurrentPhase('building');
    setCurrentStep(0);
    setTotalSteps(k + 4); // +4 for: init, final node, reverse, update
    setInitialized(true);
  };

  const reverseEdges = (elements) => {
    return elements.map(el => {
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
    });
  };

  const getBottomLayerNodes = () => {
    const bottomLayer = pyramidLayers[0];
    return bottomLayer.nodes;
  };

    const handleStep = () => {
    if (currentStep === totalSteps - 1) {
        const nextLayer = pyramidLayers[1];
        setElements([...pyramidLayers[0].nodes, ...nextLayer.nodes, ...nextLayer.edges]);
        setCurrentLayerIndex(2);
        setCurrentStep(1);
    } else if (currentStep === 0) {
        const nextLayer = pyramidLayers[1];

        // Build edges init{i} -> n{i} here:
        const initToBottomEdges = [];
        for (let i = 0; i < k; i++) {
          const initNodeId = `init${i + 1}`;
          const bottomNodeId = `n${i + 1}`;
          initToBottomEdges.push({
            data: {
              source: initNodeId,
              target: bottomNodeId,
              id: `e${initNodeId}-${bottomNodeId}`
            }
          });
        }

        setElements(prev => [...prev, ...nextLayer.nodes, ...nextLayer.edges, ...initToBottomEdges]);
        setCurrentLayerIndex(2);
        setCurrentStep(1);
    } else if (currentStep > 0 && currentStep < k) {
        const nextLayer = pyramidLayers[currentLayerIndex];
        setElements(prev => [...prev, ...nextLayer.nodes, ...nextLayer.edges]);
        setCurrentLayerIndex(prev => prev + 1);
        setCurrentStep(prev => prev + 1);
    } else if (currentStep === k) {
        const finalLayer = pyramidLayers[k + 1];
        setElements(prev => [...prev, ...finalLayer.nodes, ...finalLayer.edges]);
        setCurrentStep(prev => prev + 1);
    } else if (currentStep === k + 1) {
        setElements(prev => reverseEdges(prev));
        setCurrentStep(prev => prev + 1);
    } else if (currentStep === k + 2) {
        // Keep only initX nodes, remove all others (and all edges)
        setElements((prev) => {
          return prev.filter((el) => {
            // We keep only those whose ID starts with "init"
            return el.data?.id?.startsWith('init');
          });
        });

        // Reset the step to 0 so next click restarts the process
        setCurrentStep(0);
    }
    };


  const getPhaseDescription = () => {
    if (currentStep === 0) return "Prepare processors...";
    if (currentStep > 0 && currentStep <= k) return "Up-tree sampling...";
    if (currentStep === k + 1) return "Generating final output...";
    if (currentStep === k + 2) return "Down-tree broadcasting...";
    return "Update processors...";
  };


  useEffect(() => {
    if (selectedNode) {
      fetch(`http://localhost:5000/api/nodes/${selectedNode}`)
        .then((response) => response.json())
        .then((data) => {
          if (data.details) {
            setNodeDetailText(data.details);
          } else {
            setNodeDetailText('No details found for this node.');
          }
        })
        .catch((error) => {
          console.error('Error fetching node details:', error);
          setNodeDetailText('Error loading node details.');
        });
    } else {
      setNodeDetailText('');
    }
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
        <button onClick={handleInitialize}>start</button>
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
              <p>{nodeDetailText}</p>
            ) : (
              <p>Click a node to see details.</p>
            )}
            <hr />
            <button onClick={handleStep}>step</button>
            <p>Current move: {getPhaseDescription()}</p>
          </div>
        </div>
      ) : (
        <p>Please enter k and click "start" to begin.</p>
      )}
    </div>
  );
};

export default App;
