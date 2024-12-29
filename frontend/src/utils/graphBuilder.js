// utils/graphBuilder.js

export function buildInitialElements(kVal, processorNames) {
  const nodes = [];
  const spacing = 100;
  const startX = 400 - ((kVal - 1) * spacing) / 2;
  const startY = 500;

  for (let i = 0; i < kVal; i++) {
    // Use the actual processor name if available, fall back to p{i+1} if not
    const processorId = processorNames?.[i] || `p${i + 1}`;
    
    nodes.push({
      data: { 
        id: processorId,
        // You can customize the label to show a shorter version if needed
        label: processorId // This will show just the type (e.g., 'gpt4v')
      },
      position: { x: startX + i * spacing, y: startY },
      classes: 'rectangle'
    });
  }
  
  return { nodes, edges: [] };
}

export function buildAllLayers(kVal, processorNames) {
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
  const initElements = buildInitialElements(kVal, processorNames);
  layersData.unshift({ nodes: initElements.nodes, edges: initElements.edges });

  return layersData;
}
