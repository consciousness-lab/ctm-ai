
// Format processor name for display (e.g., "VideoProcessor_1" -> "Video")
function formatProcessorLabel(processorId) {
    const baseName = processorId.split('_')[0];
    return baseName.replace('Processor', '');
}

export function addProcessorNodes(kVal, processorNames) {
    const nodes = [];
    const count = processorNames && processorNames.length > 0 ? processorNames.length : kVal;
    const spacing = Math.min(100, 800 / (count + 1));
    const startX = 400 - ((count - 1) * spacing) / 2;
    const startY = 500;

    const processorCounts = {};

    for (let i = 0; i < count; i++) {
        const processorId = processorNames?.[i] || `p${i + 1}`;
        console.log(`Creating node: ${processorId}, index: ${i}`);
        const processorType = processorId.split('_')[0];

        if (!processorCounts[processorType]) {
            processorCounts[processorType] = 1;
        } else {
            processorCounts[processorType]++;
        }

        nodes.push({
            data: {
                id: processorId,
                label: formatProcessorLabel(processorId),
                type: processorType
            },
            position: { x: startX + i * spacing, y: startY },
            classes: `rectangle processor-${processorType}`
        });
    }

    return { nodes, edges: [] };
}

export const addProcessorEdges = (neighborhoods, processorNames) => {
  const edges = [];
  const validProcessors = new Set(processorNames || []);
  const addedEdges = new Set(); // Track added edges to avoid duplicates
  
  // Create index map for consistent left-to-right ordering
  const positionIndex = {};
  (processorNames || []).forEach((name, idx) => {
    positionIndex[name] = idx;
  });
  
  console.log('Valid processors:', processorNames);
  console.log('Neighborhoods:', neighborhoods);

  Object.entries(neighborhoods).forEach(([processorId, connectedProcessors]) => {
    // Skip if source processor is not in the valid list
    if (!validProcessors.has(processorId)) {
        console.warn(`Skipping edge from invalid processor: ${processorId}`);
        return;
    }

    connectedProcessors.forEach(targetId => {
      // Skip self-loops (processor connecting to itself)
      if (processorId === targetId) {
          return;
      }

      // Skip if target processor is not in the valid list
      if (!validProcessors.has(targetId)) {
          console.warn(`Skipping edge to invalid processor: ${targetId} (from ${processorId})`);
          return;
      }

      // Sort by position index to ensure consistent left-to-right direction
      // This makes all edges curve in the same direction
      const pair = [processorId, targetId];
      pair.sort((a, b) => (positionIndex[a] ?? 0) - (positionIndex[b] ?? 0));
      const edgeKey = pair.join('-');
      
      // Skip if this edge already exists
      if (addedEdges.has(edgeKey)) {
          return;
      }
      addedEdges.add(edgeKey);


      edges.push({
        data: {
          id: edgeKey,
          source: pair[0],  // 左边的节点作为 source
          target: pair[1],  // 右边的节点作为 target
        },
        classes: 'processor-edge'
      });
    });
  });

  return edges;
};


// Fused nodes - directly connected from processors (simplified from gist + fused)
export function addFusedNodes(kVal) {
    const nodes = [];
    const spacing = 100;
    const startX = 400 - ((kVal - 1) * spacing) / 2;
    const yPosition = 350;

    for (let i = 0; i < kVal; i++) {
        const nodeId = `n${i + 1}`;
        const xPos = startX + i * spacing;
        nodes.push({
            data: { id: nodeId, label: nodeId },
            position: { x: xPos, y: yPosition },
            classes: 'fused-layer'
        });
    }

    return { nodes, edges: [] };
}

// Edges from processors to fused nodes
export const addFusedEdges = (k, processorNames, neighborhoods) => {
    const edges = [];

    // Create edges from processors to fused nodes based on neighborhoods
    for (let i = 0; i < k; i++) {
        for (let j = 0; j < k; j++) {
            const sameIndex = i === j;
            const processorsConnected = neighborhoods &&
                neighborhoods[processorNames[i]]?.includes(processorNames[j]);

            if (sameIndex || processorsConnected) {
                edges.push({
                    data: {
                        id: `${processorNames[i]}-n${j+1}`,
                        source: processorNames[i],
                        target: `n${j+1}`,
                    },
                });
            }
        }
    }

    return { edges };
};


// Helper: calculate nodes count for each layer using ceil(prev/2)
function getLayerNodeCounts(k) {
    const counts = [k]; // layer 0 (fused layer) has k nodes
    let current = k;
    while (current > 1) {
        current = Math.ceil(current / 2);
        counts.push(current);
    }
    return counts;
}

// Helper: get starting node ID for a given layer
function getLayerStartId(k, layerIndex) {
    const counts = getLayerNodeCounts(k);
    let startId = 1;
    for (let i = 0; i < layerIndex; i++) {
        startId += counts[i];
    }
    return startId;
}

export function addUptreeNodes(kVal, layerIndex) {
    const nodes = [];
    const horizontalSpacing = 100;
    const verticalSpacing = 120; // Vertical gap between layers
    
    const counts = getLayerNodeCounts(kVal);
    const nodesInThisLayer = counts[layerIndex] || 0;

    if (nodesInThisLayer <= 0) return { nodes: [], edges: [] };

    const startX = 400 - ((nodesInThisLayer - 1) * horizontalSpacing) / 2;
    // Fused layer is at y=350, uptree layers go up from y=230
    const yPosition = 230 - ((layerIndex - 1) * verticalSpacing);

    const startId = getLayerStartId(kVal, layerIndex);

    for (let i = 0; i < nodesInThisLayer; i++) {
        const nodeId = `n${startId + i}`;
        nodes.push({
            data: { id: nodeId, label: nodeId },
            position: { x: startX + i * horizontalSpacing, y: yPosition },
        });
    }

    return { nodes, edges: [] };
}


export function addUptreeEdges(kVal, layerIndex) {
    // layerIndex=0 is fused layer (edges created by addFusedEdges)
    // layerIndex=1+ are uptree layers that need edges from previous layer
    if (layerIndex === 0) return { nodes: [], edges: [] };
    const edges = [];
    
    const counts = getLayerNodeCounts(kVal);
    const nodesInThisLayer = counts[layerIndex] || 0;
    const nodesInPrevLayer = counts[layerIndex - 1] || 0;

    if (nodesInThisLayer <= 0 || nodesInPrevLayer <= 0) return { nodes: [], edges: [] };

    const currentLayerStartId = getLayerStartId(kVal, layerIndex);
    const prevLayerStartId = getLayerStartId(kVal, layerIndex - 1);

    // Tournament-style pairing: adjacent pairs compete
    // For 6 nodes: (n1,n2)->n7, (n3,n4)->n8, (n5,n6)->n9
    // For 3 nodes: (n7,n8)->n10, n9 alone->n11
    
    let prevIdx = 0;
    for (let i = 0; i < nodesInThisLayer; i++) {
        const currentNode = `n${currentLayerStartId + i}`;
        
        // First node of the pair
        const node1 = `n${prevLayerStartId + prevIdx}`;
        edges.push({
            data: {
                source: node1,
                target: currentNode,
                id: `e${node1}-${currentNode}`,
            },
        });
        prevIdx++;
        
        // Second node of the pair (if exists)
        if (prevIdx < nodesInPrevLayer) {
            const node2 = `n${prevLayerStartId + prevIdx}`;
            edges.push({
                data: {
                    source: node2,
                    target: currentNode,
                    id: `e${node2}-${currentNode}`,
                },
            });
            prevIdx++;
        }
    }

    return { nodes: [], edges };
}

// Calculate total uptree layers needed
export function calculateTotalUptreeLayers(k) {
    return getLayerNodeCounts(k).length - 1; // exclude fused layer
}

export function addFinalNode(kVal) {
    // Calculate the ID of the top node (last node in the uptree)
    const counts = getLayerNodeCounts(kVal);
    let topNodeId = 0;
    for (let i = 0; i < counts.length; i++) {
        topNodeId += counts[i];
    }
    // topNodeId is now the total count, the last node ID is topNodeId

    const totalLayers = calculateTotalUptreeLayers(kVal);
    // Match the vertical spacing in addUptreeNodes: y = 230 - (layerIndex-1)*120
    const verticalSpacing = 120;
    const topUptreeY = 230 - ((totalLayers - 1) * verticalSpacing);
    const finalNodeY = topUptreeY - 150; // Gap between top uptree node and final node

    return {
        nodes: [{
            data: { id: 'o', label: 'o' },
            position: { x: 400, y: finalNodeY },
            classes: 'output-node'
        }],
        edges: [{
            data: {
                source: `n${topNodeId}`,
                target: 'o',
                id: `en${topNodeId}-o`
            }
        }]
    };
}

export function calculateTotalLayers(k) {
    return calculateTotalUptreeLayers(k);
}
