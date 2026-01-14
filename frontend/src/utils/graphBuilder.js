
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
  
  console.log('Valid processors:', processorNames);
  console.log('Neighborhoods:', neighborhoods);

  Object.entries(neighborhoods).forEach(([processorId, connectedProcessors]) => {
    // Skip if source processor is not in the valid list
    if (!validProcessors.has(processorId)) {
        console.warn(`Skipping edge from invalid processor: ${processorId}`);
        return;
    }

    connectedProcessors.forEach(targetId => {
      // Skip if target processor is not in the valid list
      if (!validProcessors.has(targetId)) {
          console.warn(`Skipping edge to invalid processor: ${targetId} (from ${processorId})`);
          return;
      }

      edges.push({
        data: {
          id: `${processorId}-${targetId}`,
          source: processorId,
          target: targetId,
        },
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


export function addUptreeNodes(kVal, layerIndex) {
    const nodes = [];
    const spacing = 100; // Horizontal spacing between nodes
    const nodesInThisLayer = kVal - layerIndex + 1; // Number of nodes in the current layer

    if (nodesInThisLayer <= 0) return { nodes: [], edges: [] };

    const startX = 400 - ((nodesInThisLayer - 1) * spacing) / 2; // Center alignment
    const yPosition = 300 - ((layerIndex - 1) * 100); // Vertical positioning

    // Calculate the starting node number for this layer
    let startId = 1;
    for (let i = 1; i < layerIndex; i++) {
        startId += (kVal - i + 1); // Increment startId by the number of nodes in previous layers
    }

    for (let i = 0; i < nodesInThisLayer; i++) {
        const nodeId = `n${startId + i}`;
        nodes.push({
            data: { id: nodeId, label: nodeId },
            position: { x: startX + i * spacing, y: yPosition },
        });
    }

    return { nodes, edges: [] };
}


export function addUptreeEdges(kVal, layerIndex) {
    if (layerIndex === 1) return { nodes: [], edges: [] };
    const edges = [];
    const nodesInThisLayer = kVal - layerIndex + 1; // Nodes in the current layer
    const nodesInPrevLayer = kVal - layerIndex + 2; // Nodes in the previous layer

    if (nodesInThisLayer <= 0 || nodesInPrevLayer <= 0) return { nodes: [], edges: [] };

    // Calculate the start IDs for the previous and current layers
    let currentLayerStartId = 1;
    for (let i = 1; i < layerIndex; i++) {
        currentLayerStartId += (kVal - i + 1); // Increment by the number of nodes in each previous layer
    }
    const prevLayerStartId = currentLayerStartId - nodesInPrevLayer; // Start ID for the current layer
    console.log('prevLayerStartId:', prevLayerStartId);
    console.log('currentLayerStartId:', currentLayerStartId);
    console.log('nodesInPrevLayer:', nodesInPrevLayer);
    console.log('nodesInThisLayer:', nodesInThisLayer);
    console.log('kVal:', kVal);
    console.log('layerIndex:', layerIndex);

    // Generate edges from the previous layer to the current layer
    for (let i = 0; i < nodesInThisLayer; i++) {
        const currentNode = `n${currentLayerStartId + i}`;

        // Distribute parent nodes evenly from the previous layer
        const lowerNode1Index = Math.floor(i * (nodesInPrevLayer / nodesInThisLayer));
        const lowerNode2Index = Math.min(lowerNode1Index + 1, nodesInPrevLayer - 1);

        const lowerNode1 = `n${prevLayerStartId + lowerNode1Index}`;
        const lowerNode2 = `n${prevLayerStartId + lowerNode2Index}`;

        // Create edges connecting lower nodes to the current node
        if (lowerNode1) {
            edges.push({
                data: {
                    source: lowerNode1,
                    target: currentNode,
                    id: `e${lowerNode1}-${currentNode}`,
                },
            });
        }

        if (lowerNode2 && lowerNode1 !== lowerNode2) {
            edges.push({
                data: {
                    source: lowerNode2,
                    target: currentNode,
                    id: `e${lowerNode2}-${currentNode}`,
                },
            });
        }
    }

    return { nodes: [], edges };
}

export function addFinalNode(kVal) {
    let topNodeId = 1;
    for (let i = 1; i < kVal; i++) {
        topNodeId += (kVal - i + 1);
    }

    const totalLayers = calculateTotalLayers(kVal);
    const topUptreeY = 300 - ((totalLayers - 1) * 100);
    const finalNodeY = topUptreeY - 200;

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
    return k - 1;
}
