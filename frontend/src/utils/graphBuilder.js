
// Format processor name for display (e.g., "VideoProcessor_1" -> "Video")
function formatProcessorLabel(processorId) {
    const baseName = processorId.split('_')[0];
    return baseName.replace('Processor', '');
}

export function addProcessorNodes(kVal, processorNames) {
    const nodes = [];
    const count = processorNames && processorNames.length > 0 ? processorNames.length : kVal;
    
    // 节点宽度 100px，最小间距 110px 确保不重叠
    const minSpacing = 110;
    const spacing = minSpacing;
    const totalWidth = (count - 1) * spacing;
    const startX = 400 - totalWidth / 2;
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
    const spacing = 110; // 与 processor nodes 保持一致
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

    // Create edges from processors to their corresponding fused nodes only
    // No extra edges based on processor neighborhoods
    for (let i = 0; i < k; i++) {
        edges.push({
            data: {
                id: `${processorNames[i]}-n${i+1}`,
                source: processorNames[i],
                target: `n${i+1}`,
            },
        });
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

// Legacy function - creates new node IDs (deprecated)
export function addUptreeNodes(kVal, layerIndex) {
    const nodes = [];
    const horizontalSpacing = 110;
    const verticalSpacing = 120;
    
    const counts = getLayerNodeCounts(kVal);
    const nodesInThisLayer = counts[layerIndex] || 0;

    if (nodesInThisLayer <= 0) return { nodes: [], edges: [] };

    const startX = 400 - ((nodesInThisLayer - 1) * horizontalSpacing) / 2;
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

// Legacy function - creates edges to new nodes (deprecated)
export function addUptreeEdges(kVal, layerIndex) {
    if (layerIndex === 0) return { nodes: [], edges: [] };
    const edges = [];
    
    const counts = getLayerNodeCounts(kVal);
    const nodesInThisLayer = counts[layerIndex] || 0;
    const nodesInPrevLayer = counts[layerIndex - 1] || 0;

    if (nodesInThisLayer <= 0 || nodesInPrevLayer <= 0) return { nodes: [], edges: [] };

    const currentLayerStartId = getLayerStartId(kVal, layerIndex);
    const prevLayerStartId = getLayerStartId(kVal, layerIndex - 1);

    let prevIdx = 0;
    for (let i = 0; i < nodesInThisLayer; i++) {
        const currentNode = `n${currentLayerStartId + i}`;
        
        const node1 = `n${prevLayerStartId + prevIdx}`;
        edges.push({
            data: {
                source: node1,
                target: currentNode,
                id: `e${node1}-${currentNode}`,
            },
        });
        prevIdx++;
        
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

// NEW: Process uptree with winner inheritance
// competitionResults: array of { winner, loser, parents, graphParents }
//   - winner/loser: semantic IDs (e.g., "n1", "n2") for backend
//   - graphParents: actual node IDs in the graph for edge creation
// Returns: { nodes: [], edges: [] } - creates new nodes at upper layer with winner's label
export function processUptreeWithWinners(kVal, layerIndex, competitionResults) {
    const horizontalSpacing = 110;
    const verticalSpacing = 120;
    
    const counts = getLayerNodeCounts(kVal);
    const nodesInThisLayer = counts[layerIndex] || 0;

    if (nodesInThisLayer <= 0) return { nodes: [], edges: [] };

    const startX = 400 - ((nodesInThisLayer - 1) * horizontalSpacing) / 2;
    const yPosition = 230 - ((layerIndex - 1) * verticalSpacing);

    const nodes = [];
    const edges = [];

    // Create new nodes at upper layer with winner's label
    competitionResults.forEach((result, idx) => {
        const winnerId = result.winner;  // e.g., "n1" - used for label
        // Use graphParents if available, otherwise fall back to parents
        const graphParents = result.graphParents || result.parents || [];
        const xPos = startX + idx * horizontalSpacing;

        // Create new node at upper layer position
        // ID is unique (layer_index + position), but label shows winner's name
        const newNodeId = `layer${layerIndex}_${idx}`;
        const winnerLabel = winnerId; // Display winner's name as label

        nodes.push({
            data: { 
                id: newNodeId, 
                label: winnerLabel,
                winnerId: winnerId,  // Store original winner ID for reference
            },
            position: { x: xPos, y: yPosition },
            classes: 'uptree-node'
        });

        // Create edges from graph parent nodes to this new node
        graphParents.forEach(parentId => {
            edges.push({
                data: {
                    source: parentId,
                    target: newNodeId,
                    id: `e${parentId}-${newNodeId}`,
                },
                classes: 'uptree-edge'
            });
        });
    });

    return { nodes, edges };
}

// Get the node IDs for the next layer's competition
// Returns array of new node IDs created in this layer
export function getActiveNodesAfterLayer(layerIndex, competitionResults) {
    return competitionResults.map((_, idx) => `layer${layerIndex}_${idx}`);
}

// Calculate total uptree layers needed
export function calculateTotalUptreeLayers(k) {
    return getLayerNodeCounts(k).length - 1; // exclude fused layer
}

// topWinnerId: the final winner's original ID (e.g., "n1") for label display
// The edge connects from the last uptree layer node to the output node
export function addFinalNode(kVal, topWinnerId = null) {
    const totalLayers = calculateTotalUptreeLayers(kVal);
    const verticalSpacing = 120;
    const topUptreeY = 230 - ((totalLayers - 1) * verticalSpacing);
    const finalNodeY = topUptreeY - 150;

    // The last uptree node is at layer (totalLayers), position 0
    // Its ID is `layer{totalLayers}_0`
    const lastUptreeNodeId = `layer${totalLayers}_0`;
    
    // Display winner's name as label if provided
    const outputLabel = topWinnerId ? `o (${topWinnerId})` : 'o';

    return {
        nodes: [{
            data: { id: 'o', label: outputLabel },
            position: { x: 400, y: finalNodeY },
            classes: 'output-node'
        }],
        edges: [{
            data: {
                source: lastUptreeNodeId,
                target: 'o',
                id: `e${lastUptreeNodeId}-o`
            }
        }]
    };
}

export function calculateTotalLayers(k) {
    return calculateTotalUptreeLayers(k);
}
