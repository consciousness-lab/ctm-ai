// utils/graphBuilder.js

export function addProcessorNodes(kVal, processorNames) {
    const nodes = [];
    const spacing = Math.min(100, 800 / (kVal + 1));
    const startX = 400 - ((kVal - 1) * spacing) / 2;
    const startY = 500;

    const processorCounts = {};

    for (let i = 0; i < kVal; i++) {
        const processorId = processorNames?.[i] || `p${i + 1}`;
        const processorType = processorId.split('_')[0];

        if (!processorCounts[processorType]) {
            processorCounts[processorType] = 1;
        } else {
            processorCounts[processorType]++;
        }

        const nodeLabel = `${processorType}_${processorCounts[processorType]}`;
        nodes.push({
            data: { 
                id: processorId,
                label: processorId,
                type: processorType
            },
            position: { x: startX + i * spacing, y: startY },
            classes: `rectangle processor-${processorType}`
        });
    }
    
    return { nodes, edges: [] };
}

// Separate functions for nodes and edges
export function addGistNodes(kVal) {
    const nodes = [];
    const spacing = 100;
    const startX = 400 - ((kVal - 1) * spacing) / 2;
    const yPosition = 400;

    for (let i = 0; i < kVal; i++) {
        const nodeId = `g${i + 1}`;
        const xPos = startX + i * spacing;
        nodes.push({
            data: { id: nodeId, label: nodeId },
            position: { x: xPos, y: yPosition },
            classes: 'gist-layer'
        });
    }

    return { nodes, edges: [] };
}

export function addGistEdges(kVal, processorNames) {
    const edges = [];
    for (let i = 0; i < kVal; i++) {
        edges.push({
            data: {
                source: processorNames[i],
                target: `g${i + 1}`,
                id: `ep${i + 1}-g${i + 1}`
            }
        });
    }
    return { nodes: [], edges };
}

export function addFusedNodes(kVal) {
    const nodes = [];
    const spacing = 100;
    const startX = 400 - ((kVal - 1) * spacing) / 2;
    const yPosition = 300;

    for (let i = 0; i < kVal; i++) {
        const nodeId = `n${i + 1}`;
        const xPos = startX + i * spacing;
        nodes.push({
            data: { id: nodeId, label: nodeId },
            position: { x: xPos, y: yPosition }
        });
    }

    return { nodes, edges: [] };
}

export function addFusedEdges(kVal) {
    const edges = [];
    for (let i = 0; i < kVal; i++) {
        for (let j = 0; j < kVal; j++) {
            edges.push({
                data: {
                    source: `g${i + 1}`,
                    target: `n${j + 1}`,
                    id: `eg${i + 1}-n${j + 1}`,
                },
            });
        }
    }
    return { nodes: [], edges };
}


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
    if (layerIndex == 1) return { nodes: [], edges: [] };
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
    // Calculate the ID of the top node
    let topNodeId = kVal;    // Start after n1,n2,n3
    for (let i = 1; i < kVal; i++) {
        topNodeId += (kVal - i);
    }
    
    return {
        nodes: [{
            data: { id: 'o', label: 'o' },
            position: { x: 400, y: 0 },
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

// Helper function to calculate total number of layers needed for k
export function calculateTotalLayers(k) {
    return k - 1;    // For k=3: 2 layers, for k=4: 3 layers, etc.
}