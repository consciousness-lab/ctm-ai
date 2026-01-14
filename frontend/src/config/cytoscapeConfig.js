// src/config/cytoscapeConfig.js
export const layout = {
  name: 'preset',
  directed: true,
  padding: 50,
  fit: true,
  animate: false
};

export const stylesheet = [
  {
    // Default node style (for fused nodes)
    selector: 'node',
    style: {
      content: 'data(label)',
      'text-valign': 'center',
      'text-halign': 'center',
      'background-color': '#667eea',
      'background-opacity': 0.9,
      width: 50,
      height: 50,
      'font-size': '11px',
      'text-wrap': 'wrap',
      'text-max-width': '80px',
      'font-weight': 600,
      color: '#ffffff',
      'text-outline-width': 0,
      'border-width': 2,
      'border-color': 'rgba(255, 255, 255, 0.3)',
      'box-shadow': '0 4px 15px rgba(102, 126, 234, 0.4)',
    }
  },
  {
    // Processor nodes
    selector: 'node.rectangle',
    style: {
      shape: 'round-rectangle',
      width: 100,
      height: 40,
      'background-color': '#764ba2',
      'background-opacity': 0.95,
      'text-valign': 'center',
      'text-halign': 'center',
      color: '#ffffff',
      'font-size': '10px',
      'text-wrap': 'wrap',
      'text-max-width': '95px',
      'text-outline-width': 0,
      'border-width': 2,
      'border-color': 'rgba(255, 255, 255, 0.25)',
    }
  },
  {
    // Fused nodes (combined gist + fused layer)
    selector: 'node.fused-layer',
    style: {
      'background-color': '#11998e',
      'background-opacity': 0.9,
      width: 50,
      height: 50,
      color: '#ffffff',
      'font-size': '11px',
      'text-outline-width': 0,
      'border-width': 2,
      'border-color': 'rgba(255, 255, 255, 0.3)',
    }
  },
  {
    // Output/Final node
    selector: 'node.output-node',
    style: {
      'background-color': '#f093fb',
      'background-opacity': 0.95,
      shape: 'diamond',
      width: 60,
      height: 60,
      color: '#ffffff',
      'font-size': '14px',
      'font-weight': 'bold',
      'text-outline-width': 0,
      'border-width': 3,
      'border-color': 'rgba(255, 255, 255, 0.4)',
    }
  },
  {
    // Edges
    selector: 'edge',
    style: {
      'curve-style': 'bezier',
      width: 2,
      'target-arrow-shape': 'triangle',
      'line-color': 'rgba(255, 255, 255, 0.35)',
      'target-arrow-color': 'rgba(255, 255, 255, 0.35)',
      'arrow-scale': 1.2,
      'target-distance-from-node': 5,
    }
  },
  {
    // Processor-to-processor edges
    selector: 'edge[source *= "Processor"], edge[target *= "Processor"]',
    style: {
      'line-color': 'rgba(118, 75, 162, 0.6)',
      'target-arrow-color': 'rgba(118, 75, 162, 0.6)',
      width: 2.5,
    }
  },
  {
    // Hover state for nodes
    selector: 'node:selected',
    style: {
      'border-width': 3,
      'border-color': '#f093fb',
      'background-opacity': 1,
    }
  }
];
