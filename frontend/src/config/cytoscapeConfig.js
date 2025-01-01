// src/config/cytoscapeConfig.js
export const layout = {
  name: 'preset',
  directed: true,
  padding: 20,
  fit: true
};

export const stylesheet = [
  {
    // Default node style (for fused nodes)
    selector: 'node',
    style: {
      content: 'data(label)',
      'text-valign': 'center',
      'text-halign': 'center',
      'background-color': '#3498db',  // Blue color for fused nodes
      width: 45,
      height: 45,
      'font-size': '12px',
      'text-wrap': 'wrap',
      'text-max-width': '80px',
      'font-weight': 500,
      color: '#ffffff',
      'text-outline-width': 0,
    }
  },
  {
    // Processor nodes
    selector: 'node.rectangle',
    style: {
      shape: 'rectangle',
      width: 100,
      height: 35,
      'background-color': '#9b59b6',  // Purple for processors
      'text-valign': 'center',
      'text-halign': 'center',
      color: '#ffffff',
      'font-size': '9px',
      'text-wrap': 'wrap',
      'text-max-width': '90px',
      'text-outline-width': 0,
    }
  },
  {
    // Gist nodes
    selector: 'node.gist-layer',
    style: {
      'background-color': '#2ecc71',  // Green for gist nodes
      width: 45,
      height: 45,
      color: '#ffffff',
      'font-size': '12px',
      'text-outline-width': 0,
    }
  },
  {
    // Output/Final node
    selector: 'node.output-node',
    style: {
      'background-color': '#e74c3c',  // Red for final output node
      shape: 'diamond',
      width: 50,
      height: 50,
      color: '#ffffff',
      'font-size': '14px',
      'font-weight': 'bold',
      'text-outline-width': 0,
    }
  },
  {
    // Edges
    selector: 'edge',
    style: {
      'curve-style': 'bezier',
      width: 2,
      'target-arrow-shape': 'triangle',
      'line-color': '#95a5a6',  // Gray for edges
      'target-arrow-color': '#95a5a6',
      'arrow-scale': 1.2,
      'target-distance-from-node': 3,
    }
  },
  {
    // Processor-to-processor edges
    selector: 'edge[source *= "processor"], edge[target *= "processor"]',
    style: {
      'line-color': '#8e44ad',  // Darker purple for processor edges
      'target-arrow-color': '#8e44ad',
      width: 2.5,
    }
  }
];
