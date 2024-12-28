// src/config/cytoscapeConfig.js

export const layout = {
  name: 'preset',
  directed: true,
  padding: 10
};

export const stylesheet = [
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
