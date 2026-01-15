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
      'background-opacity': 0.7,
      'text-valign': 'center',
      'text-halign': 'center',
      color: '#ffffff',
      'font-size': '10px',
      'text-wrap': 'wrap',
      'text-max-width': '95px',
      'text-outline-width': 0,
      'border-width': 2,
      'border-color': 'rgba(255, 255, 255, 0.25)',
      'z-index': 1,
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
    // Edges - solid by default
    selector: 'edge',
    style: {
      'curve-style': 'bezier',
      width: 2,
      'line-style': 'solid',
      'target-arrow-shape': 'triangle',
      'line-color': 'rgba(255, 255, 255, 0.5)',
      'target-arrow-color': 'rgba(255, 255, 255, 0.5)',
      'arrow-scale': 1.2,
      'target-distance-from-node': 5,
      'z-index': 999,
    }
  },
  {
    // Processor-to-processor edges - dashed, curved downwards, undirected (no arrow)
    selector: 'edge.processor-edge',
    style: {
      'line-style': 'dashed',
      'line-dash-pattern': [6, 3],
      'line-color': 'rgba(118, 75, 162, 0.8)',
      'target-arrow-shape': 'none',  // 无向边，不显示箭头
      width: 2.5,
      'curve-style': 'unbundled-bezier',
      'source-endpoint': '180deg',   // 从节点底部中心出发 (180deg = 下方，6点钟方向)
      'target-endpoint': '180deg',   // 连入节点底部中心
      'control-point-distances': [80],  // 控制点向下偏移，形成向下的弧线
      'control-point-weights': [0.5],
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
