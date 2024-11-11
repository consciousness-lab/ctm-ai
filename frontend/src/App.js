// src/App.js
import React, { useState } from 'react';
import CytoscapeComponent from 'react-cytoscapejs';

const App = () => {
  // Initialize with n1, n2, and n3
  const initialElements = [
    { data: { id: 'n1', label: 'n1' }, position: { x: 100, y: 400 } },
    { data: { id: 'n2', label: 'n2' }, position: { x: 300, y: 400 } },
    { data: { id: 'n3', label: 'n3' }, position: { x: 500, y: 400 } },
  ];

  const [elements, setElements] = useState(initialElements);
  const [step, setStep] = useState(1);
  const [reverse, setReverse] = useState(false); // Track if edges should be reversed
  const [selectedNode, setSelectedNode] = useState(null); // State for selected node

  // Mapping of node IDs to detailed texts
  const nodeDetails = {
    n1: 'This is node n1.',
    n2: 'This is node n2.',
    n3: 'This is node n3.',
    n4: 'This is node n4.',
    n5: 'This is node n5.',
    n6: 'This is node n6.',
    n7: 'This is node n7.',
  };

  const combineNodes = () => {
    let newElements = [...elements];

    if (!reverse) {
      // Forward steps
      if (step === 1) {
        // Add n4 linked with n1 and n2
        newElements.push(
          { data: { id: 'n4', label: 'n4' }, position: { x: 200, y: 300 } },
          { data: { source: 'n1', target: 'n4' } },
          { data: { source: 'n2', target: 'n4' } }
        );
        setStep(2);
      } else if (step === 2) {
        // Add n5 linked with n2 and n3
        newElements.push(
          { data: { id: 'n5', label: 'n5' }, position: { x: 400, y: 300 } },
          { data: { source: 'n2', target: 'n5' } },
          { data: { source: 'n3', target: 'n5' } }
        );
        setStep(3);
      } else if (step === 3) {
        // Add n6 linked with n4 and n5
        newElements.push(
          { data: { id: 'n6', label: 'n6' }, position: { x: 300, y: 200 } },
          { data: { source: 'n4', target: 'n6' } },
          { data: { source: 'n5', target: 'n6' } }
        );
        setStep(4);
      } else if (step === 4) {
        // Add n7 linked with n6, and prepare for reverse
        newElements.push(
          { data: { id: 'n7', label: 'n7' }, position: { x: 300, y: 100 } },
          { data: { source: 'n6', target: 'n7' } }
        );
        setReverse(true); // Next step will reverse all edges
        setStep(5);
      }
    } else {
      if (step === 5) {
        // Step 1 of reverse: Reverse all edges at once
        newElements = newElements.map((el) => {
          if (el.data.source && el.data.target) {
            return { data: { source: el.data.target, target: el.data.source } };
          }
          return el;
        });
        setStep(6);
      } else if (step === 6) {
        // Remove all nodes added during forward steps
        newElements = initialElements;
        setReverse(false); // Reset reverse state for next loop
        setStep(1); // Reset step counter to start loop again
      }
    }

    setElements(newElements);
  };

  const layout = {
    name: 'preset',
    directed: true,
    padding: 10,
  };

  const style = [
    {
      selector: 'node',
      style: {
        content: 'data(label)',
        'text-valign': 'center',
        'background-color': '#61bffc',
        width: 50,
        height: 50,
      },
    },
    {
      selector: 'edge',
      style: {
        'curve-style': 'bezier',
        width: 4,
        'target-arrow-shape': 'triangle',
        'line-color': '#ddd',
        'target-arrow-color': '#ddd',
      },
    },
  ];

  return (
    <div>
      {/* Main Content */}
      <div style={{ display: 'flex' }}>
        {/* Cytoscape Graph */}
        <CytoscapeComponent
          elements={elements}
          style={{ width: '800px', height: '600px' }}
          layout={layout}
          stylesheet={style}
          cy={(cy) => {
            cy.on('tap', 'node', (evt) => {
              var node = evt.target;
              setSelectedNode(node.id());
            });
          }}
        />

        {/* Right Side Node Information */}
        <div style={{ marginLeft: '20px', width: '300px', border: '1px solid #ccc', padding: '10px' }}>
          <h2>Node Information</h2>
          {selectedNode ? (
            <p>{nodeDetails[selectedNode]}</p>
          ) : (
            <p>Click on a node to see details.</p>
          )}
        </div>
      </div>

      {/* Step Button at the Bottom */}
      <div style={{ textAlign: 'center', marginTop: '10px' }}>
        <button onClick={combineNodes}>Step</button>
      </div>
    </div>
  );
};

export default App;
