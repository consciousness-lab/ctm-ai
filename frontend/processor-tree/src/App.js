import './App.css';
import React from "react";
import ProcessorTree from "./components/ProcessorTree"; // Import the component

const processorData = [
  {
    name: "Processor 1",
    children: [
      {
        name: "Child Processor 1.1",
        children: [
          { name: "Sub Child Processor 1.1.1" },
          { name: "Sub Child Processor 1.1.2" },
        ],
      },
      { name: "Child Processor 1.2" },
    ],
  },
  {
    name: "Processor 2",
    children: [
      { name: "Child Processor 2.1" },
      { name: "Child Processor 2.2" },
    ],
  },
];

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Processor Tree Viewer</h1>
      </header>
      <main className="App-main">
        <ProcessorTree data={processorData} />
      </main>
    </div>
  );
}

export default App;
