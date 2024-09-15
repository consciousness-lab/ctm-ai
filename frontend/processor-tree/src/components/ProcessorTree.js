import React from "react";
import TreeView from "react-treeview";
import "react-treeview/react-treeview.css"; // Import the CSS for react-treeview

const ProcessorTree = ({ data }) => {
  return (
    <div>
      {data.map((processor, i) => (
        <TreeNode key={i} node={processor} />
      ))}
    </div>
  );
};

const TreeNode = ({ node }) => {
  const hasChildren = node.children && node.children.length > 0;
  return (
    <TreeView nodeLabel={node.name} defaultCollapsed={false}>
      {hasChildren &&
        node.children.map((child, i) => <TreeNode key={i} node={child} />)}
    </TreeView>
  );
};

export default ProcessorTree;