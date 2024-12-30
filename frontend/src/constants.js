// constants.js
export const PHASES = {
  INIT: 0,
  OUTPUT_GIST: 1,
  FUSE_GIST: 2,
  UPTREE: 3,
  FINAL_NODE: 4,
  REVERSE: 5,
  UPDATE: 6 
};

export const PHASE_DESCRIPTIONS = {
  [PHASES.INIT]: "Prepare processors...",
  [PHASES.OUTPUT_GIST]: "Output gist from processors...",
  [PHASES.FUSE_GIST]: "Fuse gist from processors...",
  [PHASES.UPTREE]: "Up-tree sampling...",
  [PHASES.FINAL_NODE]: "Generating final output...",
  [PHASES.REVERSE]: "Down-tree broadcasting...",
  [PHASES.UPDATE]: "Update processors..."
};
