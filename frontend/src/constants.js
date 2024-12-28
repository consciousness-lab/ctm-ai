// constants.js
export const PHASES = {
  INIT: 0,
  OUTPUT_GIST: 1,
  UPTREE: 2,
  FINAL_NODE: 3,
  REVERSE: 4,
  UPDATE: 5
};

export const PHASE_DESCRIPTIONS = {
  [PHASES.INIT]: "Prepare processors...",
  [PHASES.OUTPUT_GIST]: "Output gist from processors...",
  [PHASES.UPTREE]: "Up-tree sampling...",
  [PHASES.FINAL_NODE]: "Generating final output...",
  [PHASES.REVERSE]: "Down-tree broadcasting...",
  [PHASES.UPDATE]: "Update processors..."
};