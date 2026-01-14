// constants.js
// New CTM flow: output gist → up-tree → generate → down-tree → update → fuse → back to output gist
export const PHASES = {
  INIT: 0,
  OUTPUT_GIST: 1,    // ask_processors - 处理器输出gist
  UPTREE: 2,         // uptree_competition - 上树竞争
  GENERATE: 3,       // ask_supervisor - 生成最终答案
  DOWNTREE: 4,       // downtree_broadcast - 下树广播
  UPDATE: 5,         // link_form - 更新链接
  FUSE: 6            // fuse_processor - 融合处理器
};

export const PHASE_DESCRIPTIONS = {
  [PHASES.INIT]: "Prepare processors...",
  [PHASES.OUTPUT_GIST]: "Output gist from processors...",
  [PHASES.UPTREE]: "Up-tree competition...",
  [PHASES.GENERATE]: "Generating final answer...",
  [PHASES.DOWNTREE]: "Down-tree broadcasting...",
  [PHASES.UPDATE]: "Updating processor links...",
  [PHASES.FUSE]: "Fusing processors..."
};
