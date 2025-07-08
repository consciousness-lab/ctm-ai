export TOOLBENCH_KEY=""
export OPENAI_KEY=""
export OPENAI_API_KEY=$OPENAI_KEY
python run_ctm.py \
--tool_root_dir data/toolenv/tools/ \
--openai_key $OPENAI_KEY \
--max_observation_length 1024 \
--method ctm \
--input_query_file data/instruction/G1_query.json \
--output_answer_file ./ctm_toolbench \
--toolbench_key $TOOLBENCH_KEY \
--test True
