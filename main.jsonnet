local config = import 'config.jsonnet';
local heads = import 'heads.jsonnet';
local backbone = import 'model_backbone.jsonnet';
local model_base = import 'model_base.jsonnet';
local utils = import 'utils.jsonnet';


local tasks = {

  status: utils.createTask(
    namespace=config.vocab_namespace.status,
    head=heads.ClassifierHead,
    dataset='dialogue',
    dataset_reader='status-classifier',
    enable=true,
    head_extras={},
    reader_extras={},
  ),

  supply_text: utils.createTask(
    namespace=config.vocab_namespace.supply_text,
    head=heads.TokenSubsequenceDecoderHead,
    dataset='dialogue',
    dataset_reader='supply-tokens-sequence',
    enable=true,
    head_extras={
      sequence_length: 3,
      subsequence_length: 3,
      // Full greedy search
      seq_decoder+: {
        scheduled_sampling_ratio: 1,
        decoder_net+: {
          num_layers: 6,
          num_attention_heads: 25,
        },
      },
    },
    reader_extras={
      vocab_namespace_mapping_key: 'supply_text',
    },
  ),

  outcome_text: utils.createTask(
    namespace=config.vocab_namespace.outcome_text,
    head=heads.TokenSubsequenceDecoderHead,
    dataset='dialogue',
    dataset_reader='outcome-tokens-sequence',
    enable=false,
    head_extras={
      sequence_length: 6,
      subsequence_length: 3,
      seq_decoder+: { scheduled_sampling_ratio: 0.5 },
    },
    reader_extras={
      vocab_namespace_mapping_key: 'outcome_text',
    },
  ),

  next_utterance: utils.createTask(
    namespace=config.vocab_namespace.text,
    head=heads.PretrainedTransformerSeqDecoderHead,
    dataset='utterance',
    dataset_reader='next-utterance-sequence',
    enable=true,
    head_extras={
      // Full Teacher forcing
      seq_decoder+: { scheduled_sampling_ratio: 0 },
    },
    reader_extras={},
  ),
};

local model_data_paths = utils.createModelDataPaths(tasks);

model_base + model_data_paths {

  dataset_reader+: {
    readers: utils.createTaskReaders(tasks, self.task_reader),
  },

  data_loader+: {
    [if !config.debug_flags.limit_instances then 'max_instances_in_memory']:
      utils.repeatForTasks(tasks, self.max_instances_per_task),
    [if !config.debug_flags.single_core then 'num_workers']:
      utils.repeatForTasks(tasks, self.num_workers_per_task),
  },

  vocabulary: {
    tokens_to_add: utils.getAdditionalTokensFromTasks(tasks),
  },

  model+: {
    backbone: backbone,
    arg_name_mapping: utils.mapTaskArgNames(tasks),
    heads: utils.createTaskHeads(tasks),
  },
}
