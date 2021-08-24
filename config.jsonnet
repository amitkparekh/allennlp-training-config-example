local utils = import 'utils.jsonnet';

/* ------------------------------ Debug config ------------------------------ */

local debug_flags = {
  limit_instances: false,
  exclude_validation: false,
  distribute_cpu: false,
  single_core: false,
};

local debug_instance_cap = 10;
local debug_num_cpu = 2;

// `true` if any of the `debug_flags` are true
local is_debug = std.member(std.objectValues(debug_flags), true);

/* ------------------------------- DOND config ------------------------------ */
local dond_config = {
  // Number of issues being negotiated
  num_issues: 3,
  // Maximum quantity per issue
  max_issue_quantity: 10,

  data_paths: {
    dialogue: utils.createDatasetPaths(
      train='data/train.jsonl',
      valid='data/val.jsonl',
      test='data/test.jsonl',
    ),
    turn: utils.createDatasetPaths(
      train='data/train_turn_split.jsonl',
      valid='data/val_turn_split.jsonl',
      test='data/test_turn_split.jsonl',
    ),
    utterance: utils.createDatasetPaths(
      train='data/train_utterance_split.jsonl',
      valid='data/val_utterance_split.jsonl',
      test='data/test_utterance_split.jsonl',
    ),
  },
};

/* ---------------------------- Modelling config ---------------------------- */
local modelling_config = {
  model_name: 'bert-base-uncased',

  vocab_namespace: {
    text: 'text_tags',
    speaker: 'speaker_tags',
    status: 'status_labels',
    supply_text: 'supply_labels',
    outcome_text: 'outcome_labels',
  },

  embedding_dim: {
    text: 768,
    speaker: 8,
    supply: 8,
    output: 8,
    dialogue: self.text + self.speaker + (self.supply * $.num_issues),
  },
};

/* ------------------------------ Optimization ------------------------------ */
local optimization_flags = {
  use_amp: !is_debug,
  use_distributed_data_loader: true,
};


/* ------------------------------ All together ------------------------------ */
dond_config + modelling_config + optimization_flags {
  seed: 876170670,

  gpus: [0, 1, 2, 3],

  effective_batch_size: 400,

  gpu_batch_size: 4,
  num_workers: 10,
  max_instance_in_memory_scaling: 500,

  num_epochs: 20,

  wandb_group: 'WANDB GROUP',
  wandb_name: 'RUN NAME',

  debug_flags: debug_flags,
  debug: is_debug,
  debug_instance_cap: debug_instance_cap,
  debug_num_cpu: debug_num_cpu,
}
