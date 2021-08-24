local config = import 'config.jsonnet';

local task_reader = {
  model_name: config.model_name,
  vocab_namespace_mapping: config.vocab_namespace,

  tokenizer_kwargs: {
    namespace: config.vocab_namespace.text,
  },

  [if config.debug_flags.limit_instances
  then 'max_instances']: config.debug_instance_cap,

  manual_distributed_sharding: true,
  manual_multiprocess_sharding: true,
};


{
  type: 'multitask',
  task_reader:: task_reader,
}
