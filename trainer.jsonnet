local config = import 'config.jsonnet';

local wandb_callback = {
  type: 'alternative_wandb',
  entity: 'ENTITY',
  project: 'PROJECT NAME',
  group: config.wandb_group,
  name: config.wandb_name,
  watch_model: false,
  should_log_learning_rate: false,
  should_log_parameter_statistics: true,
  files_to_save: [
    'config.json',
    'out.log',
    'out_worker0.log',
    'out_worker1.log',
    'out_worker2.log',
    'out_worker3.log',
  ],
};


{
  optimizer: {
    type: 'huggingface_adamw',
  },
  num_epochs: config.num_epochs,
  num_gradient_accumulation_steps: std.ceil(
    config.effective_batch_size / config.gpu_batch_size /
    std.max(1, std.length(config.gpus))
  ),

  use_amp: config.use_amp,

  [if !config.debug then 'callbacks']: [
    wandb_callback,
  ],
}
