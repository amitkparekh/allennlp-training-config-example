local config = import 'config.jsonnet';

local cpu = {
  cuda_devices: std.repeat([-1], config.debug_num_gpu),
};

local gpu = {
  cuda_devices: config.gpus,
  [if config.use_distributed_data_loader then 'ddp_accelerator']: {
    type: 'torch',
    find_unused_parameters: true,
  },
};

{
  [if
    std.length(config.gpus) > 1
    && !config.debug_flags.distribute_cpu
    && !config.debug_flags.single_core
  then 'distributed']: gpu,

  [if
    config.debug_num_cpu > 1
    && config.debug_flags.distribute_cpu
    && !config.debug_flags.single_core
  then 'distributed']: cpu,
}
