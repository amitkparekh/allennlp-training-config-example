local config = import 'config.jsonnet';


{
  max_instances_per_task::
    config.gpu_batch_size * config.max_instance_in_memory_scaling,
  num_workers_per_task:: config.num_workers,

  type: 'multitask',
  shuffle: true,
  scheduler: {
    batch_size: config.gpu_batch_size,
  },
}
