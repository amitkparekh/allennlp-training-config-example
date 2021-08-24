local config = import 'config.jsonnet';
local data_loader = import 'data_loader.jsonnet';
local dataset_reader = import 'dataset_reader.jsonnet';
local distributed = import 'distributed.jsonnet';
local trainer = import 'trainer.jsonnet';
local utils = import 'utils.jsonnet';


distributed {
  random_seed: config.seed,
  numpy_seed: config.seed,
  pytorch_seed: config.seed,

  data_loader: data_loader,

  trainer: trainer,

  dataset_reader: dataset_reader,

  model: {
    type: 'multitask',
  },
}
