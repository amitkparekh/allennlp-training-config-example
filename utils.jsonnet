local config = import 'config.jsonnet';

local isTaskEnabled(name, task) = if task.enable then task;

local mapEnabledTasksWithKey(func, tasks) =
  local enabledTasks = std.mapWithKey(isTaskEnabled, tasks);
  local prunedTasks = std.prune(enabledTasks);
  std.mapWithKey(func, prunedTasks);

/**
 * Structure dataset paths for a given dataset.
 *
 * The point is to provide a consistent interface for accessing the files for
 * different datasets.
 *
 * @param {string} train - Path for training data
 * @param {string} valid - Path for validation data
 * @param {string} test - Path for testing data
 */
local createDatasetPaths(train, valid, test) = {
  train: train,
  valid: valid,
  test: test,
};


/**
 * Create structured task to automatically setup other components of the model.
 *
 * @param {string} namespace - Namespace used for the vocab
 * @param {Head} head - Pointer to the multi-task head being used
 * @param {string} dataset - Name of the dataset
 * @param {string} dataset_reader - Name of the dataset reader
 * @param {string} enable - Whether to enable the head or not
 * @param {object} head_extras - Additional fields and values passed to the head
 */
local createTask(
  namespace, head, dataset, dataset_reader, enable, head_extras, reader_extras
      ) = {
  namespace: namespace,
  head: head(namespace) + head_extras,
  dataset: dataset,
  dataset_reader: dataset_reader,
  enable: enable,
  reader_extras: reader_extras,
};

local repeatForTasks(tasks, value) = {
  [task]: value
  for task in std.objectFields(tasks)
  if tasks[task].enable
};

/**
 * Create dataset reader for all tasks
 *
 * @param tasks - Object of Tasks
 * @param common_reader - Common dataset reader for all tasks to inherit from
 */
local createTaskReaders(tasks, common_reader) =
  local taskReader(name, task) = common_reader + tasks[name].reader_extras + {
    type: task.dataset_reader,
  };

  mapEnabledTasksWithKey(taskReader, tasks);

/**
 */
local createTaskDataPaths(tasks, data_type) =
  local taskDataPath(name, task) = config.data_paths[task.dataset][data_type];

  mapEnabledTasksWithKey(taskDataPath, tasks);

/**
 */
local createModelDataPaths(tasks) = {
  train_data_path: createTaskDataPaths(tasks, 'train'),

  [if !config.debug_flags.exclude_validation
  then 'validation_data_path']: createTaskDataPaths(tasks, 'valid'),
};


/**
 */
local createTaskHeads(tasks) =
  local createHead(name, task) = task.head;

  mapEnabledTasksWithKey(createHead, tasks);

/**
 *
 */
local mapTaskArgNames(tasks) =
  local mapArgName(name, task) = {
    // Cannot use `task` because it's a hidden field, but this works?
    [name]: tasks[name].head.arg_name_mapping,
  };

  mapEnabledTasksWithKey(mapArgName, tasks);


local getAdditionalTokensForNamespace(namespace, tasks) =
  // Inner functions for filters
  local doesTaskBelongToNamespace(task) = task.namespace == namespace;
  local doesTaskHaveAdditionalTokens(task) = std.objectHasAll(
    task.head, 'additional_vocab_tokens'
  );
  local isTaskEnabled(task) = task.enable == true;

  // Convert tasks to array as task name is not needed
  local tasksArray = std.objectValues(tasks);

  local enabledTasks = std.filter(isTaskEnabled, tasksArray);

  // Get all tasks that connect to the given namespace
  local tasksForNamespace = std.filter(
    doesTaskBelongToNamespace, enabledTasks
  );

  // Then, get all tasks which require additional tokens
  local tasksWithAdditionalTokens = std.filter(
    doesTaskHaveAdditionalTokens, tasksForNamespace
  );

  // Then, get the tokens for each remaining task
  local tokensPerTask = [
    task.head.additional_vocab_tokens
    for task in tasksWithAdditionalTokens
  ];

  // Then flatten into single array and remove duplicates
  std.set(std.flattenArrays(tokensPerTask));

/**
 *
 */
local getAdditionalTokensFromTasks(tasks) =
  // Get all the namespaces used by the model
  local namespaces = std.set([task.namespace for task in std.objectValues(tasks)]);

  // Get the additional tokens to add for each namespace
  local tokensPerNamespace = {
    [namespace]: getAdditionalTokensForNamespace(namespace, tasks)
    for namespace in namespaces
  };

  // Remove any namespaces that do not have additional tokens
  std.prune(tokensPerNamespace);


/**
 * Get total number of training instances from tasks.
 */
local getNumTrainingInstancesFromTasks(tasks) =
  local datasets = [task.dataset for task in std.objectValues(tasks)];
  local numInstances = [config.data_paths[dataset].num_train_instances for dataset in datasets];

  std.foldl(function(x, y) (x + y), numInstances, 0);


{
  createTask:: createTask,
  createTaskReaders:: createTaskReaders,
  repeatForTasks:: repeatForTasks,
  createModelDataPaths:: createModelDataPaths,
  createTaskHeads:: createTaskHeads,
  mapTaskArgNames:: mapTaskArgNames,
  createDatasetPaths:: createDatasetPaths,
  getAdditionalTokensFromTasks:: getAdditionalTokensFromTasks,
  getNumTrainingInstancesFromTasks:: getNumTrainingInstancesFromTasks,
}
