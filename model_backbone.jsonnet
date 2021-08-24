local config = import 'config.jsonnet';

{
  type: 'basic',


  text_field_embedder: {
    token_embedders: {
      tokens: {
        type: 'pretrained_transformer',
        model_name: config.model_name,
      },
    },
  },

  utterance_pooler: {
    type: 'max_pooler',
    embedding_dim: config.embedding_dim.text,
    dropout: 0.1,
  },

  speaker_embedder: {
    type: 'embedding',
    embedding_dim: config.embedding_dim.speaker,
    vocab_namespace: config.vocab_namespace.speaker,
  },

  supply_embedder: {
    type: 'subsequence',
    embedding_dim: config.embedding_dim.supply,
    subsequence_length: config.num_issues,
    num_embeddings: config.max_issue_quantity,
  },

  context_encoder: {
    type: 'context',
    embedding_dim: config.embedding_dim.dialogue,
    dropout: 0.1,
    context_flags: {
      speakers: true,
      supply: true,
    },
  },

  dialogue_encoder: {
    type: 'pytorch_transformer',
    input_dim: config.embedding_dim.dialogue,
    num_layers: 3,
    positional_encoding: 'sinusoidal',
  },

}
