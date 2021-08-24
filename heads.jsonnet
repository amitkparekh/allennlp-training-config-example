local config = import 'config.jsonnet';


local FeedForwardHiddenDims(num_layers, multiplier) = [
  std.floor(config.embedding_dim.dialogue / std.pow(multiplier, dim))
  for dim in std.range(1, num_layers)
];

local ClassifierHead(label_namespace) = {
  type: 'extended_classifier',

  arg_name_mapping:: 'label',
  input_dim:: config.embedding_dim.dialogue,

  label_namespace: label_namespace,

  seq2vec_encoder: {
    type: 'pytorch_transformer',
    input_size: $.input_dim,
    num_layers: 3,
    num_attention_heads: 8,
    pooler: {
      type: 'max_pooler',
      embedding_dim: $.input_dim,
      dropout: 0.1,
    },
  },

  feedforward: {
    input_dim: $.seq2vec_encoder.input_size,
    num_layers: 3,
    hidden_dims: FeedForwardHiddenDims(self.num_layers, 2),
    activations: 'gelu',
    dropout: 0.1,
  },
};

local TokenSubsequenceDecoderHead(label_namespace) = {
  type: 'token_subsequence_decoder',

  arg_name_mapping:: 'target_tokens',
  input_dim:: config.embedding_dim.dialogue,
  additional_vocab_tokens:: ['@start@', '@end@'],

  sequence_length:: 1,
  subsequence_length:: 1,

  seq_decoder: {
    sequence_length: $.sequence_length,
    scheduled_sampling_ratio: 0.5,
    target_namespace: label_namespace,

    decoder_net: {
      type: 'pytorch_transformer',
      decoding_dim: $.input_dim,
      feedforward_hidden_dim: 2048,
      num_layers: 6,
      num_attention_heads: 16,
      use_causal_mask: true,
    },
    target_embedder: {
      embedding_dim: $.seq_decoder.decoder_net.decoding_dim,
      vocab_namespace: label_namespace,
    },
    beam_search: {
      beam_size: 5,
    },
  },
};

local PretrainedTransformerSeqDecoderHead(label_namespace) = {
  type: 'pretrained_transformer_seq_decoder',
  model_name: config.model_name,

  arg_name_mapping:: 'target_tokens',
  input_dim:: config.embedding_dim.dialogue,

  target_namespace: label_namespace,


  seq_decoder: {
    scheduled_sampling_ratio: 0.5,
    target_namespace: label_namespace,

    decoder_net: {
      type: 'pytorch_transformer',
      decoding_dim: $.input_dim,
      feedforward_hidden_dim: 2048,
      num_layers: 6,
      num_attention_heads: 16,
      use_causal_mask: true,
    },
    target_embedder: {
      embedding_dim: $.seq_decoder.decoder_net.decoding_dim,
      vocab_namespace: label_namespace,
    },
    beam_search: {
      beam_size: 5,
    },
  },
};


{
  ClassifierHead:: ClassifierHead,
  TokenSubsequenceDecoderHead:: TokenSubsequenceDecoderHead,
  PretrainedTransformerSeqDecoderHead:: PretrainedTransformerSeqDecoderHead,
}
