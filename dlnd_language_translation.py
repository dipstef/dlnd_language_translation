import tensorflow as tf


def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    eos = source_vocab_to_int['<EOS>']

    source = source_text.split('\n')
    target = target_text.split('\n')

    source = [[source_vocab_to_int[word] for word in sentence.split()] for sentence in source]
    target = [[target_vocab_to_int[word] for word in sentence.split()] + [eos] for sentence in target]

    return source, target


def model_inputs():
    """
    Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences.
    :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """
    input_ = tf.placeholder(tf.int32, shape=(None, None), name='input')
    targets = tf.placeholder(tf.int32, shape=(None, None), name='targets')
    learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')

    keep_prob = tf.placeholder(tf.float32, shape=(), name='keep_prob')

    target_sequence_len = tf.placeholder(tf.int32, shape=(None,), name='target_sequence_length')
    max_target_len = tf.reduce_max(target_sequence_len, name='max_target_len')
    source_sequence_len = tf.placeholder(tf.int32, shape=(None,), name='source_sequence_length')

    return (input_, targets, learning_rate,
            keep_prob,
            target_sequence_len, max_target_len, source_sequence_len)


def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])

    return tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), ending], 1)


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
                   source_sequence_length, source_vocab_size,
                   encoding_embedding_size):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :param source_sequence_length: a list of the lengths of each sequence in the batch
    :param source_vocab_size: vocabulary size of source data
    :param encoding_embedding_size: embedding size of source data
    :return: tuple (RNN output, RNN state)
    """
    enc_embed_input = tf.contrib.layers.embed_sequence(rnn_inputs,
                                                       vocab_size=source_vocab_size,
                                                       embed_dim=encoding_embedding_size)

    def rnn_cell(size):
        return tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2)),
            input_keep_prob=keep_prob
        )

    enc_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell(rnn_size) for _ in range(num_layers)])

    enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell,
                                              inputs=enc_embed_input,
                                              sequence_length=source_sequence_length,
                                              dtype=tf.float32)

    return enc_output, enc_state


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         target_sequence_length, max_summary_length,
                         output_layer, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_summary_length: The length of the longest sequence in the batch
    :param output_layer: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    from tensorflow.contrib import seq2seq

    helper = seq2seq.TrainingHelper(dec_embed_input, sequence_length=target_sequence_length)

    decoder = seq2seq.BasicDecoder(dec_cell,
                                   helper=helper,
                                   initial_state=encoder_state,
                                   output_layer=output_layer)

    output = seq2seq.dynamic_decode(decoder,
                                    impute_finished=True,
                                    maximum_iterations=max_summary_length)[0]
    return output


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param max_target_sequence_length: Maximum length of target sequences
    :param vocab_size: Size of decoder/target vocabulary
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_layer: Function to apply the output layer
    :param batch_size: Batch size
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    from tensorflow.contrib import seq2seq

    start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32),
                           multiples=[batch_size],
                           name='start_tokens')

    helper = seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                           start_tokens=start_tokens,
                                           end_token=end_of_sequence_id)

    decoder = seq2seq.BasicDecoder(dec_cell,
                                   helper=helper,
                                   initial_state=encoder_state,
                                   output_layer=output_layer)

    # Perform dynamic decoding using the decoder
    output = seq2seq.dynamic_decode(decoder,
                                    impute_finished=True,
                                    maximum_iterations=max_target_sequence_length)[0]

    return output


def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size,
                   num_layers, target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, decoding_embedding_size):
    """
    Create decoding layer
    :param dec_input: Decoder input
    :param encoder_state: Encoder state
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_target_sequence_length: Maximum length of target sequences
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param target_vocab_size: Size of target vocabulary
    :param batch_size: The size of the batch
    :param keep_prob: Dropout keep probability
    :param decoding_embedding_size: Decoding embedding size
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    from tensorflow.python.layers.core import Dense

    dec_embeddings = tf.Variable(tf.random_uniform(shape=(target_vocab_size, decoding_embedding_size)))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, ids=dec_input)

    def rnn_cell(size):
        return tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMCell(size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2)),
            input_keep_prob=keep_prob
        )

    dec_cell = tf.contrib.rnn.MultiRNNCell([rnn_cell(rnn_size) for _ in range(num_layers)])

    # 3. Dense layer to translate the decoder's output at each time
    # step into a choice from the target vocabulary
    output_layer = Dense(target_vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    with tf.variable_scope('decode'):
        training_output = decoding_layer_train(encoder_state=encoder_state,
                                               dec_cell=dec_cell,
                                               dec_embed_input=dec_embed_input,
                                               target_sequence_length=target_sequence_length,
                                               max_summary_length=max_target_sequence_length,
                                               output_layer=output_layer,
                                               keep_prob=keep_prob)

    with tf.variable_scope('decode', reuse=True):
        inference_output = decoding_layer_infer(encoder_state=encoder_state,
                                                dec_cell=dec_cell,
                                                dec_embeddings=dec_embeddings,
                                                start_of_sequence_id=target_vocab_to_int['<GO>'],
                                                end_of_sequence_id=target_vocab_to_int['<EOS>'],
                                                max_target_sequence_length=max_target_sequence_length,
                                                vocab_size=target_vocab_size,
                                                output_layer=output_layer,
                                                batch_size=batch_size,
                                                keep_prob=keep_prob)

    return training_output, inference_output


def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  source_sequence_length, target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param source_sequence_length: Sequence Lengths of source sequences in the batch
    :param target_sequence_length: Sequence Lengths of target sequences in the batch
    :param max_target_sentence_length:
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    _, encoder_state = encoding_layer(input_data,
                                      rnn_size=rnn_size,
                                      num_layers=num_layers,
                                      source_sequence_length=source_sequence_length,
                                      source_vocab_size=source_vocab_size,
                                      encoding_embedding_size=enc_embedding_size,
                                      keep_prob=keep_prob)

    # Prepare the target sequences we'll feed to the decoder in training mode
    dec_input = process_decoder_input(target_data,
                                      target_vocab_to_int=target_vocab_to_int,
                                      batch_size=batch_size)

    # Pass encoder state and decoder inputs to the decoders

    return decoding_layer(dec_input,
                          encoder_state=encoder_state,
                          target_sequence_length=target_sequence_length,
                          max_target_sequence_length=max_target_sentence_length,
                          rnn_size=rnn_size,
                          num_layers=num_layers,
                          target_vocab_to_int=target_vocab_to_int,
                          target_vocab_size=target_vocab_size,
                          batch_size=batch_size,
                          keep_prob=keep_prob,
                          decoding_embedding_size=dec_embedding_size)


def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    unknown = vocab_to_int["<UNK>"]
    return [vocab_to_int.get(word, unknown) for word in sentence.lower().split()]


def main():
    import problem_unittests as tests

    tests.test_text_to_ids(text_to_ids)
    tests.test_model_inputs(model_inputs)
    tests.test_process_encoding_input(process_decoder_input)
    tests.test_encoding_layer(encoding_layer)
    tests.test_decoding_layer_train(decoding_layer_train)
    tests.test_decoding_layer_infer(decoding_layer_infer)
    tests.test_decoding_layer(decoding_layer)
    tests.test_seq2seq_model(seq2seq_model)
    tests.test_sentence_to_seq(sentence_to_seq)


if __name__ == '__main__':
    main()

