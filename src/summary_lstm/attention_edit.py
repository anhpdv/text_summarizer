import tensorflow as tf


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Input shapes should be 3D tensors: (batch_size, sequence_length, hidden_size)
        assert len(input_shape) == 2
        # Shape of the encoder and decoder outputs
        enc_output_shape, dec_output_shape = input_shape

        print("Encoder output shape:", enc_output_shape)
        print("Decoder output shape:", dec_output_shape)

        # Validate the shapes
        assert len(enc_output_shape) == 3
        assert len(dec_output_shape) == 2  # Updated to handle 2D decoder output

        # The last dimension of the encoder output should match the last dimension of the decoder output
        assert (
            enc_output_shape[2] == dec_output_shape[1]
        )  # Updated to handle 2D decoder output

        # The shape of the attention weights should match the shape of the decoder output
        self.W_a = self.add_weight(
            name="W_a",
            shape=tf.TensorShape(
                (dec_output_shape[1], dec_output_shape[1])
            ),  # Updated to handle 2D decoder output
            initializer="uniform",
            trainable=True,
        )

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        assert isinstance(inputs, list)
        # Input tensors: [decoder_output, encoder_output]
        decoder_output, encoder_output = inputs

        # Compute attention scores
        score = tf.matmul(
            decoder_output, tf.matmul(encoder_output, self.W_a, transpose_b=True)
        )

        # Compute attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # Compute context vector
        context_vector = tf.reduce_sum(attention_weights * encoder_output, axis=1)

        return context_vector

    def compute_output_shape(self, input_shape):
        print("input_shape", input_shape)
        """Outputs produced by the layer"""
        batch_size = input_shape[0][0]  # Get the batch size
        sequence_length = input_shape[0][1]  # Get the sequence length
        hidden_size = input_shape[0][2]  # Get the hidden size
        return [
            tf.TensorShape((batch_size, sequence_length, hidden_size)),
            tf.TensorShape(
                (batch_size, sequence_length, hidden_size)
            ),  # Adjust based on your desired output shape
        ]
