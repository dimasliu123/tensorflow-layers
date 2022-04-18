import tensorflow as tf

class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 embed_dim : int, 
                 vocab_size : int, 
                 max_len : int, 
                 **kwargs):

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.vocab_embed = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_embed = tf.keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim) 
        self.pos = tf.range(start=0, limit=max_len, delta=1)

    def call(self, inputs):
        x1 = self.vocab_embed(inputs) 
        x2 = self.pos_embed(self.pos)
        return x1 + x2

    def get_config(self, **kwargs):
        config = super(PositionEmbedding, self).get_config()
        config.update({
            "embed_dim" : self.embed_dim,
            "vocab_size" : self.vocab_size,
            "max_len" : self.max_len,
            "vocab_embed" : self.vocab_embed,
            "pos_embed" : self.pos_embed,
            "pos" : self.pos
        })
        return config

class Transformer(tf.keras.layers.Layer):
	def __init__(self, 
                 embed_dim : int, 
                 ff_dim : int, 
                 num_heads : int, 
                 act_ff : str ="gelu", 
                 **kwargs):
		super(Transformer, self).__init__()
		self.embed_dim = embed_dim
		self.ff_dim = ff_dim
		self.act_ff = act_ff
		self.num_heads = num_heads
		self.att = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=embed_dim)
		self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation=tf.nn.gelu if act_ff == "gelu" else tf.nn.relu),
                                         tf.keras.layers.Dense(embed_dim), ])
		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs):
		att = self.att(inputs, inputs)
		x = self.layernorm1(att + inputs)
		out = self.ffn(x)
		return self.layernorm2(out + x)

	def get_config(self):
		config = super(Transformer, self).get_config()
		config.update({
			'embed_dim' : self.embed_dim,
			'ff_dim' : self.ff_dim,
			'num_heads' : self.num_heads,
			'act_ff' : self.act_ff,
			'att' : self.att,
			'ffn' : self.ffn,
			'layernorm1' : self.layernorm1,
			'layernorm2' : self.layernorm2
		})
		return config

class CausalTransformer(tf.keras.layers.Layer):
    def __init__(self, 
                 embed_dim : int, 
                 ff_dim : int, 
                 num_heads : int, 
                 act_ff : str = "gelu", 
                 use_dropout : bool = False, 
                 **kwargs):

        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.act_ff = act_ff
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation=tf.nn.gelu if act_ff == "gelu" else tf.nn.relu), tf.keras.layers.Dense(embed_dim, ) ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def causal_attention_mask(self, inputs):
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        i = tf.range(seq_len)[:, tf.newaxis]
        j = tf.range(seq_len)
        mask = tf.cast(i >= j, dtype=tf.int32)
        mask = tf.reshape(mask, (1, seq_len, seq_len))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)

    def call(self, inputs):
        causal_mask = self.causal_attention_mask(inputs[0])
        att = self.mha(query=inputs[0], key=inputs[1], key=inputs[2], attention_mask=causal_mask)
        x = self.norm1(att + inputs[0], inputs[1], inputs[2])
        ff = self.ffn(x)
        return self.norm2(x + ff)

    def get_config(self, **kwargs):
        config = super(CausalTransformer, self).get_config()
        config.update({
           "embed_dim" : self.embed_dim,
           "ff_dim" : self.ff_dim,
           "num_heads" : self.num_heads,
           "act_ff" : self.act_ff,
           "mha" : self.mha,
           "ffn" : self.ffn,
           "norm1" : self.norm1,
           "norm2" : self.norm2
        })
        return config

class VectorQuantizer(layers.Layer):
    def __init__(self, 
                 num_embed : int, 
                 embed_dim : int, 
                 beta : float=0.25, 
                 w_init = tf.random_uniform_initializer(), 
                 **kwargs):

        super(VectorQuantizer, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_embed = num_embed
        self.beta = beta
        self.w_init = w_init 
        self.embedding = tf.Variable(initial_value=w_init(shape=(self.embed_dim, self.num_embed), dtype=tf.float32), trainable=True)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        flattened = tf.reshape(inputs, [-1, self.embed_dim])

        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embed)

        quantized = tf.matmul(encodings, self.embedding, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        commitment_loss = self.beta * tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True) + tf.reduce_sum(self.embeddings ** 2, axis=0) - 2 * similarity)
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

    def get_config(self, **kwargs):
        config = super(VectorQuantizer, self).get_config()
        config.update({
            "embed_dim" : self.embed_dim,
            "num_embed" : self.num_embed,
            "beta" : self.beta,
            "w_init" : self.w_init,
            "embedding" : self.embedding
        })
        return config
