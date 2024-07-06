import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MultiHeadAttention, LayerNormalization,GlobalAveragePooling1D, Layer, Dot, Softmax, Reshape, concatenate, Flatten
from tensorflow.keras.regularizers import L2 as l2
    
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_transformer(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], dropout=0.25, mlp_dropout=0.4 ):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu", kernel_regularizer=l2(0.0005))(x)
        x = Dropout(mlp_dropout)(x)

    outputs = x
    return inputs, outputs


############################# unimodal ###################################

def get_unimodal(input_shape, n_classes):
    inputs, intermediate_outputs = build_transformer(input_shape)
    probs = Dense(n_classes, activation='sigmoid', kernel_regularizer=l2(0.0005))(intermediate_outputs)
    model =  tf.keras.Model(inputs, probs)
    
    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    return model


########################### multimodal ##################################


class CrossAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(CrossAttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(CrossAttentionLayer, self).build(input_shape)
        
    def call(self, inputs):
        embedding_1, embedding_2 = inputs
        
        # Calculate attention weights
        attention_weights = Dot(axes=-1, normalize=True)([embedding_2, embedding_1])
        attention_weights = Softmax()(attention_weights)
        
        # Apply attention weights to embedding_1
        attended_embedding_1 = Dot(axes=1)([attention_weights, embedding_1])
        
        return attended_embedding_1
    
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
def get_multimodal(power_shape, em_shape, n_classes):
    power_inputs, power_embeddings = build_transformer(power_shape)
    em_inputs, em_embeddings = build_transformer(em_shape)

    power_embeddings = Reshape(target_shape=(1,128))(power_embeddings)
    em_embeddings = Reshape(target_shape=(1,128))(em_embeddings)
    attended1 = CrossAttentionLayer()([power_embeddings, em_embeddings])
    attended2 = CrossAttentionLayer()([em_embeddings, power_embeddings])
    fused_embeddings = concatenate([attended1,attended2])
    flat = Flatten()(fused_embeddings)
    final_output = Dense(n_classes, activation='sigmoid', kernel_regularizer=l2(0.0005))(flat)
    
    model = tf.keras.Model([power_inputs, em_inputs], final_output)
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
    # print(model.summary())
    return model