from swin_transformer import Swintransformer
import tensorflow as tf

class SwinModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.swin_transformer = Swintransformer(
            img_size=(224,224),
            num_classes=config['DATA']['NUM_CLASSES'],
            num_stages=config['MODEL']['SWIN']['NUM_STAGES'],
            patch_size=config['MODEL']['SWIN']['PATCH_SIZE'],
            embed_dim=config['MODEL']['SWIN']['EMBED_DIM'],
            num_depths=config['MODEL']['SWIN']['NUM_DEPTHS'],
            norm_layer=tf.keras.layers.LayerNormalization,
            window_size=(7,7),
            num_heads=config['MODEL']['SWIN']['NUM_HEADS'],
            qkv_bias=True,
            qk_scale=None,
            attn_drop=config['MODEL']['ATTN_DROP'],
            mlp_drop=config['MODEL']['SWIN']['MLP_DROP'],
            drop_path=config['MODEL']['DROP_PATH_RATE'],
            relative_pos=True
        )

    def call(self, inputs, training=False):
        x = self.swin_transformer(inputs)
        return x