import tensorflow as tf
from tensorflow import keras
import numpy as np

class PatchEmbed(keras.layers.Layer):
    """ Image to Patch Embedding

    Args:
        img_size (tuple | (224,224)): input image size.
        patch_size (int | 4): Patch token size.
        in_chans (int | 3): Number of input image channels.
        embed_dim (int | 96): Number of linear projection output channels.
        norm_layer (tf.keras.Model| None): Normalization layer.
    """

    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()

        patch_size = (patch_size, patch_size)

        assert (img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[
            1] == 0), f"Input image size is not divisible by patch size."
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_num = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = keras.layers.Conv2D(filters=embed_dim, kernel_size=patch_size, strides=patch_size,
                                        name="Conv2D")

        self.norm = norm_layer(epsilon=1e-6, name="layer_norm") if norm_layer else None

    def call(self, x):
        B, H, W, C = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        B, H, W, C = x.shape
        # [B, H * W, C]
        x = tf.reshape(x, [B, -1, C])
        if self.norm is not None:
            x = self.norm(x)
        # [B, H * W, C]
        x = tf.reshape(x, [B, H, W, C])
        return x
    

    
class Mlp(keras.layers.Layer):
    """ MLP layer, with two dense layers

    Args:
        hidden_features (int): Layer1 output
        out_features (int): Layer2 output
        act_layer (int): Default: GELU.
    """

    def __init__(self, in_features, hidden_features, out_features,
                 act_layer=keras.layers.Activation('gelu'), drop=0.0):
        super().__init__()

        self.fc1 = keras.layers.Dense(hidden_features)
        self.act = act_layer
        self.fc2 = keras.layers.Dense(out_features)
        self.drop = keras.layers.Dropout(drop)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x
    
    
    
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    windows = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    # * -> [B*num_windows, Mh, Mw, C]
    windows = tf.reshape(windows, [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, -1])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    # * -> [B, H, W, C]
    x = tf.reshape(x, [B, H, W, -1])
    return x



class Window_MSA(keras.layers.Layer):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int] | (7,7)): The height and width of the window.
        num_heads (int| 3): Attention heads number.
        qkv_bias (bool| True):  Add a learnable bias to query, key, value.
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float| 0.0): Dropout ratio of attention weight.
        proj_drop (float| 0.0): Dropout ratio of output.
        relative_pos (bool | True): use relative position or not
    """

    def __init__(self, dim, window_size=(7, 7), num_heads=3, qkv_bias=True, qk_scale=None, attn_drop=0.0,
                 proj_drop =0.0, relative_pos=True):
        super().__init__()
        self.dim = dim

        assert window_size[0] == window_size[1]
        self.window_size = window_size[0]  # Wh, Ww
        self.num_heads = num_heads
        self.relative_pos = relative_pos

        assert dim % num_heads == 0
        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5 if qk_scale == None else qk_scale
        self.qkv = keras.layers.Dense(dim * 3, use_bias=qkv_bias, name="qkv")
        self.attn_drop = keras.layers.Dropout(attn_drop)

        self.proj = keras.layers.Dense(dim, activation=None, use_bias=True)
        self.proj_drop = keras.layers.Dropout(proj_drop)

        self.softmax = keras.layers.Softmax(axis=-1)

        # [2*Wh-1 * 2*Ww-1, num_heads]
        self.relative_position_table = tf.Variable(
            tf.zeros(((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)),
            trainable=True
        )

        coordinates_x = np.arange(window_size[0])
        coordinates_y = np.arange(window_size[1])
        coords = np.stack(np.meshgrid(coordinates_x, coordinates_y, indexing="ij"))  # [2, Mh, Mw]
        coordinates_flatten = np.reshape(coords, [2, -1])  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coordinates = coordinates_flatten[:, :, None] - coordinates_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coordinates = np.transpose(relative_coordinates, [1, 2, 0])  # [Mh*Mw, Mh*Mw, 2]
        relative_coordinates[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coordinates[:, :, 1] += window_size[1] - 1
        relative_coordinates[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coordinates.sum(-1)
        # [Mh*Mw, Mh*Mw]
        self.relative_position_index_tensor = tf.Variable(
            initial_value=tf.constant(relative_position_index),
            trainable=False
        )

    def call(self, x, training=True):
        """
        Args:
            x: input features with shape of [B, H, W, C]
            training: training mode or inference mode
        """
        # [B, H, W, C] -> [num_windows * batch_size, Mh * Mw, Channels]
        B, H, W, C = x.shape

        x = window_partition(x, window_size = self.window_size)

        B_Nwins, Mh, Mw, C = x.shape
        x = tf.reshape(x, [B_Nwins, Mh * Mw, C])

        B_Nwins, Ntokens, C = x.shape
        assert self.dim == C

        # * -> [batch_size * num_windows, Mh * Mw, 3 * dim]
        qkv = self.qkv(x)

        # * -> [batch_size * num_windows, Mh * Mw, 3, num_heads, dim_per_head]
        qkv = tf.reshape(qkv, [B_Nwins, Ntokens, 3, self.num_heads, C // self.num_heads])

        # * -> [3, batch_size * num_windows, num_heads, Mh*Mw, dim_per_head]
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])

        # * -> 3 * [batch_size * num_windows, num_heads, Mh*Mw, dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        # v -> [batch_size * num_windows, num_heads, dim_per_head, Mh*Mw]
        k = tf.transpose(k, [0, 1, 3, 2])

        # * -> [batch_size * num_windows, num_heads, Mh*Mw, Mh*Mw]
        attention = q @ k
        attention = self.softmax(attention)
        attention = self.attn_drop(attention, training=training)

        if self.relative_pos == True:
            # [Mh*Mw, Mh*Mw] -> [Mh*Mw * Mh*Mw]
            relative_position_index = tf.reshape(self.relative_position_index_tensor, [-1])
            # [Mh * Mw * Mh * Mw]
            relative_position = tf.gather(self.relative_position_table, relative_position_index)
            relative_position = tf.reshape(relative_position,
                                           [self.window_size * self.window_size,
                                            self.window_size * self.window_size,
                                            -1])  # Wh*Ww, Wh*Ww, nH

            relative_position = tf.transpose(relative_position, perm=[2, 0, 1])  # nH, Wh*Ww, Wh*Ww

            attention = attention + tf.expand_dims(relative_position, 0)

        # * -> [batch_size * num_windows, num_heads, Mh*Mw, dim_per_head]
        x = attention @ v
        # x -> [batch_size * num_windows, Mh*Mw, num_heads, dim_per_head]
        x = tf.transpose(x, [0, 2, 1, 3])
        # x -> [num_windows * batch_size, Mh * Mw, Channels (num_heads*dim_per_head)]
        assert x.shape[-1] * x.shape[-2] == C
        x = tf.reshape(x, [B_Nwins, Ntokens, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        # * ->  [num_windows * batch_size, Mh, Mw, C]
        x = tf.reshape(x, [B_Nwins, self.window_size, self.window_size, C])

        # [num_windows * batch_size, Mh, Mw, C] -> [B, H, W, C]
        x = window_reverse(x, self.window_size, H, W)

        return x
    

    
    
class Shift_Window_MSA(keras.layers.Layer):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        shift_size (int | 3): Shift size for SW-MSA.
        window_size (tuple[int] | (7,7)): The height and width of the window.
        num_heads (int| 3): Attention heads number.
        qkv_bias (bool| True):  Add a learnable bias to query, key, value.
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float| 0.0): Dropout ratio of attention weight.
        proj_drop (float| 0.0): Dropout ratio of output.
        relative_pos (bool | True): use relative position or not
    """

    def __init__(self, dim, shift_size=3, window_size=(7, 7), num_heads=3, qkv_bias=True, qk_scale=None, attn_drop=0.0,
                 proj_drop = 0.0, relative_pos=True):
        super().__init__()
        self.dim = dim

        assert window_size[0] == window_size[1]
        self.window_size = window_size[0]  # Wh, Ww
        self.num_heads = num_heads
        self.relative_pos = relative_pos

        assert dim % num_heads == 0
        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5 if qk_scale == None else qk_scale
        self.qkv = keras.layers.Dense(dim * 3, use_bias=qkv_bias, name="qkv")
        self.attn_drop = keras.layers.Dropout(attn_drop)

        self.softmax = keras.layers.Softmax(axis=-1)

        # [2*Wh-1 * 2*Ww-1, num_heads]
        self.relative_position_table = tf.Variable(
            tf.zeros(((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)),
            trainable=True
        )

        coordinates_x = np.arange(window_size[0])
        coordinates_y = np.arange(window_size[1])
        coords = np.stack(np.meshgrid(coordinates_x, coordinates_y, indexing="ij"))  # [2, Mh, Mw]
        coordinates_flatten = np.reshape(coords, [2, -1])  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coordinates = coordinates_flatten[:, :, None] - coordinates_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coordinates = np.transpose(relative_coordinates, [1, 2, 0])  # [Mh*Mw, Mh*Mw, 2]
        relative_coordinates[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coordinates[:, :, 1] += window_size[1] - 1
        relative_coordinates[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coordinates.sum(-1)
        # [Mh*Mw, Mh*Mw]
        self.relative_position_index_tensor = tf.Variable(
            initial_value=tf.constant(relative_position_index),
            trainable=False
        )
        self.shift_size = shift_size
        self.proj = keras.layers.Dense(dim, activation=None, use_bias=True)
        self.proj_drop = keras.layers.Dropout(proj_drop)

    def get_mask(self, H, W):
        img_mask = np.zeros([1, H, W, 1])  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
        # calculate attention mask for SW-MSA
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        img_mask = tf.convert_to_tensor(img_mask, dtype=tf.float32)
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = tf.reshape(mask_windows, [-1, self.window_size * self.window_size])  # [nW, Mh*Mw]
        # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
        attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
        attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)

        #[num_w * num_w, Mh * mW, Mh * mW]
        return attn_mask

    def call(self, x, training=True):
        """
        Args:
            x: input features with shape of [B, H, W, C]
            training: training mode or inference mode
        """
        # [B, H, W, C] -> [num_windows * batch_size, Mh * Mw, Channels]
        B, H, W, C = x.shape
        if self.shift_size > 0:
            mask = self.get_mask(H, W)  # [num_w, Mh * mW, Mh * mW]

            # shift window
            x = tf.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            mask = None

        x = window_partition(x, window_size = self.window_size)

        B_Nwins, Mh, Mw, C = x.shape
        x = tf.reshape(x, [B_Nwins, Mh * Mw, C])

        B_Nwins, Ntokens, C = x.shape
        assert self.dim == C

        # * -> [batch_size * num_windows, Mh * Mw, 3 * dim]
        qkv = self.qkv(x)

        # * -> [batch_size * num_windows, Mh * Mw, 3, num_heads, dim_per_head]
        qkv = tf.reshape(qkv, [B_Nwins, Ntokens, 3, self.num_heads, C // self.num_heads])

        # * -> [3, batch_size * num_windows, num_heads, Mh*Mw, dim_per_head]
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])

        # * -> 3 * [batch_size * num_windows, num_heads, Mh*Mw, dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        # v -> [batch_size * num_windows, num_heads, dim_per_head, Mh*Mw]
        k = tf.transpose(k, [0, 1, 3, 2])

        # * -> [batch_size * num_windows, num_heads, Mh*Mw, Mh*Mw]
        attention = q @ k
        attention = self.softmax(attention)
        attention = self.attn_drop(attention, training=training)

        if self.relative_pos == True:
            # [Mh*Mw, Mh*Mw] -> [Mh*Mw * Mh*Mw]
            relative_position_index = tf.reshape(self.relative_position_index_tensor, [-1])
            # [Mh * Mw * Mh * Mw]
            relative_position = tf.gather(self.relative_position_table, relative_position_index)
            relative_position = tf.reshape(relative_position,
                                           [self.window_size * self.window_size,
                                            self.window_size * self.window_size,
                                            -1])  # Wh*Ww, Wh*Ww, nH

            relative_position = tf.transpose(relative_position, perm=[2, 0, 1])  # nH, Wh*Ww, Wh*Ww

            attention = attention + tf.expand_dims(relative_position, 0)

        # mask: [num_w, Mh * mW, Mh * mW]
        if mask is not None:
            num_windows = mask.shape[0]
            # attention [batch_size * num_windows, num_heads, Mh*Mw, Mh*Mw] -> [batch_size, num_windows, num_heads, Mh * Mw, Mh * Mw]
            attention = tf.reshape(attention, [B_Nwins // num_windows, num_windows, self.num_heads, Ntokens, Ntokens])

            # mask: [num_w, Mh * mW, Mh * mW] -> mask: [1, num_w, 1, Mh * mW, Mh * mW]
            mask_expanded = tf.expand_dims(tf.expand_dims(mask, 1), 0)

            # broadcast
            attention += mask_expanded

            # attention : [batch_size, num_windows, num_heads, Mh * Mw, Mh * Mw] -> [batch_size * num_windows, num_heads, Mh*Mw, Mh*Mw]
            attention = tf.reshape(attention, [-1, self.num_heads, Ntokens, Ntokens])

        # * -> [batch_size * num_windows, num_heads, Mh*Mw, dim_per_head]
        x = attention @ v
        # x -> [batch_size * num_windows, Mh*Mw, num_heads, dim_per_head]
        x = tf.transpose(x, [0, 2, 1, 3])
        # x -> [num_windows * batch_size, Mh * Mw, Channels (num_heads*dim_per_head)]
        assert x.shape[-1] * x.shape[-2] == C
        x = tf.reshape(x, [B_Nwins, Ntokens, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        # * ->  [num_windows * batch_size, Mh, Mw, C]
        x = tf.reshape(x, [B_Nwins, self.window_size, self.window_size, C])

        # [num_windows * batch_size, Mh, Mw, C] -> [B, H, W, C]
        x = window_reverse(x, self.window_size, H, W)

        if self.shift_size > 0:
            x = tf.roll(x, shift=(self.shift_size, self.shift_size), axis=(1, 2))

        return x
    
    
    
class PatchMerging(keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = keras.layers.Dense(2*dim, use_bias=False, name = "patch_merging")
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")

    def call(self, x, training):
        
        """
        Args:
        x: input features with shape of [B, H, W, C]
        """
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = tf.concat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = tf.reshape(x, [B, -1, 4*C])  # [B, H/2*W/2, 4*C]
        x = self.norm(x)
        x = self.linear(x)  # [B, H/2*W/2, 2*C]
        x = tf.reshape(x, [B, int(H/2), int(W/2), C * 2])
        return x
    
    
    
    
    
class SwintransformerBlock(keras.layers.Layer):
    """ Two Successive Swin Transformer Blocks.

    Args:
        dim (int): Number of input channels.
        shift_size (int | 3): Shift size for SW-MSA.
        window_size (tuple[int] | (7,7)): The height and width of the window.
        num_heads (int| 3): Attention heads number.
        qkv_bias (bool| True):  Add a learnable bias to query, key, value.
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float| 0.0): Dropout ratio of attention weight.
        mlp_drop (float| 0.0): Dropout ratio of mlp weight.
        relative_pos (bool | True): use relative position or not
        drop_path (list(float)| [0.0, 0.0]): Stochastic depth rate.
    """
    def __init__(self, dim, shift_size = 3, window_size=(7, 7), num_heads=3, qkv_bias=True, qk_scale=None, attn_drop=0.0,
                 mlp_drop=0.0, drop_path = [0.0, 0.0], relative_pos=True):
        super().__init__()
        """
        Initialize the Swin Transformer Block.
        """
        self.dim = dim
        assert window_size[0] == window_size[1]
        self.window_size = window_size[0]
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.mlp_drop = mlp_drop
        self.relative_pos = relative_pos
        self.patch_merging = PatchMerging(dim=dim)
        self.window_msa = Window_MSA(dim=dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop, relative_pos=relative_pos)
        self.shift_window_msa = Shift_Window_MSA(dim=dim, window_size=window_size, shift_size = shift_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                                qk_scale=qk_scale, attn_drop=attn_drop, relative_pos=relative_pos)
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm")
        self.mlp = keras.layers.Dense(dim, name="mlp")
        self.mlp_drop = keras.layers.Dropout(mlp_drop)
        self.drop_path_0 = keras.layers.Dropout(drop_path[0])
        self.drop_path_1 = keras.layers.Dropout(drop_path[1])

    def call(self, x, training):
        # x: [B, H, W, C]
        shortcat = x
        B, H, W, C = x.shape
        L = H * W
        # * -> [B, L, C]
        x = tf.reshape(x, [B, L, C])
        x = self.norm(x)
        x = tf.reshape(x, [B, H, W, C])
        x = self.window_msa(x, training=training)
        x = shortcat + self.drop_path_0(x)

        # * -> [B, L, C]
        x = tf.reshape(x, [B, L, C])

        x = x + self.drop_path_0(self.mlp(self.norm(x), training = training))

        """
        second block
        """
        # x: [B, L, C]
        shortcat_2 = x
        x = self.norm(x)
        # * -> [B, H, W, C]
        x = tf.reshape(x, [B, H, W, C])

        if self.shift_size > 0:
            x = self.shift_window_msa(x, training=training)
        else:
            x = self.window_msa(x, training=training)
        # * -> [B, L, C]
        x = tf.reshape(x, [B, L, C])
        x = shortcat_2 + self.drop_path_1(x)

        x = x + self.drop_path_1(self.mlp(self.norm(x), training = training))

        # [B, L, C] -> [B, H, W, C]
        x = tf.reshape(x, [B, H, W, C])
        return x
    
    
    
    
    
class Swintransformer(keras.layers.Layer):
    """
      Args:
        img_size (tuple | (224,224)): input image size.
        num_classes (int | 1000): Number of classes for classification head.
        num_stages (int | 4): Number of layers
        num_depths (list(int)| [2,2,6,2])
        patch_size (int | 4): Patch token size.
        embed_dim (int | 96): Number of linear projection output channels.
        norm_layer (tf.keras.Model| None): Normalization layer.
        window_size (tuple[int] | (7,7)): The height and width of the window.
        num_heads (list(int)| [3,6,12,24]): Attention heads number.
        qkv_bias (bool| True):  Add a learnable bias to query, key, value.
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float| 0.0): Dropout ratio of attention weight.
        mlp_drop (float| 0.0): Dropout ratio of mlp weight.
        drop_path (float| 0.0): Stochastic depth rate.
        relative_pos (bool | True): use relative position or not.

    """

    def __init__(self, img_size=(224, 224), num_classes=1000, num_stages=4, patch_size=4, embed_dim=96,
                 num_depths=[1, 1, 3, 1], norm_layer=keras.layers.LayerNormalization, window_size=(7, 7),
                 num_heads=[3, 6, 12, 24], qkv_bias=True, qk_scale=None, attn_drop=0.0, mlp_drop=0.0,
                 drop_path=0.1, relative_pos=True):
        super().__init__()
        
        # shift_size = 0
        shift_size  = window_size[0] // 2

        # path drop rate
        num_points = sum(num_depths) * 2
        dpr = tf.linspace(0.0, drop_path, num_points)
        dpr = [x.numpy().item() for x in dpr]

        # Initialize layers
        assert len(num_depths) == num_stages, "lack num of depths hypoparameter"
        assert len(num_heads) == num_stages, "lack num of heads hypoparameter"
        self.layers = []

        # norm layer
        self.norm = norm_layer(epsilon=1e-6, name="layer_norm")
        # embedding
        self.embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim,
                                norm_layer=norm_layer)

        # first layer, do not contain patch merging, but contain patch embedding
        drop = 0
        for j in range(num_depths[0]):
            self.layers.append(SwintransformerBlock(dim=embed_dim, shift_size=shift_size, window_size=window_size,
                                                    num_heads=num_heads[0], qkv_bias=True,
                                                    qk_scale=None, attn_drop=attn_drop, mlp_drop=mlp_drop,
                                                    drop_path=dpr[drop:drop+2], relative_pos=True))
            drop += 2

        # from 2nd layer to the last one
        dim = embed_dim
        for i in range(1, num_stages):
            self.layers.append(PatchMerging(dim=dim))
            dim = dim * 2
            depth = num_depths[i]
            for j in range(depth):
                self.layers.append(SwintransformerBlock(dim=dim, shift_size=shift_size, window_size=window_size,
                                                        num_heads=num_heads[1], qkv_bias=True,
                                                        qk_scale=None, attn_drop=attn_drop, mlp_drop=mlp_drop,
                                                        drop_path=dpr[drop:drop+2], relative_pos=True))
                drop += 2

        # classification head
        # self.head = keras.layers.Dense(num_classes, name="head")
        self.head = keras.layers.Dense(num_classes, activation='softmax', name="head")


    def forward_feature(self, x, training):
        x = self.embed(x, training=training)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        B, H, W, C = x.shape
        x = tf.reshape(x, [B, H * W, C])
        x = self.norm(x)

        # * -> [B, C, L]
        x = tf.transpose(x, perm=[0, 2, 1])
        # * -> [B ,C, 1]
        x = tf.reduce_mean(x, axis=-1, keepdims=True)
        # * -> [B, C]
        x = tf.squeeze(x, axis=-1)
        return x

    def call(self, x, training):
        x = self.forward_feature(x, training)
        x = self.head(x)
        return x