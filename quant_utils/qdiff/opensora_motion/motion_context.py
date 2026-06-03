class MotionContext:
    def __init__(self):
        self.clear()

    def clear(self):
        self.enabled = False
        self.batch_size = None
        self.num_frames = None
        self.num_spatial_tokens = None
        self.visual_bits = None
        self.spatial_bits = None
        self.temporal_bits = None

    def set_bits(self, token_bits, num_frames, num_spatial_tokens):
        self.enabled = True
        self.batch_size = token_bits.shape[0]
        self.num_frames = num_frames
        self.num_spatial_tokens = num_spatial_tokens
        self.visual_bits = token_bits.reshape(self.batch_size, num_frames * num_spatial_tokens)
        self.spatial_bits = token_bits.reshape(self.batch_size * num_frames, num_spatial_tokens)
        self.temporal_bits = token_bits.permute(0, 2, 1).reshape(self.batch_size * num_spatial_tokens, num_frames)

    def resolve_bits(self, x_shape):
        if not self.enabled or len(x_shape) != 3:
            return None

        batch, token_num, _ = x_shape
        if batch == self.batch_size and token_num == self.num_frames * self.num_spatial_tokens:
            return self.visual_bits
        if batch == self.batch_size * self.num_frames and token_num == self.num_spatial_tokens:
            return self.spatial_bits
        if batch == self.batch_size * self.num_spatial_tokens and token_num == self.num_frames:
            return self.temporal_bits
        return None


motion_context = MotionContext()

