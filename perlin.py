# perlin.py
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.autoinit
from config import Config

KERNEL_CODE = r"""
__device__ float fade(float t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ float grad(int hash, float x, float y) {
    int h = hash & 3;
    float su = 1.0f - 2.0f * (float)(h & 1);
    float sv = 1.0f - 2.0f * (float)((h & 2) >> 1);
    float u = (h < 2) ? x : y;
    float v = (h < 2) ? y : x;
    return u * su + v * sv;
}

__device__ float perlin2d(unsigned char *p, float Xf, float Yf) {
    int X = ((int)floorf(Xf)) & 255;
    int Y = ((int)floorf(Yf)) & 255;

    float xf = Xf - floorf(Xf);
    float yf = Yf - floorf(Yf);

    float u = fade(xf);
    float v = fade(yf);

    int aa = p[p[X] + Y];
    int ab = p[p[X] + (Y+1)];
    int ba = p[p[X+1] + Y];
    int bb = p[p[X+1] + (Y+1)];

    float x1 = (1 - u)*grad(aa, xf, yf) + u*grad(ba, xf-1, yf);
    float x2 = (1 - u)*grad(ab, xf, yf-1) + u*grad(bb, xf-1, yf-1);
    float val = (1 - v)*x1 + v*x2;
    float mapped = (val + 1.0f)*127.5f;
    if (mapped < 0.0f) mapped = 0.0f;
    if (mapped > 255.0f) mapped = 255.0f;
    return mapped;
}

__global__ void perlin_kernel(unsigned char *p,
                              unsigned char *img,
                              int width, int height,
                              float Rx, float Ry, float Rscale,
                              float Gx, float Gy, float Gscale,
                              float Bx, float By, float Bscale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float Rval = perlin2d(p, (x + Rx)*Rscale, (y + Ry)*Rscale);
    float Gval = perlin2d(p, (x + Gx)*Gscale, (y + Gy)*Gscale);
    float Bval = perlin2d(p, (x + Bx)*Bscale, (y + By)*Bscale);

    int idx = (y * width + x)*3;
    img[idx]   = (unsigned char)Rval;
    img[idx+1] = (unsigned char)Gval;
    img[idx+2] = (unsigned char)Bval;
}
"""

mod = SourceModule(KERNEL_CODE, options=['-std=c++11'])
perlin_func = mod.get_function("perlin_kernel")

class PerlinOptimizedGenerator:
    def __init__(self,
                 width=Config.EMISSION_WIDTH,
                 height=Config.EMISSION_HEIGHT,
                 block_size=Config.PERLIN_BLOCK_SIZE,
                 offset_range=Config.PERLIN_OFFSET_RANGE,
                 scale_modulus=Config.PERLIN_SCALE_MODULUS,
                 scale_divisor=Config.PERLIN_SCALE_DIVISOR,
                 scale_min=Config.PERLIN_SCALE_MIN):

        self.width = width
        self.height = height
        self.block_size = block_size
        self.grid_size = ((width + block_size[0]-1)//block_size[0],
                          (height + block_size[1]-1)//block_size[1])

        self.offset_range = offset_range
        self.scale_modulus = scale_modulus
        self.scale_divisor = scale_divisor
        self.scale_min = scale_min

        self.gpu_image = np.zeros((height, width, 3), dtype=np.uint8)
        self.img_gpu = drv.mem_alloc(self.gpu_image.nbytes)

        self.p = np.zeros(512, dtype=np.uint8)
        self.p_gpu = drv.mem_alloc(self.p.nbytes)

    def __del__(self):
        self.p_gpu.free()
        self.img_gpu.free()

    def _bytes_to_signed_int(self, b):
        return int.from_bytes(b, byteorder='little', signed=True)

    def _derive_params_from_seed(self, seed: bytes):
        x_int = self._bytes_to_signed_int(seed[0:4])
        y_int = self._bytes_to_signed_int(seed[4:8])

        x_off = (x_int / (2**31)) * self.offset_range
        y_off = (y_int / (2**31)) * self.offset_range

        scale_factor = (abs(x_int) % self.scale_modulus) / self.scale_divisor + self.scale_min
        return (x_off, y_off, scale_factor)

    def _make_permutation(self, seed_bytes):
        perm = np.arange(256, dtype=np.uint8)
        idx = 0
        seed_len = len(seed_bytes)
        for i in range(255, 0, -1):
            r = seed_bytes[idx % seed_len]
            idx += 1
            swap_index = r % (i+1)
            temp = perm[i]
            perm[i] = perm[swap_index]
            perm[swap_index] = temp
        return perm

    def generate_image_from_hash(self, digest: bytes):
        if not isinstance(digest, bytes) or len(digest) != 32:
            raise ValueError("digest must be exactly 32 bytes (256 bits).")

        global_seed = digest[0:8]
        R_seed = digest[8:16]
        G_seed = digest[16:24]
        B_seed = digest[24:32]

        global_perm = self._make_permutation(global_seed)
        for i in range(512):
            self.p[i] = global_perm[i & 255]

        drv.memcpy_htod(self.p_gpu, self.p)

        R_x_off, R_y_off, R_scale = self._derive_params_from_seed(R_seed)
        G_x_off, G_y_off, G_scale = self._derive_params_from_seed(G_seed)
        B_x_off, B_y_off, B_scale = self._derive_params_from_seed(B_seed)

        perlin_func(self.p_gpu,
                    self.img_gpu,
                    np.int32(self.width), np.int32(self.height),
                    np.float32(R_x_off), np.float32(R_y_off), np.float32(R_scale),
                    np.float32(G_x_off), np.float32(G_y_off), np.float32(G_scale),
                    np.float32(B_x_off), np.float32(B_y_off), np.float32(B_scale),
                    block=self.block_size, grid=self.grid_size)

        drv.memcpy_dtoh(self.gpu_image, self.img_gpu)
        from PIL import Image
        return Image.fromarray(self.gpu_image, 'RGB')

perlin_generator = PerlinOptimizedGenerator()

def generate_image_from_hash(digest: bytes):
    return perlin_generator.generate_image_from_hash(digest)
