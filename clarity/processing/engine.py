"""Super-resolution engine: spandrel-loaded ESRGAN-family models on the GPU with tiling, or a

Lanczos fallback (CPU, no torch) so the plumbing runs anywhere.

Model slots → weight files. The tracked mapping is `clarity/models/registry.json` (package
data); `<models_dir>/registry.json` beside the weights overrides it per slot. The defaults are the
models Kartoffels' ChaiNNer chains use (KB 08), all on openmodeldb.info:

    bc1clean  1x_BC1-smooth2.pth            1×, run on every BC1 source before anything else
    normal    4x-Normal-RG0-BC7.pth         4×, tangent normals with B zeroed
    color     4x_scalenx_90k.pth            4×, general colour (diffuse / spec / base / world)
    face      4xFaceUpDAT.pth               4×, face base colour
    skin      x1_ITF_SkinDiffDDS_v1.pth     1×, skin de-artefact
    hair      4x_UltraFArt_v3.pth           4×, hair (unused by default: Hair Defined 2 covers hair)
    ui        4x_foolhardy_Remacri.pth      4×, UI sheets and icons (alpha handled outside the model)
"""

import os

import numpy as np

from .. import paths
from ..jsonio import read_json

DEFAULT_REGISTRY = {
    "bc1clean": "1x_BC1-smooth2.pth",  # set to null in registry.json when the colour model removes BC artefacts itself
    "normal": "4x-Normal-RG0-BC7.pth",  # BC7 / uncompressed normal sources
    "normal_bc1": "4x-Normal-RG0-BC1.pth",  # BC1 normal sources (trained on exactly that degradation); falls back to "normal"
    "color": "4x_scalenx_90k.pth",
    "mask": None,  # scalar channels; None = use "color"
    "face": "4xFaceUpDAT.pth",
    "skin": "x1_ITF_SkinDiffDDS_v1.pth",
    "hair": "4x_UltraFArt_v3.pth",
    "ui": "4x_foolhardy_Remacri.pth",
}
# The 2024–2025 picks (see KB 30-postprocess/17). Slot names match the tracked registry.
RECOMMENDED_REGISTRY = {
    "bc1clean": None,  # PBRify models were trained with BC/dds compression in the LR
    "normal": "4x-Normal-RG0-BC7.pth",
    "normal_bc1": "4x-Normal-RG0-BC1.pth",
    "color": "4x-PBRify_RPLKSRd_V3.pth",  # RealPLKSR-DySample, game textures, ~8x faster than DAT2; 4x-PBRify_UpscalerV4.pth for the slow, sharper DAT2
    "mask": "4x-PBRify_UpscalerSPANV4.pth",  # SPAN, fast: three passes per mask
    "face": "4xFaceUpDAT.pth",
    "skin": "x1_ITF_SkinDiffDDS_v1.pth",
    "hair": "4x_UltraFArt_v3.pth",
    "ui": "4x-UltraSharpV2.safetensors",  # DAT2, text/anti-aliasing aware; CC-BY-NC-SA (personal use)
}


class Engine:
    # tile defaults to 512 to match the CLI, so constructing an Engine directly behaves the
    # same as running the command. The two drifting apart is how a "default" stops meaning one.
    def __init__(self, models_dir, device=None, tile=512, pad=16, fp16=True, allow_fallback=True):
        self.models_dir = models_dir
        self.registry = dict(DEFAULT_REGISTRY)
        # registry.json beside the weights first, then the tracked one in scripts-local\clarity-upscale\models
        # (paths.REGISTRY) -- the tracked copy is the configuration of record and wins.
        for reg in ([os.path.join(models_dir, "registry.json")] if models_dir else []) + [
            paths.REGISTRY
        ]:
            if reg and os.path.isfile(reg):
                self.registry.update(read_json(reg))
        self.tile, self.pad, self.fp16 = tile, pad, fp16
        self.allow_fallback = allow_fallback
        self._models = {}
        self.torch = None
        self.device = "cpu"
        self.missing = set()
        try:
            import torch

            self.torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            pass

    # ------------------------------------------------------------------ models
    def path(self, slot):
        if not self.models_dir or not self.registry.get(slot):
            return None
        p = os.path.join(self.models_dir, self.registry[slot])
        return p if os.path.isfile(p) else None

    def has(self, slot: str) -> bool:
        return self.torch is not None and self.path(slot) is not None

    def enabled(self, slot: str) -> bool:
        """A slot mapped to null in registry.json is deliberately off (no fallback)."""
        return bool(self.registry.get(slot))

    def resolve(self, slot):
        """Slot aliases: normal_bc1 → normal, mask → color when unset."""
        if slot == "normal_bc1" and not self.has("normal_bc1"):
            return "normal"
        if slot == "mask" and not self.has("mask"):
            return "color"
        return slot

    def describe(self):
        return ", ".join(
            f"{s}={os.path.basename(self.path(s))}" for s in self.registry if self.has(s)
        )

    def model(self, slot):
        if slot in self._models:
            return self._models[slot]
        from spandrel import ModelLoader

        m = ModelLoader().load_from_file(self.path(slot))
        m = m.to(self.device).eval()
        if self.fp16 and self.device == "cuda" and m.supports_half:
            m = m.half()
        self._models[slot] = m
        return m

    # ------------------------------------------------------------------ inference
    def run(self, slot, img, scale):
        """img: float32 (H, W, C) in [0, 1] → float32 (H*scale, W*scale, C). If the slot's model is

        missing (or torch is), falls back to Lanczos and records the slot in `self.missing`.
        """
        slot = self.resolve(slot)
        if self.has(slot):
            return self._run_model(slot, img, scale)
        if not self.allow_fallback:
            raise RuntimeError(f"model for slot {slot!r} not available ({self.registry.get(slot)})")
        self.missing.add(slot)
        return lanczos(img, scale)

    def run_batch(self, slot, imgs, scale):
        """Several same-sized images through one forward pass (the batch dimension never mixes, so

        each result equals `run` on that image alone). Falls back per image like `run`.
        """
        if not imgs:
            return []
        slot = self.resolve(slot)
        if self.has(slot) and len({i.shape for i in imgs}) == 1:
            # How many fit in one forward pass. The cost is batch x the area actually pushed
            # through, which is the tile for anything larger than a tile and the image itself for
            # anything smaller -- and the icon families are all smaller (40x40, 80x80), which is why
            # keying this off the tile alone was wrong: at `--tile 512` it computed 1 and quietly
            # disabled batching for exactly the textures that need it most.
            ih, iw = imgs[0].shape[:2]
            area = min(ih, self.tile) * min(iw, self.tile)
            per = max(1, (self.tile * self.tile) // max(1, area))
            out = []
            for s in range(0, len(imgs), per):
                chunk = imgs[s : s + per]
                try:
                    out.extend(
                        self._run_model(slot, chunk, scale)
                        if len(chunk) > 1
                        else [self._run_model(slot, chunk[0], scale)]
                    )
                except RuntimeError as e:  # CUDA OOM on a batch: fall back to one at a time
                    if "out of memory" not in str(e).lower() or len(chunk) == 1:
                        raise
                    self.torch.cuda.empty_cache()
                    out.extend(self._run_model(slot, i, scale) for i in chunk)
            return out
        return [self.run(slot, i, scale) for i in imgs]

    def _run_model(self, slot, img, scale):
        torch = self.torch
        m = self.model(slot)
        ms = m.scale
        batch = isinstance(img, (list, tuple))
        imgs = list(img) if batch else [img]
        _H, _W, C = imgs[0].shape
        cin = m.input_channels
        xs = []
        for x in imgs:
            if cin == 1 and C != 1:
                x = x.mean(2, keepdims=True)
            elif cin == 3 and C == 1:
                x = np.repeat(x, 3, 2)
            elif cin == 3 and C == 4:
                x = x[..., :3]
            xs.append(np.ascontiguousarray(x.transpose(2, 0, 1)))
        t = torch.from_numpy(np.stack(xs)).to(self.device)
        if self.fp16 and self.device == "cuda" and m.supports_half:
            t = t.half()
        out = self._tiled(m, t, ms)
        ys = []
        for n in range(len(imgs)):
            y = out[n].float().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            if y.shape[2] != C:
                if C == 1:
                    y = y.mean(2, keepdims=True)
                else:
                    y = np.repeat(y[..., :1], C, 2) if y.shape[2] == 1 else y[..., :C]
            if ms != scale:
                y = lanczos(y, scale / ms)
            ys.append(y.astype(np.float32))
        return ys if batch else ys[0]

    def _tiled(self, m, t, ms):
        torch = self.torch
        N, _C, H, W = t.shape
        tile, pad = self.tile, self.pad
        if tile + 2 * pad >= H and tile + 2 * pad >= W:
            with torch.no_grad():
                return m(_pad8(t))[:, :, : H * ms, : W * ms]
        out = torch.zeros((N, m.output_channels, H * ms, W * ms), dtype=t.dtype, device=t.device)
        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                y1, x1 = min(H, y0 + tile), min(W, x0 + tile)
                py0, px0 = max(0, y0 - pad), max(0, x0 - pad)
                py1, px1 = min(H, y1 + pad), min(W, x1 + pad)
                with torch.no_grad():
                    o = m(_pad8(t[:, :, py0:py1, px0:px1]))
                oy, ox = (y0 - py0) * ms, (x0 - px0) * ms
                out[:, :, y0 * ms : y1 * ms, x0 * ms : x1 * ms] = o[
                    :, :, oy : oy + (y1 - y0) * ms, ox : ox + (x1 - x0) * ms
                ]
        return out


def _pad8(t):
    """Pad to a multiple of 8 (many arches need it); the caller crops the result.

    Reflection has to read a row for every row it invents, so torch refuses a pad that is not
    strictly smaller than the dimension: a 4x4 texture wants 4 and gets
    "padding (0, 4) at dimension 3 of input [1, 3, 4, 4]". The game has such textures --
    `e8805/v01_c0101e8805_glv_n` is 4x4 -- so the degenerate case falls back to replicate, which
    has no such rule. It differs from reflect only in the invented rows, and those are exactly
    the ones cropped off the result.
    """
    import torch.nn.functional as F  # noqa: N812 - the torch convention

    _, _, h, w = t.shape
    ph, pw = (-h) % 8, (-w) % 8
    if not (ph or pw):
        return t
    mode = "reflect" if (pw < w and ph < h) else "replicate"
    return F.pad(t, (0, pw, 0, ph), mode=mode)


def lanczos(img, scale):
    """PIL Lanczos per channel; float32 (H, W, C) → float32."""
    from PIL import Image

    H, W, C = img.shape
    nh, nw = max(1, round(H * scale)), max(1, round(W * scale))
    out = np.empty((nh, nw, C), np.float32)
    for c in range(C):
        im = Image.fromarray((np.clip(img[..., c], 0, 1) * 65535).astype(np.uint16), "I;16")
        out[..., c] = np.asarray(im.resize((nw, nh), Image.LANCZOS), np.float32) / 65535.0
    return out


def box_down(img, factor):
    """Area-average downsample by an integer factor (used to derive the lower tiers)."""
    H, W, C = img.shape
    h, w = H // factor, W // factor
    return (
        img[: h * factor, : w * factor]
        .reshape(h, factor, w, factor, C)
        .mean((1, 3))
        .astype(np.float32)
    )
