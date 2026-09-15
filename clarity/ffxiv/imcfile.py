"""FFXIV .imc (Image Change) parser. Spec: file-formats/010editor/imc.bt

Maps an item VARIANT to the material folder / decal / vfx actually used.
This is the link between Item.ModelMain's variant number and which
material/v#### directory to load - they are NOT the same number.
"""

import struct

SLOT_PART = {
    "met": 0,
    "top": 1,
    "glv": 2,
    "dwn": 3,
    "sho": 4,
    "ear": 0,
    "nek": 1,
    "wrs": 2,
    "rir": 3,
    "ril": 4,
}


class ImcEntry:
    __slots__ = (
        "material_id",
        "decal_id",
        "attribute_and_sound",
        "vfx_id",
        "material_animation_id",
    )

    def __init__(self, m, d, a, v, ma):
        self.material_id = m
        self.decal_id = d
        self.attribute_and_sound = a
        self.vfx_id = v
        self.material_animation_id = ma

    @property
    def attribute_mask(self):
        return self.attribute_and_sound & 0x03FF

    @property
    def sound_id(self):
        return self.attribute_and_sound >> 10

    def __repr__(self):
        return (
            f"Imc(mat={self.material_id} decal={self.decal_id} "
            f"vfx={self.vfx_id} attr=0x{self.attribute_mask:03x})"
        )


class ImcFile:
    def __init__(self, data: bytes):
        count, part_mask = struct.unpack_from("<2H", data, 0)
        self.part_count = bin(part_mask).count("1")
        self.part_mask = part_mask
        self.count = count
        self.variants = []  # [variant][part] -> ImcEntry
        o = 4
        for _ in range(count + 1):  # entry 0 is the default
            parts = []
            for _ in range(self.part_count):
                m, d, a, v, ma = struct.unpack_from("<2BH2B", data, o)
                o += 6
                parts.append(ImcEntry(m, d, a, v, ma))
            self.variants.append(parts)

    def entry(self, variant, slot):
        """variant is 1-based as stored in Item.ModelMain."""
        part = SLOT_PART.get(slot, 0)
        if self.part_count == 1:
            part = 0
        if variant >= len(self.variants) or part >= self.part_count:
            return None
        return self.variants[variant][part]


def load(gd, eid, kind="equipment"):
    p = f"chara/{kind}/e{eid:04d}/e{eid:04d}.imc"
    if not gd.exists(p):
        return None
    return ImcFile(gd.read(p))
