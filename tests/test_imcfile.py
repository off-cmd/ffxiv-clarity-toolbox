"""``clarity.ffxiv.imcfile``: the .imc variant table.

Layout (little-endian)::

     0  u16 count       number of variants, *excluding* the default entry
     2  u16 part_mask   one bit per equipment part present
     4  (count + 1) x popcount(part_mask) entries of 6 bytes:
            u8 material_id, u8 decal_id, u16 attribute_and_sound, u8 vfx_id, u8 material_animation_id

Entry 0 is the default; ``entry(variant, slot)`` takes the 1-based variant from Item.ModelMain.
``attribute_and_sound`` packs a 10-bit attribute mask in the low bits and the sound id above it.
"""

import struct

import numpy as np
import pytest

from clarity.ffxiv import imcfile

ALL_PARTS = 0x1F  # met, top, glv, dwn, sho


def _entry(m: int, d: int, attr: int, sound: int, v: int, ma: int) -> bytes:
    return struct.pack("<2BH2B", m, d, (sound << 10) | attr, v, ma)


def _imc(count: int, part_mask: int, entries: list[bytes]) -> bytes:
    parts = bin(part_mask).count("1")
    assert len(entries) == (count + 1) * parts
    return struct.pack("<2H", count, part_mask) + b"".join(entries)


def _random_imc(seed: int, count: int, part_mask: int) -> tuple[bytes, list[list[tuple]]]:
    """A table with distinct random fields per (variant, part), plus the values used."""
    rng = np.random.default_rng(seed)
    parts = bin(part_mask).count("1")
    table, entries = [], []
    for _variant in range(count + 1):
        row = []
        for _part in range(parts):
            m, d, v, ma = (int(x) for x in rng.integers(0, 256, 4))
            attr, sound = int(rng.integers(0, 1 << 10)), int(rng.integers(0, 1 << 6))
            row.append((m, d, attr, sound, v, ma))
            entries.append(_entry(m, d, attr, sound, v, ma))
        table.append(row)
    return _imc(count, part_mask, entries), table


def test_header_counts() -> None:
    data, _ = _random_imc(0, count=3, part_mask=ALL_PARTS)
    imc = imcfile.ImcFile(data)
    assert imc.count == 3
    assert imc.part_mask == ALL_PARTS
    assert imc.part_count == 5
    assert len(imc.variants) == 4  # default + 3
    assert all(len(v) == 5 for v in imc.variants)


@pytest.mark.parametrize(("slot", "part"), sorted(imcfile.SLOT_PART.items()))
def test_entry_decodes_every_field_for_every_slot(slot: str, part: int) -> None:
    data, table = _random_imc(1, count=2, part_mask=ALL_PARTS)
    imc = imcfile.ImcFile(data)
    for variant in range(3):
        m, d, attr, sound, v, ma = table[variant][part]
        e = imc.entry(variant, slot)
        assert e is not None
        assert e.material_id == m
        assert e.decal_id == d
        assert e.attribute_and_sound == (sound << 10) | attr
        assert e.attribute_mask == attr
        assert e.sound_id == sound
        assert e.vfx_id == v
        assert e.material_animation_id == ma


def test_attribute_mask_and_sound_id_split_the_packed_word() -> None:
    data = _imc(0, 0b1, [_entry(1, 0, 0x3FF, 0x3F, 0, 0)])
    e = imcfile.ImcFile(data).entry(0, "top")
    assert e is not None
    assert e.attribute_and_sound == 0xFFFF
    assert e.attribute_mask == 0x3FF
    assert e.sound_id == 0x3F
    assert repr(e) == "Imc(mat=1 decal=0 vfx=0 attr=0x3ff)"


def test_variant_zero_is_the_default_entry() -> None:
    data = _imc(
        1,
        0b1,
        [_entry(9, 0, 0, 0, 0, 0), _entry(1, 0, 0, 0, 0, 0)],  # default: material 9
    )
    imc = imcfile.ImcFile(data)
    assert imc.entry(0, "top").material_id == 9
    assert imc.entry(1, "top").material_id == 1
    assert imc.variants[0][0] is imc.entry(0, "top")


def test_single_part_tables_ignore_the_slot() -> None:
    # Weapons and monsters store one part; any slot name resolves to it.
    data = _imc(0, 0b1, [_entry(4, 0, 0, 0, 0, 0)])
    imc = imcfile.ImcFile(data)
    assert imc.part_count == 1
    for slot in ("met", "sho", "ril", "not-a-slot"):
        assert imc.entry(0, slot).material_id == 4


def test_out_of_range_variant_or_part_returns_none() -> None:
    data, _ = _random_imc(2, count=1, part_mask=0b111)  # 3 parts: indices 0..2
    imc = imcfile.ImcFile(data)
    assert imc.entry(2, "top") is None  # only variants 0 and 1 exist
    assert imc.entry(1, "sho") is None  # part 4 is beyond the 3 stored parts
    assert imc.entry(1, "glv") is not None


def test_unknown_slot_falls_back_to_part_zero() -> None:
    # The module does not raise: SLOT_PART.get(slot, 0) silently picks the first part. This
    # pins that behaviour so a change to raising is a deliberate one.
    data, table = _random_imc(3, count=1, part_mask=ALL_PARTS)
    imc = imcfile.ImcFile(data)
    e = imc.entry(1, "xyz")
    assert e is not None
    assert e.material_id == table[1][0][0]
    assert e is imc.entry(1, "met")


def test_truncated_table_raises() -> None:
    data, _ = _random_imc(4, count=1, part_mask=ALL_PARTS)
    with pytest.raises(struct.error):
        imcfile.ImcFile(data[:-1])
