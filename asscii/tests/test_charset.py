from __future__ import annotations

import pytest

from src.charset import get_ramp


def test_sparsest_ramp_is_two_chars_space_then_ink():
    ramp = get_ramp(0.0)
    assert len(ramp) == 2
    assert ramp[0] == " "
    assert ramp[-1] != " "


def test_densest_ramp_is_long_and_ends_on_heavy_glyph():
    ramp = get_ramp(1.0)
    assert len(ramp) >= 32
    assert ramp[0] == " "
    assert ramp[-1] in "@$#B&"


def test_ramp_length_grows_monotonically_with_density():
    lengths = [len(get_ramp(d)) for d in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert lengths == sorted(lengths)
    assert len(set(lengths)) >= 3


def test_ramp_first_char_always_space():
    for d in (0.0, 0.33, 0.66, 1.0):
        assert get_ramp(d)[0] == " "


def test_density_clamped_outside_unit_interval():
    assert get_ramp(-0.2) == get_ramp(0.0)
    assert get_ramp(1.5) == get_ramp(1.0)
