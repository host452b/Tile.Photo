from __future__ import annotations

_MASTER = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
_MIN_LEN = 2
_MAX_LEN = len(_MASTER)


def get_ramp(density: float) -> str:
    d = max(0.0, min(1.0, float(density)))
    length = round(_MIN_LEN + d * (_MAX_LEN - _MIN_LEN))
    if length >= _MAX_LEN:
        return _MASTER
    step = (_MAX_LEN - 1) / (length - 1)
    indices = [round(i * step) for i in range(length)]
    return "".join(_MASTER[i] for i in indices)
