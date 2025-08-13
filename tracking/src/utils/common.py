def sec_to_timestamp(sec: float) -> str:
    s = int(sec)
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"