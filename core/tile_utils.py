def get_closest_level_for_mpp(reader, target_mpp: float) -> int:
    """
    Given a reader and a target microns-per-pixel (MPP), returns the closest level.
    Assumes reader has a method get_mpp_for_level(level) -> float
    """
    min_diff = float('inf')
    closest_level = 0
    for level in range(reader.get_level_count()):
        level_mpp = reader.get_mpp_for_level(level)
        diff = abs(level_mpp - target_mpp)
        if diff < min_diff:
            min_diff = diff
            closest_level = level
    return closest_level