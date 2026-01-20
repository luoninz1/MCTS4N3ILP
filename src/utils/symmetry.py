def get_d4_orbit(action, n, symmetry_mode):
    """
    Generate all valid (row, col) pairs corresponding to the 'symmetry_mode' string.
    The string is split by '_then_', and each part is treated as a generator applied
    to the current set of points.
    """
    row = action // n
    col = action % n
    points = {(row, col)}
    
    if not symmetry_mode:
        return points

    # Available transformations
    transforms = {
        'horizontal_flip':      lambda r, c: (r, n - 1 - c),          # Reflect across vertical axis |
        'vertical_flip':        lambda r, c: (n - 1 - r, c),          # Reflect across horizontal axis -
        'diagonal_flip':        lambda r, c: (c, r),                  # Reflect across main diagonal \
        'anti_diagonal_flip':   lambda r, c: (n - 1 - c, n - 1 - r),  # Reflect across anti-diagonal /
        'rotation_90':          lambda r, c: (c, n - 1 - r),          # 90 deg clockwise
        'rotation_180':         lambda r, c: (n - 1 - r, n - 1 - c),  # 180 deg
        'rotation_270':         lambda r, c: (n - 1 - c, r)           # 270 deg clockwise
    }
    
    # Parse operations sequence (e.g., "horizontal_flip_then_vertical_flip")
    operations = symmetry_mode.split('_then_')
    
    for op in operations:
        if op in transforms:
            func = transforms[op]
            # Apply this transform to all currently found points and extend the set
            new_points = set()
            for (r, c) in points:
                new_points.add(func(r, c))
            points.update(new_points)
        else:
            raise ValueError(f"Unknown symmetry operation: {op}")
            
    return points