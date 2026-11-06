"""Pure palette shared by masks, point clouds and Blender diagnostics."""
import numpy as np
PALETTE = np.array([[0, 0, 0], [242, 98, 70], [57, 178, 228], [113, 185, 76],
                    [180, 110, 222], [242, 181, 53], [242, 101, 165], [68, 211, 190],
                    [157, 156, 240], [204, 136, 86], [164, 194, 59]], np.uint8)


def color(index):
    return tuple(int(x) for x in PALETTE[(index - 1) % (len(PALETTE) - 1) + 1])
