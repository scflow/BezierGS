import numpy as np
import cv2


def draw_3d_box_on_img(vertices, image, color=(0, 255, 0), thickness=2):
    verts = np.asarray(vertices)
    if verts.ndim == 4 and verts.shape[-1] == 2:
        verts = verts.reshape(-1, 2)
    if verts.shape != (8, 2):
        raise ValueError(f"vertices should have shape (8, 2) or (2, 2, 2, 2), got {verts.shape}")
    verts = np.round(verts).astype(np.int32)

    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        p1 = tuple(verts[i])
        p2 = tuple(verts[j])
        cv2.line(image, p1, p2, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    return image
