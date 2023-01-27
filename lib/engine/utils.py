import taichi as ti

@ti.func
def cofactor(F):
    if ti.static(F.n == 2):
        return ti.Matrix([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]])
    else:
        return ti.Matrix([[F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1], F[1, 2] * F[2, 0] - F[1, 0] * F[2, 2], F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]],
                          [F[0, 2] * F[2, 1] - F[0, 1] * F[2, 2], F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0], F[0, 1] * F[2, 0] - F[0, 0] * F[2, 1]],
                          [F[0, 1] * F[1, 2] - F[0, 2] * F[1, 1], F[0, 2] * F[1, 0] - F[0, 0] * F[1, 2], F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]]])