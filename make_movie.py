import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from Wave2D import Wave2D_Neumann


def make_neumann_gif():
    L = 1.0
    c = 1.0
    N = 40
    mx = 2
    my = 2
    cfl = 1.0 / np.sqrt(2)

    solver = Wave2D_Neumann(L=L, c=c)
    w_val = solver.w(mx, my)
    T = 2 * np.pi / w_val

    dx = L / N
    dt = cfl * dx / c
    Nt = int(np.ceil(1.5 * T / dt))

    xij, yij, data = solver(
        N=N, Nt=Nt, cfl=cfl, c=c, mx=mx, my=my, store_data=1
    )

    fig, ax = plt.subplots(
        figsize=(4.5, 3.5), dpi=60, subplot_kw={"projection": "3d"}
    )

    frames = []
    step = 2
    time_keys = sorted(data.keys())[::step]

    for key in time_keys:
        val = data[key]
        frame = ax.plot_wireframe(
            xij, yij, val, rstride=2, cstride=2, color="tab:blue", linewidth=0.8
        )
        ax.set_zlim(-1.1, 1.1)
        ax.set_title(f"Neumann Wave (t = {key * dt:.2f}s)", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        frames.append([frame])

    ani = animation.ArtistAnimation(
        fig, frames, interval=60, blit=True, repeat_delay=500
    )

    output_path = "mandatory1/report/neumannwave.gif"
    ani.save(output_path, writer="pillow", fps=15)
    plt.close(fig)

    print(f"Movie saved to: {output_path}")


if __name__ == "__main__":
    make_neumann_gif()
