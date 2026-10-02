import numpy as np
import sympy as sp
from scipy import sparse

x, y, t = sp.symbols("x,y,t")


class Wave2D:
    """Class for solving the 2D wave equation"""

    def __init__(self, L: float = 1.0, c: float = 1.0):
        self.L = L
        self.c = c
        self.mx = 3
        self.my = 3
        self.cfl = 0.5
        self.N = 8

    def create_mesh(
        self, N: int, sparse_mesh: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh"""
        xi = np.linspace(0, self.L, N + 1)
        return np.meshgrid(xi, xi, indexing="ij", sparse=sparse_mesh)

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """
        return sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), "lil")

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return float(self.c * sp.pi * sp.sqrt(self.mx**2 + self.my**2))

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        xij, yij = self.create_mesh(N)
        return np.sin(mx * np.pi * xij / self.L) * np.sin(my * np.pi * yij / self.L)

    @property
    def dt(self) -> float:
        """Return the time step"""
        dx = self.L / self.N
        return float(self.cfl * dx / self.c)

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        N = u.shape[0] - 1
        dx = self.L / N
        xij, yij = self.create_mesh(N)

        ue_func = sp.lambdify((x, y, t), self.ue(self.mx, self.my), modules="numpy")
        ue_val = ue_func(xij, yij, t0)

        return float(np.sqrt(dx * dx * np.sum((u - ue_val)**2)))

    def apply_bcs(self, u: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """
        u[0, :] = 0.0
        u[-1, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N
        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my

        dx = self.L / N
        dt = self.dt

        Unm1 = self.initialize(N, mx, my)
        D = (self.D2(N) / (dx**2)).tocsr()

        Un = Unm1 + 0.5 * (self.c * dt)**2 * (D @ Unm1 + Unm1 @ D.T)
        self.apply_bcs(Un)

        plotdata = {0: Unm1.copy()}
        if store_data == 1:
            plotdata[1] = Un.copy()

        errors = [self.l2_error(Unm1, 0.0)]

        Unp1 = np.zeros_like(Un)
        for n in range(1, Nt):
            Unp1[:] = 2 * Un - Unm1 + (self.c * dt)**2 * (D @ Un + Un @ D.T)
            self.apply_bcs(Unp1)

            t_curr = (n + 1) * dt
            errors.append(self.l2_error(Unp1, t_curr))

            Unm1[:] = Un
            Un[:] = Unp1
            
            if store_data > 0 and (n + 1) % store_data == 0:
                plotdata[n + 1] = Un.copy()

        if store_data == -1:
            return dx, np.array(errors)
        return plotdata

    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]
        return np.array(r), np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), "lil")
        D[0, 0] = -2.0
        D[0, 1] = 2.0
        D[-1, -1] = -2.0
        D[-1, -2] = 2.0
        return D

    def ue(self, mx: int, my: int) -> sp.Expr:
        w_val = self.c * sp.pi * sp.sqrt(mx**2 + my**2)
        return sp.cos(mx * sp.pi * x / self.L) * sp.cos(my * sp.pi * y / self.L) * sp.cos(w_val * t)

    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        xij, yij = self.create_mesh(N)
        return np.cos(mx * np.pi * xij / self.L) * np.cos(my * np.pi * yij / self.L)

    def apply_bcs(self, u: np.ndarray):
        pass


def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    cfl = 1.0 / np.sqrt(2)
    mx = my = 2

    sol_dirichlet = Wave2D()
    _, err_d = sol_dirichlet(N=10, Nt=10, cfl=cfl, mx=mx, my=my, store_data=-1)
    assert np.max(err_d) < 1e-12 or err_d[-1] < 1e-12, f"Dirichlet error too high: {err_d[-1]}"

    sol_neumann = Wave2D_Neumann()
    _, err_n = sol_neumann(N=10, Nt=10, cfl=cfl, mx=mx, my=my, store_data=-1)
    assert np.max(err_n) < 1e-12 or err_n[-1] < 1e-12, f"Neumann error too high: {err_n[-1]}"
