import numpy as np

def test_k(E, nu, dx, dy, t):
    prefactor = E / (1.0 - nu**2)
    D = prefactor * np.array([
        [1.0,  nu,           0.0        ],
        [nu,   1.0,          0.0        ],
        [0.0,  0.0, (1.0 - nu) / 2.0   ]
    ])

    g = 1.0 / np.sqrt(3.0)
    gauss = [(-g, -g), (g, -g), (g, g), (-g, g)]
    w = 1.0

    xn = np.array([-dx/2,  dx/2,  dx/2, -dx/2])
    yn = np.array([-dy/2, -dy/2,  dy/2,  dy/2])

    K0 = np.zeros((8, 8))

    for (xi, eta) in gauss:
        dNdxi = 0.25 * np.array([-(1 - eta),  (1 - eta), (1 + eta), -(1 + eta)])
        dNdeta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi),  (1 - xi)])

        J = np.array([
            [dNdxi @ xn,  dNdxi @ yn],
            [dNdeta @ xn, dNdeta @ yn]
        ])
        Ji = np.linalg.inv(J)
        detJ = np.linalg.det(J)

        dNdx = Ji[0, 0] * dNdxi + Ji[0, 1] * dNdeta
        dNdy = Ji[1, 0] * dNdxi + Ji[1, 1] * dNdeta

        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i] = dNdx[i]
            B[1, 2*i+1] = dNdy[i]
            B[2, 2*i] = dNdy[i]
            B[2, 2*i+1] = dNdx[i]

        K0 += B.T @ D @ B * detJ * t * w
    return K0

nex = 60
ney = 30
Lx = 120.0
Ly = 40.0
dx = Lx / nex
dy = Ly / ney
t = 1.0
E = 200000.0  # MPa = N/mm^2
nu = 0.3

K0 = test_k(E, nu, dx, dy, t)

# Build a small global system to check the initial compliance if rho = 0.5
from tda.core.fem import ensamblar_K_global, resolver_FEM
import scipy.sparse as sp

nnx = nex + 1
nny = ney + 1
n_dof = 2 * nnx * nny

idx = np.arange(nex * ney)
ey = idx // nex
ex = idx % nex
n1 = ey * nnx + ex
n2 = n1 + 1
n3 = (ey + 1) * nnx + ex + 1
n4 = (ey + 1) * nnx + ex

DOFS = np.stack([2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1], axis=1)

rho = np.full(nex * ney, 0.5)
K_global = ensamblar_K_global(rho, DOFS, n_dof, K0, 3)

F = np.zeros(n_dof)
node_load = (ney // 2) * nnx + nex
F[2 * node_load + 1] = -1000.0  # -1 kN in N

fixed = []
for j in range(nny):
    fixed.extend([2 * j * nnx, 2 * j * nnx + 1])

U = resolver_FEM(K_global, F, fixed, n_dof)

c = F.T @ U
print(f"Compliance inicial (fV=0.5, p=3 sin optimizar): {c} N.mm")

# Let's run a full optimization to see the final compliance
from tda.optimization.metric_simp import MetricaTDA_SIMP
import types

# Monkey patch calcular_K_elemental in tda.core.fem temporarily
import tda.core.fem
tda.core.fem.calcular_K_elemental = lambda E, nu: test_k(E, nu, dx, dy, t)

opt = MetricaTDA_SIMP(nex=nex, ney=ney, f_V=0.5, p=3, r_min=2.4, alpha=0.012, max_iter=200)
opt.K0 = test_k(E, nu, dx, dy, t) # override
opt.definir_problema(F, fixed)
opt.optimizar(verbose=False)
print(f"Compliance final optimizada: {opt.c_final} N.mm")
