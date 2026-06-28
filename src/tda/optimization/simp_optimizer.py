import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from ripser import ripser

class SimpTda2DOptimizer:
    """SIMP 2D Topology Optimizer with Topological Data Analysis (TDA) post-processing.

    This class handles the Solid Isotropic Material with Penalty (SIMP) method for
    topology optimization on a 2D cantilever beam. It computes finite element analysis,
    optimizes density distribution, and extracts Betti numbers using persistence diagrams.
    """
    def __init__(self, nelx=60, nely=30, volfrac=0.5, penal=3.0, rmin=1.5):
        """Initializes the SimpTda2DOptimizer with grid and optimization parameters.

        Args:
            nelx (int): Number of elements in the horizontal direction. Defaults to 60.
            nely (int): Number of elements in the vertical direction. Defaults to 30.
            volfrac (float): Target volume fraction. Defaults to 0.5.
            penal (float): Penalty parameter for SIMP. Defaults to 3.0.
            rmin (float): Filter radius. Defaults to 1.5.
        """
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.tol_c = 1e-4
        self.tol_x = 1e-3
        self.min_iter = 10 
        self.max_iter = 100
        self.E0 = 1.0
        self.Emin = 1e-9
        self.nu = 0.3
        self.c_base = 0
        self._setup_fem()

    def _setup_fem(self):
        """Sets up the element stiffness matrix and loads boundary conditions for FEM.

        This method prepares the global stiffness matrix indices and the filter matrix H
        based on the grid dimensions and filter radius.
        """
        k = np.array([1/2-self.nu/6, 1/8+self.nu/8, -1/4-self.nu/12, -1/8+3*self.nu/8, 
                      -1/4+self.nu/12, -1/8-self.nu/8, self.nu/6, 1/8-3*self.nu/8])
        self.KE = self.E0/(1-self.nu**2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
        ])
        
        self.ndof = 2*(self.nelx+1)*(self.nely+1)
        self.edofMat = np.zeros((self.nelx*self.nely, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx*self.nely
                n1 = (self.nely+1)*elx + ely
                n2 = (self.nely+1)*(elx+1) + ely
                self.edofMat[el, :] = np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1])
                
        self.iK = np.kron(self.edofMat, np.ones((8,1))).flatten()
        self.jK = np.kron(self.edofMat, np.ones((1,8))).flatten()
        
        iH = np.ones(self.nelx*self.nely*(int(np.ceil(self.rmin)-1)*2+1)**2)
        jH = np.ones(iH.shape)
        sH = np.zeros(iH.shape)
        k_idx = 0
        for i1 in range(self.nelx):
            for j1 in range(self.nely):
                e1 = i1*self.nely + j1
                for i2 in range(max(i1-(int(np.ceil(self.rmin))-1),0), min(i1+(int(np.ceil(self.rmin))),self.nelx)):
                    for j2 in range(max(j1-(int(np.ceil(self.rmin))-1),0), min(j1+(int(np.ceil(self.rmin))),self.nely)):
                        e2 = i2*self.nely + j2
                        iH[k_idx] = e1
                        jH[k_idx] = e2
                        sH[k_idx] = max(0, self.rmin - np.sqrt((i1-i2)**2 + (j1-j2)**2))
                        k_idx += 1
        self.H = coo_matrix((sH, (iH, jH)), shape=(self.nelx*self.nely, self.nelx*self.nely)).tocsc()
        self.Hs = self.H.sum(1)
        
        self.F = np.zeros((self.ndof, 1))
        # Ajuste Caso 2 (Viga Voladizo): Carga puntual en el extremo libre (derecha, centro/abajo)
        self.F[2*(self.nelx+1)*(self.nely+1)-2*self.nely-1, 0] = -1.0 
        
        # Ajuste Caso 2 (Viga Voladizo): Borde izquierdo completamente empotrado
        self.fixeddofs = np.arange(0, 2*(self.nely+1))
        self.freedofs = np.setdiff1d(np.arange(self.ndof), self.fixeddofs)

    def _solve_fem(self, rho):
        """Solves the Finite Element Analysis (FEA) for the given material density.

        Args:
            rho (numpy.ndarray): 2D array of element densities.

        Returns:
            numpy.ndarray: Displacement vector U.
        """
        sK = ((rho.flatten(order='F')**self.penal * (self.E0 - self.Emin) + self.Emin)[:, np.newaxis] * self.KE.flatten()).flatten()
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
        K_free = K[self.freedofs, :][:, self.freedofs]
        
        U = np.zeros((self.ndof, 1))
        U[self.freedofs, 0] = spsolve(K_free, self.F[self.freedofs, 0])
        return U

    def run_optimization(self, callback=None):
        """Runs the topology optimization using the SIMP method and calculates topological persistence.

        Args:
            callback (callable, optional): Callback function invoked at each iteration with signature
                callback(loop, xPhys, compliance, compliance_reduction_pct, max_iter). Defaults to None.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The optimized physical density distribution (xPhys).
                - list: Ripser persistence diagrams.
                - int: Estimated Betti-1 number.
                - float: Final optimized compliance value.
                - float: Reduction percentage of compliance from the baseline.
        """
        rho_base = self.volfrac * np.ones((self.nely, self.nelx))
        U_base = self._solve_fem(rho_base)
        ce_base = (np.dot(U_base[self.edofMat].reshape(self.nelx*self.nely, 8), self.KE) * U_base[self.edofMat].reshape(self.nelx*self.nely, 8)).sum(1)
        self.c_base = (rho_base.flatten(order='F')**self.penal * ce_base).sum()

        x = self.volfrac * np.ones((self.nely, self.nelx))
        xPhys = x.copy()
        
        loop = 0
        c_hist = []
        
        while loop < self.max_iter:
            loop += 1
            U = self._solve_fem(xPhys)
            
            ce = (np.dot(U[self.edofMat].reshape(self.nelx*self.nely, 8), self.KE) * U[self.edofMat].reshape(self.nelx*self.nely, 8)).sum(1)
            c = (xPhys.flatten(order='F')**self.penal * ce).sum()
            dc = -self.penal * xPhys.flatten(order='F')**(self.penal-1) * ce
            dc = np.asarray((self.H * (x.flatten(order='F') * dc)) / self.Hs)[:, 0] / np.maximum(1e-3, x.flatten(order='F'))
            
            l1, l2 = 0.0, 1e9
            move = 0.2
            xnew = np.zeros(self.nelx*self.nely)
            
            while (l2 - l1) > 1e-4:
                lmid = 0.5 * (l2 + l1)
                x_bisection = x.flatten(order='F') * np.sqrt(-dc / lmid)
                xnew = np.maximum(0.0, np.maximum(x.flatten(order='F')-move, 
                       np.minimum(1.0, np.minimum(x.flatten(order='F')+move, x_bisection))))
                if np.sum(xnew) - self.volfrac*self.nelx*self.nely > 0:
                    l1 = lmid
                else:
                    l2 = lmid
                    
            change_x = np.max(np.abs(xnew - x.flatten(order='F')))
            x = xnew.reshape((self.nely, self.nelx), order='F')
            xPhys = x
            c_hist.append(c)
            
            rel_c = abs(c_hist[-1] - c_hist[-2]) / c_hist[-2] if loop > 1 else 0.0
            
            # Llamar al callback si existe (para UI o logging)
            if callback:
                reduccion = ((self.c_base - c) / self.c_base) * 100
                callback(loop, xPhys, c, reduccion, self.max_iter)
            
            if loop > 1 and change_x < self.tol_x and rel_c < self.tol_c and loop >= self.min_iter:
                break
                
        y_coords, x_coords = np.where(xPhys > 0.5)
        nube_puntos = np.column_stack((x_coords, -y_coords))
        resultado_tda = ripser(nube_puntos, maxdim=1, thresh=2.0)
        
        H1 = resultado_tda['dgms'][1]
        betti_1 = sum(1 for v in (H1[:, 1] - H1[:, 0]) if np.isfinite(v) and v > 1.5) if len(H1) > 0 else 0
        reduccion_pct = ((self.c_base - c) / self.c_base) * 100
        
        return xPhys, resultado_tda['dgms'], betti_1, c, reduccion_pct
