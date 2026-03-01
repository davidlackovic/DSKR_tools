import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from typing import Optional, Union

class Truss():
    '''A universal class for 2D truss analysis using FEM.
   
    '''
    def __init__(self, nodes: np.ndarray, elements: np.ndarray , A: np.float64 | np.ndarray, E:np.float64 | np.ndarray, rho: np.float64 | np.ndarray,  constraints: np.ndarray | None = None):
        '''
        Initialization of Truss object.

        Parameters
        ----------
        nodes: np.ndarray, shape (n_vozlisc, 2) 
            list of coordinates of nodes

        elements: np.ndarray, shape (n_ele, 2)
            list of node index pairs defining the elements

        A: float or np.ndarray, shape (n_ele)
            cross-sectional area of elements (can be a vector of areas for every element or a scalar)

        E: float or np.ndarray, shape (n_ele)
            Young’s modulus of elements (can be a vector of moduli for every element or a scalar)
        
        rho: float or np.ndarray, shape (n_ele)
            material density (can be a vector of densities for every element or a scalar)

        constraints: np.ndarray or None, shape (n_constraints, n_constraints)
            matrix of constraints or None if there are no constraints
        
            

        Example
        -------
        >>> A = 100.e-6 # m^2
        >>> rho = 7850. # kg/m^3
        >>> E = 2.e11 # Pa
        >>> L0 = 1. # m

        >>> nodes = np.array([[0,0], 
                              [L0,0], 
                              [0, L0],])

        >>> elements = np.array([[0,1],
                                 [1,2],
                                 [0,2],])


        >>> phi = np.pi/4
        >>> constraints = np.zeros((3,6))
        >>> constraints[0,0]=1
        >>> constraints[1,1]=1
        >>> constraints[2,2]=np.sin(phi)
        >>> constraints[2,3]=-np.cos(phi)

        
        >>> truss = funkcije_DSKR.Truss(nodes, elements, A, E, rho, constraints) 
        '''

        self.nodes = nodes
        self.elements = elements
        self.constraints = constraints
        self.A = A
        self.E = E
        self.rho = rho

        M_glob = calculate_M_glob(self.nodes, self.elements, self.A, self.rho)
        K_glob = calculate_K_glob(self.nodes, self.elements, self.A, self.E)

        
        if self.constraints is not None:
            L = sp.linalg.null_space(self.constraints)

            M_glob_constrained = L.T @ M_glob @ L
            K_glob_constrained = L.T @ K_glob @ L

            self.eig_val, self.eig_vec = sp.linalg.eigh(K_glob_constrained, M_glob_constrained)
            self.eig_vec=L@self.eig_vec
            self.eig_val = np.abs(self.eig_val)
        
        else:
            self.eig_val, self.eig_vec = sp.linalg.eigh(K_glob, M_glob)
            self.eig_val = np.abs(self.eig_val)

        eig_freq = np.sqrt(self.eig_val) / 2 / np.pi
        self.eig_freq = eig_freq.round(3)

        
        
    
    def display_truss(self):
        display_truss(self.nodes, self.elements)
    
    def animate_mode_shapes(self, scale: float = 0.1):
        ''' Animate mode shapes of a truss object.


        Parameters
        ----------
        scale: float
            scaling factor for animation
        '''
        self.anim = animate_mode_shapes(self.nodes, self.elements, self.eig_vec, scale=scale)
        return self.anim


    

def calculate_K_glob(nodes=np.ndarray, elements=np.ndarray, A=np.float64 or np.ndarray, E = np.float64 or np.ndarray):
    '''Compute the global stiffness matrix K_glob for the given mesh of elements.
     
    Parameters
    ----------

    nodes: np.ndarray, shape (n_vozlisc, 2) 
        list of coordinates of nodes

    elements: np.ndarray, shape (n_ele, 2)
        list of node index pairs defining the elements


    A: float
        cross-sectional area (can be a vector of areas for every element or a scalar)

    E: float
        Young’s modulus (can be a vector of moduli for every element or a scalar)
    
        

    Returns
    ----------

    K_glob: np.ndarray, shape (2 * n_vozlisc, 2 * n_vozlisc)
        global stiffness matrix of the truss system.


    '''
    ndim = 2
    n_ps = ndim * len(nodes)
    n_ele = len(elements)

    K_glob = np.zeros((n_ps, n_ps))

    diffs = nodes[elements[:,1]] - nodes[elements[:,0]]
    Le = np.hypot(diffs[:, 0], diffs[:, 1])
    c = diffs[:, 0] / Le  
    s = diffs[:, 1] / Le  

    k_const = (E * A / Le) 

    cc = c*c; ss = s*s; cs = c*s
    R = np.array([[cc,  cs, -cc, -cs],
                [cs,  ss, -cs, -ss],
                [-cc, -cs, cc,  cs],
                [-cs, -ss, cs,  ss]])
    
    K_vsi = (R*k_const.reshape(1, 1, n_ele)).T
        

    indeksi = (elements.flatten()[:, np.newaxis] * 2 + [0, 1]).flatten().reshape(-1, 4)
    rows = indeksi[:, :, np.newaxis]
    cols = indeksi[:, np.newaxis, :]

    np.add.at(K_glob, (rows, cols), K_vsi)

    return K_glob




def calculate_M_glob(nodes=np.ndarray, elements=np.ndarray, A=np.float64 or np.ndarray, rho = np.float64 or np.ndarray):
    '''Compute the global mass matrix M_glob for the given mesh of elements.

    Parameters
    ----------

    nodes: np.ndarray, shape (n_vozlisc, 2) 
        list of coordinates of nodes

    elements: np.ndarray, shape (n_ele, 2)
        list of node index pairs defining the elements

    A: float
        cross-sectional area (can be a vector or a scalar)

    rho: float
        material density (can be a vector or a scalar)
        
    Returns
    ----------

    M_glob: np.ndarray, shape (2 * n_vozlisc, 2 * n_vozlisc)
        Global mass matrix of the truss system.



    '''

    ndim = 2
    n_ps = ndim * len(nodes)


    M_glob = np.zeros((n_ps, n_ps))

    diffs = nodes[elements[:,1]] - nodes[elements[:,0]]
    Le = np.hypot(diffs[:, 0], diffs[:, 1])
    
    m_konst = (rho*A*Le/6).reshape(-1, 1, 1)
    
    R = np.array([[2, 0, 1, 0],
                [0, 2, 0, 1],
                [1, 0, 2, 0],
                [0, 1, 0, 2]])
    
    M_vsi = R*m_konst

    indeksi = (elements.flatten()[:, np.newaxis] * 2 + [0, 1]).flatten().reshape(-1, 4)
    rows = indeksi[:, :, np.newaxis]
    cols = indeksi[:, np.newaxis, :]

    np.add.at(M_glob, (rows, cols), M_vsi)

    return M_glob

def display_truss(nodes=np.ndarray, elements=np.ndarray):
    ''' Display a truss structure according to given nodes and elements.
    
    Parameters
    ----------
    nodes: np.ndarray, shape (n_vozlisc, 2) 
        list of coordinates of nodes

    elements: np.ndarray, shape (n_ele, 2)
        list of node index pairs defining the elements 
    '''
    # TODO: add arrows showing degrees of freedom
    for e in elements:
        plt.plot(nodes[e,0], nodes[e,1], '-o', c='C0')
        plt.ylim(np.min(nodes[:,1])-0.5, np.max(nodes[:,1])+0.5)
        plt.xlim(np.min(nodes[:,0])-0.5, np.max(nodes[:,0])+0.5)


def animate_mode_shapes(nodes=np.ndarray, elements=np.ndarray, eig_vec=np.ndarray, scale=0.1):
    ''' Animate mode shapes of a truss structure according to eigenvectors of the truss.
    
    
        Parameters
        ----------
        nodes: np.ndarray, shape (n_vozlisc, 2) 
            list of coordinates of nodes

        elements: np.ndarray, shape (n_ele, 2)
            list of node index pairs defining the elements 

        eig_vec: np.ndarray, shape()
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Slider

    n_modes = eig_vec.shape[1]
    
    state = {
        'U': eig_vec[:, 0].reshape(len(nodes), 2)
    }

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.set_aspect('equal')
    
    ax.set_xlim(np.min(nodes[:,0])-0.5, np.max(nodes[:,0])+0.5)
    ax.set_ylim(np.min(nodes[:,1])-0.5, np.max(nodes[:,1])+0.5)

    lines = [ax.plot([], [], '-o', c='C0')[0] for _ in elements]

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Lastni način', 0, n_modes - 1, valinit=0, valstep=1)

    def update_mode(val):
        mode_idx = int(val)

        state['U'] = eig_vec[:, mode_idx].reshape(len(nodes), 2)
        ax.set_title(f"Lastni način: {mode_idx}")

    slider.on_changed(update_mode)

    def func(frame):
        t = frame
        displacement = scale * state['U'] * np.sin(t)
        trenutna_voz = nodes + displacement
        
        for i, e in enumerate(elements):
            lines[i].set_data(trenutna_voz[e, 0], trenutna_voz[e, 1])
        
        return lines

    anim = FuncAnimation(fig, func, frames=np.linspace(0, 2*np.pi, 60), 
                        blit=True, interval=20, repeat=True)

    fig.canvas.manager.slider = slider 

    plt.show()
    return anim