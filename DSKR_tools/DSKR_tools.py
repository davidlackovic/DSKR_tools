import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from typing import Optional, Union
import pyvista as pv

class Truss2D():
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
        self.A = A
        self.E = E
        self.rho = rho

        self.constraints = constraints if constraints is not None else np.empty((0, 2 * len(nodes)))

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

        
        self._update_solver()
    
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
    
    def _update_solver(self):
        '''Method for updating matrices.'''

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
        
        print("Solver has been updated.")





    def edit_constraints(self):
        if hasattr(self, 'constraints') and self.constraints is not None:
            self.temp_rows = self.constraints.tolist()
        else:
            self.temp_rows = []

        pv.set_jupyter_backend(None)
        p = pv.Plotter(notebook=False, title="Truss2D - Edit Constraints")
        p.set_background("white")
        p.enable_parallel_projection()

        pts = np.column_stack((self.nodes, np.zeros(len(self.nodes))))
        cells = np.hstack([[2, e[0], e[1]] for e in self.elements])
        mesh = pv.UnstructuredGrid(cells, [pv.CellType.LINE]*len(self.elements), pts)

        p.add_mesh(mesh, color="#555555", line_width=2, render_lines_as_tubes=True)
        p.add_point_labels(pts, [f"{i}" for i in range(len(self.nodes))], 
                          point_size=10, font_size=18, text_color="black", 
                          always_visible=True, name="node_labels", shadow=True)

        axis_len = np.max(np.abs(self.nodes)) * 0.25 if np.any(self.nodes) else 1.0
        p.add_mesh(pv.Arrow(start=(0,0,0), direction=(1,0,0), scale=axis_len, tip_radius=0.03, shaft_radius=0.01), color="red")
        p.add_mesh(pv.Arrow(start=(0,0,0), direction=(0,1,0), scale=axis_len, tip_radius=0.03, shaft_radius=0.01), color="green")

        p.view_xy()
        p.camera.zoom(0.7)

        self.last_picked_idx = None
        self.current_angle = 0 
        n_dof = 2 * len(self.nodes)

        # --- POMOŽNE FUNKCIJE ZA IZRIS ---

        def draw_fixed_icon(node_idx):
            p.add_mesh(pv.Sphere(radius=0.2, center=pts[node_idx]), 
                       color="firebrick", name=f"fixed_{node_idx}")

        def draw_angled_icon(node_idx, angle):
            p.add_mesh(pv.Sphere(radius=0.2, center=pts[node_idx]), 
                       color="royalblue", name=f"angled_{node_idx}")
            offset_pos = pts[node_idx] + [axis_len*0.05, axis_len*0.05, 0]
            p.add_point_labels([offset_pos], [f"{int(angle)}°"], 
                              font_size=25, text_color="royalblue", 
                              name=f"angle_label_{node_idx}", 
                              shape=None, always_visible=True, shadow=True)

        # 2. AVTOMATSKI IZRIS OBSTOJEČIH CONSTRAINT-OV
        if self.temp_rows:
            processed_nodes = set()
            for row in self.temp_rows:
                # Najdemo kje v vrstici so vrednosti (DOF-i)
                active_dofs = np.where(np.abs(row) > 1e-6)[0]
                if len(active_dofs) == 0: continue
                
                node_idx = active_dofs[0] // 2
                if node_idx in processed_nodes: continue
                
                # Preverimo če je fiksno (dve vrstici na vozlišče običajno pomenita fiksno)
                # Tu preverimo specifično vrstico: če sta oba DOF-a 1, je fiksno
                # Bolj varna metoda: če je v vrstici samo en DOF aktiven in je to X ali Y
                if np.abs(row[2*node_idx]) > 0.99 and np.abs(row[2*node_idx+1]) < 0.01:
                    # Verjetno fiksno (prenaša X), preverimo če obstaja še vrstica za Y
                    draw_fixed_icon(node_idx)
                    processed_nodes.add(node_idx)
                else:
                    # Izračunamo kot iz vrednosti v matriki: tan(phi + 90) = row_y / row_x
                    # phi_val = atan2(row_y, row_x) - pi/2
                    angle_rad = np.arctan2(row[2*node_idx+1], row[2*node_idx]) - np.pi/2
                    angle_deg = np.degrees(angle_rad) % 180
                    draw_angled_icon(node_idx, round(angle_deg))
                    processed_nodes.add(node_idx)

        # --- INTERAKTIVNE FUNKCIJE ---

        def slider_callback(value):
            self.current_angle = int(round(value))
            if self.last_picked_idx is not None:
                update_status_text()

        def update_status_text():
            status = (f"Node: {self.last_picked_idx}\n"
                      f"[1] Pinned support     [2] Roller support at angle {self.current_angle}°         [3] Remove constraint")
            p.add_text(status, position='lower_left', color='black', font_size=19, name="status_text", shadow=True)

        def pick_callback(point_data, *args):
            idx = mesh.find_closest_point(point_data)
            self.last_picked_idx = idx
            p.add_mesh(pv.Sphere(radius=0.1, center=pts[idx]), color="yellow", opacity=0.4, name="selection_glow")
            p.add_mesh(pv.Sphere(radius=0.01, center=pts[idx]), color="black", name="selection_center")
            update_status_text()

        def set_fixed():
            if self.last_picked_idx is not None:
                # Najprej počistimo stare pogoje na tem vozlišču
                remove_at_node()
                r1, r2 = np.zeros(n_dof), np.zeros(n_dof)
                r1[2*self.last_picked_idx], r2[2*self.last_picked_idx+1] = 1, 1
                self.temp_rows.extend([r1, r2])
                draw_fixed_icon(self.last_picked_idx)

        def set_angled():
            if self.last_picked_idx is not None:
                remove_at_node()
                phi = np.radians(self.current_angle)
                r = np.zeros(n_dof)
                r[2*self.last_picked_idx] = np.cos(phi + np.pi/2)
                r[2*self.last_picked_idx + 1] = np.sin(phi + np.pi/2)
                self.temp_rows.append(r)
                draw_angled_icon(self.last_picked_idx, self.current_angle)
        
        def remove_at_node():
            if self.last_picked_idx is not None:
                idx = self.last_picked_idx
                dof1, dof2 = 2*idx, 2*idx + 1
                self.temp_rows = [r for r in self.temp_rows if np.abs(r[dof1]) < 1e-6 and np.abs(r[dof2]) < 1e-6]
                p.remove_actor(f"fixed_{idx}")
                p.remove_actor(f"angled_{idx}")
                p.remove_actor(f"angle_label_{idx}")

        p.add_slider_widget(callback=slider_callback, rng=[0, 180], value=0,
                            pointa=(0.6, 0.9), pointb=(0.9, 0.9), style='modern', color="black",
                            tube_width=0.003, slider_width=0.02)

        p.add_text("Right click to select a node", position='upper_left', 
                   font_size=12, color='black', name="instruction_text")

        
        p.enable_point_picking(
            callback=pick_callback, 
            show_message=False,
            left_clicking=False
        )
        p.add_key_event('1', set_fixed)
        p.add_key_event('2', set_angled)
        p.add_key_event('3', remove_at_node)

        p.show()

        # Shranjevanje in posodobitev ob zaprtju
        if self.temp_rows:
            self.constraints = np.array(self.temp_rows)
            print("Solver has been updated.")
            self._update_solver()
        else:
            self.constraints = None
    

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
    
    ax.set_xlim(np.min(nodes[:,0])-1.5, np.max(nodes[:,0])+1.5)
    ax.set_ylim(np.min(nodes[:,1])-1.5, np.max(nodes[:,1])+1.5)

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