import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from typing import Optional, Union
import pyvista as pv

class Truss2D():
    '''A universal class for 2D truss analysis using FEM with Truss elements.
   
    '''
    def __init__(
        self, 
        nodes: np.ndarray, 
        elements: np.ndarray, 
        A: float | np.ndarray, 
        E: float | np.ndarray, 
        rho: float | np.ndarray, 
        constraints: np.ndarray | None = None
    ):
        '''
        Initialization of Truss2D object.

        Parameters
        ----------
        nodes : np.ndarray, shape (n_nodes, 2)
            Array of nodal coordinates [x, y].
            * n_nodes: Number of nodes in the geometry.

        elements : np.ndarray, shape (n_ele, 2)
            Array of node index pairs [node_start, node_end] defining the elements.
            * n_ele: Number of elements in the geometry.

        A : float or np.ndarray, shape (n_ele,)
            Cross-sectional area of elements. Can be a scalar (uniform for all) 
            or a vector specifying the area for each element.

        E : float or np.ndarray, shape (n_ele,)
            Young’s modulus of elasticity. Can be a scalar (uniform for all) 
            or a vector specifying the modulus for each element.
        
        rho : float or np.ndarray, shape (n_ele,)
            Material density. Can be a scalar (uniform for all) or a vector 
            specifying the density for each element. Used for mass matrix calculation.

        constraints : np.ndarray, shape (n_constraints, n_dof), optional
            Global constraint matrix where each row represents a linear 
            boundary condition.
            * n_constraints: Total number of prescribed degrees of freedom.
            * n_dof: Total degrees of freedom in the system (n_nodes * 2).
            If None, the system is unconstrained.

        '''

        # preveri da gre za 2D in ne 3D
        if nodes.shape[1] != 2:
            raise TypeError("List of nodes must be a 2D array with shape (N, 2)")

        self.nodes = nodes
        self.elements = elements
        self.A = A
        self.E = E
        self.rho = rho
        self.type = "truss"

        # s skalarnimi parametri
        # TODO: dodaj vektorske parametre

        self.constraints = constraints if constraints is not None else np.empty((0, 2 * len(nodes)))
        self._update_solver()
    
    def display(self):
        ''' Display a truss structure according to given nodes and elements.
        '''

        for e in self.elements:
            plt.plot(self.nodes[e,0], self.nodes[e,1], '-o', c='C0')
            plt.ylim(np.min(self.nodes[:,1])-0.5, np.max(self.nodes[:,1])+0.5)
            plt.xlim(np.min(self.nodes[:,0])-0.5, np.max(self.nodes[:,0])+0.5)
    
    
    def animate_mode_shapes(self, scale=1.0):
        """
        Animacija lastnih oblik za Truss2D z uporabo Open3D.
        
        Kontrole:
          - SPACE: Start/Stop animacije
          - D / DESNA PUŠČICA: Naslednji lastni način
          - A / LEVA PUŠČICA: Prejšnji lastni način
          - Q / ESC: Izhod
        """
        import open3d as o3d
        import numpy as np
        import time

        nodes_2d = self.nodes
        n_nodes = len(nodes_2d)
        pts_orig = np.zeros((n_nodes, 3))
        pts_orig[:, :2] = nodes_2d

        model_size = np.max(np.ptp(nodes_2d, axis=0)) if np.any(nodes_2d) else 1.0
        axis_len = model_size * 0.2

        lines_indices = [[e[0], e[1]] for e in self.elements]

        line_set_orig = o3d.geometry.LineSet()
        line_set_orig.points = o3d.utility.Vector3dVector(pts_orig)
        line_set_orig.lines = o3d.utility.Vector2iVector(lines_indices)
        line_set_orig.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7] for _ in range(len(lines_indices))])


        line_set_deformed = o3d.geometry.LineSet()
        line_set_deformed.points = o3d.utility.Vector3dVector(pts_orig)
        line_set_deformed.lines = o3d.utility.Vector2iVector(lines_indices)
        line_set_deformed.colors = o3d.utility.Vector3dVector([[0.0, 0.4, 1.0] for _ in range(len(lines_indices))])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_orig)
        pcd.colors = o3d.utility.Vector3dVector([[1.0, 0.2, 0.2] for _ in range(n_nodes)])

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Truss2D - Open3D Animacija", width=1200, height=800)
        
        vis.add_geometry(line_set_orig)
        vis.add_geometry(line_set_deformed)
        vis.add_geometry(pcd)

        def get_displacement(m_idx):
            mode = self.eig_vec[:, int(m_idx)]

            disp_2d = mode.reshape(n_nodes, 2)
            disp_3d = np.zeros((n_nodes, 3))
            disp_3d[:, :2] = disp_2d
            
            max_val = np.max(np.linalg.norm(disp_3d, axis=1))
            if max_val < 1e-12: return disp_3d

            return disp_3d * (model_size * 0.15 / max_val) * scale

        state = {
            't': 0.0,
            'animate': True,
            'current_mode': 0,
            'active_disp': get_displacement(0)
        }

        def update_title():
            freq = self.eig_freq[state['current_mode']]
            print(f"\rNačin: {state['current_mode']} | Frekvenca: {freq:.2f} Hz", end="", flush=True)

        def next_mode(v):
            state['current_mode'] = (state['current_mode'] + 1) % len(self.eig_freq)
            state['active_disp'] = get_displacement(state['current_mode'])
            update_title()
            return False

        def prev_mode(v):
            state['current_mode'] = (state['current_mode'] - 1) % len(self.eig_freq)
            state['active_disp'] = get_displacement(state['current_mode'])
            update_title()
            return False

        def toggle_anim(v):
            state['animate'] = not state['animate']
            return False

        vis.register_key_callback(ord('D'), next_mode)
        vis.register_key_callback(262, next_mode) # desno puscica
        vis.register_key_callback(ord('A'), prev_mode)
        vis.register_key_callback(263, prev_mode) # levo puscica
        vis.register_key_callback(ord(' '), toggle_anim)

        render_opt = vis.get_render_option()
        render_opt.background_color = np.array([1, 1, 1])
        render_opt.point_size = 10.0
        render_opt.line_width = 8.0

        update_title()

        try:
            while True:
                if state['animate']:
                    state['t'] += 0.15
                    factor = np.sin(state['t'])
                    deformed_pts = pts_orig + state['active_disp'] * factor
                    
                    line_set_deformed.points = o3d.utility.Vector3dVector(deformed_pts)
                    pcd.points = o3d.utility.Vector3dVector(deformed_pts)
                    
                    vis.update_geometry(line_set_deformed)
                    vis.update_geometry(pcd)
                
                if not vis.poll_events():
                    break
                vis.update_renderer()
                time.sleep(0.01)
        finally:
            vis.destroy_window()
        
    
    def _update_solver(self):
        '''Method for updating matrices.'''

        M_glob = calculate_M_glob_truss(self.nodes, self.elements, self.A, self.rho)
        K_glob = calculate_K_glob_truss(self.nodes, self.elements, self.A, self.E)
        self.M_glob = M_glob
        self.K_glob = K_glob

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
        '''Open a UI to edit constraints of a Truss2D object. 

        Supports:
        - [1] Pinned support (ux=uy=uz=0)
        - [2] Roller support (set angle with slider)
        - [3] Remove constraints at node

        If there are existing constraints you can remove them or add more within this function. \n
        Compatible with passing constraint matrix when creating Truss2D object. 
          

        '''

        self.font_size = 21
        self.y_pos = 0.03  
        axis_len = np.max(np.abs(self.nodes)) * 0.25 if np.any(self.nodes) else 1.0
        self.r_size = axis_len * 0.1 # radij sfer

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

        p.add_mesh(pv.Arrow(start=(0,0,0), direction=(1,0,0), scale=axis_len, tip_radius=0.03, shaft_radius=0.01), color="red")
        p.add_mesh(pv.Arrow(start=(0,0,0), direction=(0,1,0), scale=axis_len, tip_radius=0.03, shaft_radius=0.01), color="green")


        p.add_text("[1] Pinned support", position=(0.02, self.y_pos), 
                       color='firebrick', font_size=self.font_size, name="st_1", 
                       shadow=True, viewport=True)

        p.add_text("[3] Remove constraint", position=(0.82, self.y_pos), 
                       color='#333333', font_size=self.font_size, name="st_4", 
                       shadow=True, viewport=True)

        p.view_xy()
        p.camera.zoom(0.7)

        self.last_picked_idx = None
        self.current_angle = 0 
        n_dof = 2 * len(self.nodes)

        def draw_fixed_icon(node_idx):
            p.add_mesh(pv.Sphere(radius=self.r_size, center=pts[node_idx]), 
                    color="firebrick", name=f"fixed_{node_idx}")

        def draw_angled_icon(node_idx, angle):
            p.add_mesh(pv.Sphere(radius=self.r_size, center=pts[node_idx]), 
                    color="royalblue", name=f"angled_{node_idx}")
            
           # skaliranje puscic
            arr_len = axis_len * 0.5  
            shaft_r = arr_len * 0.25
            tip_r = arr_len * 0.6
            
            # smer puscice
            phi = np.radians(angle)
            direction = np.array([np.cos(phi), np.sin(phi), 0.0])
            
            # dvosmerna
            a1 = pv.Arrow(start=pts[node_idx], direction=direction, scale=arr_len, 
                        shaft_radius=shaft_r, tip_radius=tip_r)
            a2 = pv.Arrow(start=pts[node_idx], direction=-direction, scale=arr_len, 
                        shaft_radius=shaft_r, tip_radius=tip_r)
            
            p.add_mesh(a1 + a2, color="royalblue", name=f"arrow_{node_idx}", lighting=False)

        
        if self.temp_rows:
            processed_nodes = set()
            for row in self.temp_rows:
                active_dofs = np.where(np.abs(row) > 1e-6)[0]
                if len(active_dofs) == 0: continue
                
                node_idx = active_dofs[0] // 2
                if node_idx in processed_nodes: continue
                
                base = 2 * node_idx
                node_rows = [r for r in self.temp_rows if np.abs(r[base]) > 1e-6 or np.abs(r[base+1]) > 1e-6]
                
                if len(node_rows) == 2:
                    # dve enacbi je pinned (u=0, v=0)
                    draw_fixed_icon(node_idx)
                elif len(node_rows) == 1:
                    # ena enacba je roller
                    r = node_rows[0]
                    # kot normale
                    angle_rad = np.arctan2(r[base+1], r[base])
                    angle_deg = (np.degrees(angle_rad) - 90) % 180
                    draw_angled_icon(node_idx, round(angle_deg))
                
                processed_nodes.add(node_idx)


        def slider_callback(value):
            self.current_angle = int(round(value))
            update_angle_text()
        
        def update_angle_text():
            p.remove_actor("st_2")
            p.add_text(f"[2] Roller ({self.current_angle}°)", position=(0.22, self.y_pos), 
                       color='royalblue', font_size=self.font_size, name="st_2", 
                       shadow=True, viewport=True)

        def update_status_text():
            p.remove_actor("st_node")
            p.add_text(f"Node: {self.last_picked_idx}", position=(0.02, 0.12), 
                       color='black', font_size=self.font_size+10, name="st_node", 
                       shadow=True, viewport=True)
            
            

        def pick_callback(point_data, *args):
            idx = mesh.find_closest_point(point_data)
            self.last_picked_idx = idx
            p.add_mesh(pv.Sphere(radius=0.1, center=pts[idx]), color="yellow", opacity=0.4, name="selection_glow")
            p.add_mesh(pv.Sphere(radius=0.01, center=pts[idx]), color="black", name="selection_center")
            update_status_text()

        def set_fixed():
            if self.last_picked_idx is not None:
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
                            tube_width=0.003, slider_width=0.02, fmt="{:.0f}")

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

        update_angle_text()
        p.show()

        
        if self.temp_rows:
            self.constraints = np.array(self.temp_rows)
            print("Solver has been updated.")
            self._update_solver()
        else:
            self.constraints = None
    

def calculate_K_glob_truss(nodes=np.ndarray, elements=np.ndarray, A=np.float64 or np.ndarray, E = np.float64 or np.ndarray):
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




def calculate_M_glob_truss(nodes=np.ndarray, elements=np.ndarray, A=np.float64 or np.ndarray, rho = np.float64 or np.ndarray):
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




