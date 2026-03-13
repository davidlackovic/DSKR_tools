import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from typing import Optional, Union
import pyvista as pv

class Frame2D():
    '''A universal class for 2D truss analysis using FEM with Frame elements.
   
    '''
    def __init__(
        self, 
        nodes: np.ndarray, 
        elements: np.ndarray, 
        A: float | np.ndarray, 
        E: float | np.ndarray, 
        I: float | np.ndarray, 
        rho: float | np.ndarray, 
        constraints: np.ndarray | None = None, 
        n_mesh: int | None = None
    ):
        '''
        Initialization of Frame2D object.

        Parameters
        ----------
        nodes : np.ndarray, shape (n_nodes, 2)
            Array of nodal coordinates [x, y]. 
            * n_nodes: Number of nodes in the original geometry.

        elements : np.ndarray, shape (n_ele, 2)
            Array of node index pairs [node_start, node_end] defining the elements.
            * n_ele: Number of elements in the original geometry.

        A : float or np.ndarray, shape (n_ele,)
            Cross-sectional area of elements. Can be a scalar (uniform for all) 
            or a vector specifying the area for each element.

        E : float or np.ndarray, shape (n_ele,)
            Young’s modulus of elasticity. Can be a scalar (uniform for all) 
            or a vector specifying the modulus for each element.

        I : float or np.ndarray, shape (n_ele,)
            Second moment of area (bending moment of inertia). Can be a scalar 
            (uniform for all) or a vector specifying the inertia for each element.
        
        rho : float or np.ndarray, shape (n_ele,)
            Material density. Can be a scalar (uniform for all) or a vector 
            specifying the density for each element. Used for mass matrix calculation.

        constraints : np.ndarray, shape (n_constraints, n_dof), optional
            Global constraint matrix where each row represents a linear 
            boundary condition (e.g., u=0, v=0, phi=0, or inclined supports).
            * n_constraints: Total number of prescribed degrees of freedom.
            * n_dof: Total degrees of freedom in the system (nodes * 3).
            If provided, this matrix enforces essential boundary conditions.
        
        n_mesh : int, optional
            Subdivision factor for elements. Each original element is divided 
            into `n_mesh` equal segments. 
            * If `n_mesh > 1`: New internal nodes and elements are generated, 
              allowing for higher resolution of displacement and internal forces 
              (crucial for capturing buckling modes and bending shapes).
            * If `None` or 1: The original geometry is used (point-to-point).

            **Note**: Increasing n_mesh improves accuracy and visualization of 
            deformations but increases the size of the global stiffness matrix 
            and total computation time.
        '''

        self.nodes = nodes
        self.elements = elements
        self.A = A
        self.E = E
        self.I = I
        self.rho = rho

        # TODO fix converting E, A, I, rho to vectors

        if n_mesh is not None:
            n_segments = n_mesh  # vsak element razdelimo na n pod-elementov
            all_nodes = list(self.nodes)
            all_elements = []

            def get_node_idx(point):
                """Poišče indeks vozlišča ali doda novo, če še ne obstaja."""
                for i, n in enumerate(all_nodes):
                    if np.allclose(n, point, atol=1e-6):
                        return i
                all_nodes.append(point)
                return len(all_nodes) - 1

            # generiramo gostejso mrezo
            for start_idx, end_idx in elements:
                p1 = nodes[start_idx]
                p2 = nodes[end_idx]
                
                # interpolacija vmesnih tock
                current_start_idx = start_idx
                for i in range(1, n_segments + 1):
                    t = i / n_segments
                    new_point = p1 + t * (p2 - p1)
                    
                    current_end_idx = get_node_idx(new_point)
                    all_elements.append([current_start_idx, current_end_idx])
                    current_start_idx = current_end_idx

            # izhodni podatki mreze
            self.nodes = np.array(all_nodes)
            self.elements = np.array(all_elements)

            print(f"Generated a mesh with {len(self.nodes)} nodes and {len(self.elements)} elements.")

        n_dof = 3 * len(self.nodes)
        self.constraints = constraints if constraints is not None else np.empty((0, n_dof))
        self._update_solver()
    
    def display(self):
        ''' Display a truss structure according to given nodes and elements in Frame2D object.
    
        Parameters
        ----------
        nodes: np.ndarray, shape (n_nodes, 2) 
            list of coordinates of nodes

        elements: np.ndarray, shape (n_ele, 2)
            list of node index pairs defining the elements 
        '''
        # TODO: add arrows showing degrees of freedom
        for e in self.elements:
            plt.plot(self.nodes[e,0], self.nodes[e,1], '-o', c='C0')
            plt.ylim(np.min(self.nodes[:,1])-0.5, np.max(self.nodes[:,1])+0.5)
            plt.xlim(np.min(self.nodes[:,0])-0.5, np.max(self.nodes[:,0])+0.5)
    
    
    def animate_mode_shapes(self, scale=1.0):
        """
        Animates mode shapes of a truss structure according to eigenvectors of the truss.

        This function creates an interactive matplotlib window with an oscillation 
        animation and a slider to switch between different vibration modes.

        Parameters
        ----------
        scale : float, optional
            Scaling factor for displacements to enhance visualization. Default is 0.1.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            The animation object. A reference to this object must be kept 
            (e.g., by assigning it to a variable), otherwise Python's garbage 
            collector will delete it and the animation will stop.
        """
        import open3d as o3d
        import numpy as np
        import time

        # 1. PRIPRAVA 3D PODATKOV IZ 2D MODELA
        # Vozlišča razširimo v 3D (dodamo z=0)
        pts_orig = np.zeros((len(self.nodes), 3))
        pts_orig[:, :2] = self.nodes
        
        lines = [[e[0], e[1]] for e in self.elements]
        
        # 2. GEOMETRIJA
        # Nedeformirana mreža (siva)
        line_set_orig = o3d.geometry.LineSet()
        line_set_orig.points = o3d.utility.Vector3dVector(pts_orig)
        line_set_orig.lines = o3d.utility.Vector2iVector(lines)
        line_set_orig.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7] for _ in range(len(lines))])
        
        # Deformirana mreža (modra)
        line_set_deformed = o3d.geometry.LineSet()
        line_set_deformed.points = o3d.utility.Vector3dVector(pts_orig)
        line_set_deformed.lines = o3d.utility.Vector2iVector(lines)
        line_set_deformed.colors = o3d.utility.Vector3dVector([[0.0, 0.4, 1.0] for _ in range(len(lines))])
        
        # Pikice na vozliščih (rdeče)
        points_cloud = o3d.geometry.PointCloud()
        points_cloud.points = o3d.utility.Vector3dVector(pts_orig)
        points_cloud.colors = o3d.utility.Vector3dVector([[1.0, 0.2, 0.2] for _ in range(len(pts_orig))])
        
        # 3. VIZUALIZATOR
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Frame2D - Modal Analysis (Open3D Engine)", width=1200, height=800)
        
        vis.add_geometry(line_set_orig)
        vis.add_geometry(line_set_deformed)
        vis.add_geometry(points_cloud)
        
        # Nastavitve renderiranja
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])
        opt.line_width = 10.0
        opt.point_size = 12.0

        # 4. FUNKCIJA ZA POMIKE (2D -> 3D)
        def get_displacement(m_idx):
            mode = self.eig_vec[:, int(m_idx)]
            # V 2D imamo 3 prostostne stopnje na vozlišče (u, v, phi)
            # Vzamemo le u (0::3) in v (1::3), z-pomik je 0
            u_pomiki = mode[0::3]
            v_pomiki = mode[1::3]
            w_pomiki = np.zeros_like(u_pomiki)
            
            disp = np.column_stack((u_pomiki, v_pomiki, w_pomiki))
            
            model_size = np.max(np.ptp(pts_orig, axis=0)) if np.any(pts_orig) else 1.0
            max_val = np.max(np.linalg.norm(disp, axis=1))
            # Normalizacija in skaliranje
            if max_val > 1e-12:
                return disp * (model_size * 0.15 / max_val) * scale
            return disp

        # Stanje
        state = {
            't': 0.0,
            'animate': True,
            'current_mode': 0,
            'active_disp': get_displacement(0)
        }

        # 5. CALLBACKI ZA TIPKE
        def update_info():
            print(f"\rMode: {state['current_mode']} | Freq: {self.eig_freq[state['current_mode']]:.2f} Hz", end="")

        def next_mode(vis):
            state['current_mode'] = (state['current_mode'] + 1) % self.eig_vec.shape[1]
            state['active_disp'] = get_displacement(state['current_mode'])
            update_info()
            return False

        def prev_mode(vis):
            state['current_mode'] = (state['current_mode'] - 1) % self.eig_vec.shape[1]
            state['active_disp'] = get_displacement(state['current_mode'])
            update_info()
            return False

        def toggle_anim(vis):
            state['animate'] = not state['animate']
            return False

        # Registracija tipk (enak sistem kot v 3D različici)
        vis.register_key_callback(ord(' '), toggle_anim)
        vis.register_key_callback(262, next_mode) # Desno
        vis.register_key_callback(263, prev_mode) # Levo
        vis.register_key_callback(ord('D'), next_mode)
        vis.register_key_callback(ord('A'), prev_mode)

        print("\nKONTROLE: [Preslednica] Start/Stop | [A/D] ali [<- / ->] Menjava načina | [Q] Izhod\n")
        update_info()

        # 6. GLAVNA ZANKA
        try:
            while True:
                if state['animate']:
                    state['t'] += 0.15
                    deformed_pts = pts_orig + state['active_disp'] * np.sin(state['t'])
                    
                    # Update geometrije
                    line_set_deformed.points = o3d.utility.Vector3dVector(deformed_pts)
                    points_cloud.points = o3d.utility.Vector3dVector(deformed_pts)
                    
                    vis.update_geometry(line_set_deformed)
                    vis.update_geometry(points_cloud)
                
                if not vis.poll_events():
                    break
                vis.update_renderer()
                time.sleep(0.01)
        finally:
            vis.destroy_window()
            print("\nAnimacija končana.")
    
    def _update_solver(self):
        '''Method for updating matrices.'''

        M_glob = calculate_M_glob_frame(self.nodes, self.elements, self.A, self.rho)
        K_glob = calculate_K_glob_frame(self.nodes, self.elements, self.A, self.I, self.E)

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
        '''Open a UI to edit constraints of a Frame2D object. 

        If there are existing constraints you can remove them or add more within this function. \n
        Compatible with passing constraint matrix when creating Frame2D object. 
          

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
        p = pv.Plotter(notebook=False, title="Frame2D - Edit Constraints")
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

        p.add_text("[3] Fixed support", position=(0.45, self.y_pos), 
                    color='forestgreen', font_size=self.font_size, name="st_3", 
                    shadow=True, viewport=True)

        p.add_text("[4] Remove constraint", position=(0.72, self.y_pos), 
                    color='#333333', font_size=self.font_size, name="st_4", 
                    shadow=True, viewport=True)
        

        p.view_xy()
        p.camera.zoom(0.7)

        self.last_picked_idx = None
        self.current_angle = 0 
        n_dof = 3 * len(self.nodes)

        def draw_pinned_icon(node_idx):
            p.add_mesh(pv.Sphere(radius=self.r_size, center=pts[node_idx]), 
                       color="firebrick", name=f"fixed_{node_idx}")
        
        def draw_fixed_icon(node_idx):
            p.add_mesh(pv.Sphere(radius=self.r_size, center=pts[node_idx]), 
                       color="forestgreen", name=f"fixed_{node_idx}")

        def draw_angled_icon(node_idx, angle):
            p.add_mesh(pv.Sphere(radius=self.r_size, center=pts[node_idx]), 
                       color="royalblue", name=f"angled_{node_idx}")
            
            # smer puscice
            phi = np.radians(angle)
            direction = np.array([np.cos(phi), np.sin(phi), 0.0])
            
            # skaliranje puscic
            arr_len = self.r_size * 3 
            shaft_r = self.r_size * 0.4
            tip_r = self.r_size * 0.7
            
            # dvosmerna
            a1 = pv.Arrow(start=pts[node_idx], direction=direction, scale=arr_len, shaft_radius=shaft_r, tip_radius=tip_r)
            a2 = pv.Arrow(start=pts[node_idx], direction=-direction, scale=arr_len, shaft_radius=shaft_r, tip_radius=tip_r)
            combined_arrow = a1 + a2
            
            p.add_mesh(combined_arrow, color="royalblue", 
                       name=f"arrow_{node_idx}",
                       render_lines_as_tubes=True, lighting=False)

            offset_pos = pts[node_idx] + [axis_len*0.1, axis_len*0.1, 0]
            p.add_point_labels([offset_pos], [f"{int(angle)}°"], 
                               font_size=25, text_color="royalblue", 
                               name=f"angle_label_{node_idx}", 
                               shape=None, always_visible=True, shadow=True)

        if self.temp_rows:
            processed_nodes = set()
            for row in self.temp_rows:
                # kje v vrstici so vrednosti 
                active_dofs = np.where(np.abs(row) > 1e-6)[0]
                if len(active_dofs) == 0: continue
                
                node_idx = active_dofs[0] // 3
                if node_idx in processed_nodes: continue
                
                if np.abs(row[2*node_idx]) > 0.99 and np.abs(row[2*node_idx+1]) < 0.01:
                    draw_fixed_icon(node_idx)
                    processed_nodes.add(node_idx)
                else:
                    angle_rad = np.arctan2(row[2*node_idx+1], row[2*node_idx]) - np.pi/2
                    angle_deg = np.degrees(angle_rad) % 180
                    draw_angled_icon(node_idx, round(angle_deg))
                    processed_nodes.add(node_idx)


        def slider_callback(value):
            self.current_angle = int(round(value))
            update_angle_text()
        
        def update_angle_text():
            p.remove_actor("st_2")
            p.add_text(f"[2] Roller ({self.current_angle}°)", position=(0.25, self.y_pos), 
                       color='royalblue', font_size=self.font_size, name="st_2", 
                       shadow=True, viewport=True)


        def update_status_text():
            p.remove_actor("st_node")
            p.add_text(f"Node: {self.last_picked_idx}", position=(0.02, 0.08), 
                       color='black', font_size=self.font_size+10, name="st_node", 
                       shadow=True, viewport=True)
            

        def pick_callback(point_data, *args):
            idx = mesh.find_closest_point(point_data)
            self.last_picked_idx = idx
            p.add_mesh(pv.Sphere(radius=0.1, center=pts[idx]), color="yellow", opacity=0.4, name="selection_glow")
            p.add_mesh(pv.Sphere(radius=0.01, center=pts[idx]), color="black", name="selection_center")
            update_status_text()

        def set_pinned():
            if self.last_picked_idx is not None:
                remove_at_node()
                r1, r2 = np.zeros(n_dof), np.zeros(n_dof)
                r1[3*self.last_picked_idx], r2[3*self.last_picked_idx+1] = 1, 1
                self.temp_rows.extend([r1, r2])
                draw_pinned_icon(self.last_picked_idx)

        def set_angled():
            if self.last_picked_idx is not None:
                remove_at_node()
                phi = np.radians(self.current_angle)
                r = np.zeros(n_dof)
                r[3*self.last_picked_idx] = np.cos(phi + np.pi/2)
                r[3*self.last_picked_idx + 1] = np.sin(phi + np.pi/2)
                self.temp_rows.append(r)
                draw_angled_icon(self.last_picked_idx, self.current_angle)
        
        def set_fixed():
            if self.last_picked_idx is not None:
                remove_at_node()
                phi = np.radians(self.current_angle)
                r1, r2, r3 = np.zeros(n_dof), np.zeros(n_dof), np.zeros(n_dof)
                r1[3*self.last_picked_idx], r2[3*self.last_picked_idx+1], r3[3*self.last_picked_idx+2] = 1, 1, 1
                self.temp_rows.extend([r1, r2, r3])
                draw_fixed_icon(self.last_picked_idx)
        
        def remove_at_node():
            if self.last_picked_idx is not None:
                idx = self.last_picked_idx
                dofs = [3*idx, 3*idx + 1, 3*idx + 2]
                
                self.temp_rows = [r for r in self.temp_rows 
                                  if all(np.abs(r[d]) < 1e-6 for d in dofs)]
                # izbrisemo vse
                for prefix in ["pinned_", "fixed_", "angled_", "angle_label_", "arrow_"]:
                    p.remove_actor(f"{prefix}{idx}")
                

        p.add_slider_widget(callback=slider_callback, rng=[0, 180], value=0,
                            pointa=(0.6, 0.9), pointb=(0.9, 0.9), style='modern', color="black",
                            tube_width=0.003, slider_width=0.02, title="Set roller support angle:")

        p.add_text("Right click to select a node", position='upper_left', 
                   font_size=self.font_size, color='black', name="instruction_text")

        
        p.enable_point_picking(
            callback=pick_callback, 
            show_message=False,
            left_clicking=False
        )
        p.add_key_event('1', set_pinned)
        p.add_key_event('2', set_angled)
        p.add_key_event('3', set_fixed)
        p.add_key_event('4', remove_at_node)

        update_angle_text()
        p.show()

        
        if self.temp_rows:
            self.constraints = np.array(self.temp_rows)
            print("Solver has been updated.")
            self._update_solver()
        else:
            self.constraints = None


def calculate_K_glob_frame(nodes, elements, A, I, E):
    '''
    Compute the global stiffness matrix K_glob for 2D Frame elements.
    Each node has 3 DOF: [u, v, phi].
    '''
    n_nodes = len(nodes)
    n_ele = len(elements)
    ndim = 3  # u, v, phi
    n_total_dof = n_nodes * ndim

    K_glob = np.zeros((n_total_dof, n_total_dof))

    diffs = nodes[elements[:, 1]] - nodes[elements[:, 0]]
    Le = np.hypot(diffs[:, 0], diffs[:, 1])
    c = diffs[:, 0] / Le
    s = diffs[:, 1] / Le


    Ke_loc = np.zeros((n_ele, 6, 6))
    
    ka = E * A / Le
   
    b1 = 12 * E * I / Le**3
    b2 = 6 * E * I / Le**2
    b3 = 4 * E * I / Le
    b4 = 2 * E * I / Le

    Ke_loc[:, 0, 0] = ka;   Ke_loc[:, 0, 3] = -ka
    Ke_loc[:, 1, 1] = b1;   Ke_loc[:, 1, 2] = b2;  Ke_loc[:, 1, 4] = -b1; Ke_loc[:, 1, 5] = b2
    Ke_loc[:, 2, 1] = b2;   Ke_loc[:, 2, 2] = b3;  Ke_loc[:, 2, 4] = -b2; Ke_loc[:, 2, 5] = b4
    

    Ke_loc[:, 3, 0] = -ka;  Ke_loc[:, 3, 3] = ka
    Ke_loc[:, 4, 1] = -b1;  Ke_loc[:, 4, 2] = -b2; Ke_loc[:, 4, 4] = b1;  Ke_loc[:, 4, 5] = -b2
    Ke_loc[:, 5, 1] = b2;   Ke_loc[:, 5, 2] = b4;  Ke_loc[:, 5, 4] = -b2; Ke_loc[:, 5, 5] = b3


    T = np.zeros((n_ele, 6, 6))
    for i in range(n_ele):
        R = np.array([[c[i],  s[i], 0],
                      [-s[i], c[i], 0],
                      [ 0,     0,    1]])
        T[i, :3, :3] = R
        T[i, 3:, 3:] = R

    Ke_glob_all = T.swapaxes(1, 2) @ Ke_loc @ T

    idx = np.zeros((n_ele, 6), dtype=int)
    for i in range(2):
        node_indices = elements[:, i]
        idx[:, i*3]   = node_indices * 3     # u
        idx[:, i*3+1] = node_indices * 3 + 1 # v
        idx[:, i*3+2] = node_indices * 3 + 2 # phi

    rows = idx[:, :, np.newaxis]
    cols = idx[:, np.newaxis, :]

    np.add.at(K_glob, (rows, cols), Ke_glob_all)

    return K_glob

def calculate_M_glob_frame(nodes, elements, A, rho):
    '''
    Vektoriziran izračun globalne masne matrike brez for zank.
    '''
    ndim = 3
    n_ps = ndim * len(nodes)
    n_ele = len(elements)

    M_glob = np.zeros((n_ps, n_ps))

    diffs = nodes[elements[:, 1]] - nodes[elements[:, 0]]
    Le = np.hypot(diffs[:, 0], diffs[:, 1])
    c = diffs[:, 0] / Le
    s = diffs[:, 1] / Le

    # 1. Gradnja lokalne masne matrike (6x6)
    m_loc = np.zeros((n_ele, 6, 6))
    coeff = (rho * A * Le / 420)
    
    # Osni del
    m_loc[:, 0, 0] = 140; m_loc[:, 0, 3] = 70
    m_loc[:, 3, 0] = 70;  m_loc[:, 3, 3] = 140
    
    # Upogibni del
    m_loc[:, 1, 1] = 156;    m_loc[:, 1, 2] = 22*Le;   m_loc[:, 1, 4] = 54;     m_loc[:, 1, 5] = -13*Le
    m_loc[:, 2, 1] = 22*Le;  m_loc[:, 2, 2] = 4*Le**2;  m_loc[:, 2, 4] = 13*Le;  m_loc[:, 2, 5] = -3*Le**2
    m_loc[:, 4, 1] = 54;     m_loc[:, 4, 2] = 13*Le;   m_loc[:, 4, 4] = 156;    m_loc[:, 4, 5] = -22*Le
    m_loc[:, 5, 1] = -13*Le; m_loc[:, 5, 2] = -3*Le**2; m_loc[:, 5, 4] = -22*Le; m_loc[:, 5, 5] = 4*Le**2
    
    m_loc *= coeff[:, np.newaxis, np.newaxis]

    # 2. Vektorizirana gradnja transformacijske matrike T (6, 6)
    # Namesto zanke uporabimo napredno indeksiranje
    T = np.zeros((n_ele, 6, 6))
    
    # Vozlišče 1 rotacijski blok
    T[:, 0, 0] = c;  T[:, 0, 1] = s
    T[:, 1, 0] = -s; T[:, 1, 1] = c
    T[:, 2, 2] = 1.0
    
    # Vozlišče 2 rotacijski blok
    T[:, 3, 3] = c;  T[:, 3, 4] = s
    T[:, 4, 3] = -s; T[:, 4, 4] = c
    T[:, 5, 5] = 1.0

    # 3. Transformacija vseh elementov hkrati: M_glob_ele = T.T @ M_loc @ T
    # swapaxes(1, 2) opravi transponiranje 6x6 matrik za vse elemente hkrati
    M_vsi = T.swapaxes(1, 2) @ m_loc @ T

    # 4. Sestavljanje v globalno matriko
    indeksi = (elements.flatten()[:, np.newaxis] * 3 + [0, 1, 2]).flatten().reshape(-1, 6)
    rows = indeksi[:, :, np.newaxis]
    cols = indeksi[:, np.newaxis, :]

    np.add.at(M_glob, (rows, cols), M_vsi)

    return M_glob