import numpy as np
import scipy as sp
from typing import Optional, Union
import pyvista as pv
import open3d as o3d
import time

class Truss3D():
    '''A universal class for 3D truss analysis using FEM with Truss elements.
   
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
        nodes : np.ndarray, shape (n_nodes, 3)
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
        if nodes.shape[1] != 3:
            raise TypeError("List of nodes must be a 3D array with shape (N, 3)")

        self.nodes = nodes
        self.elements = elements
        self.A = A
        self.E = E
        self.rho = rho
        self.type = "truss"

        self.n_dof = 3 * len(self.nodes)
        self.constraints = constraints if constraints is not None else np.empty((0, self.n_dof))
        self._update_solver()

    def _update_solver(self):
        '''Method for updating matrices.'''
        M_glob = calculate_M_glob_truss_3d(self.nodes, self.elements, self.A, self.rho)
        K_glob = calculate_K_glob_truss_3d(self.nodes, self.elements, self.A, self.E)
        self.M_glob = M_glob
        self.K_glob = K_glob

        if self.constraints.size > 0:
            L = sp.linalg.null_space(self.constraints)
            M_c = L.T @ M_glob @ L
            K_c = L.T @ K_glob @ L
            
            val, vec_c = sp.linalg.eigh(K_c, M_c)
            self.eig_vec = L @ vec_c
            self.eig_val = np.abs(val)
        else:
            self.eig_val, self.eig_vec = sp.linalg.eigh(K_glob, M_glob)
            self.eig_val = np.abs(self.eig_val)

        self.eig_freq = (np.sqrt(self.eig_val) / (2 * np.pi)).round(3)
        print("Solver has been updated.")

    def animate_mode_shapes(self, scale=1.0):
        pts_orig = self.nodes.copy()
        model_size = np.max(np.ptp(pts_orig, axis=0))
        axis_len = model_size * 0.2
        
        lines = [[e[0], e[1]] for e in self.elements]
        
        line_set_orig = o3d.geometry.LineSet()
        line_set_orig.points = o3d.utility.Vector3dVector(pts_orig)
        line_set_orig.lines = o3d.utility.Vector2iVector(lines)
        line_set_orig.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7] for _ in range(len(lines))])
        
        line_set_deformed = o3d.geometry.LineSet()
        line_set_deformed.points = o3d.utility.Vector3dVector(pts_orig)
        line_set_deformed.lines = o3d.utility.Vector2iVector(lines)
        line_set_deformed.colors = o3d.utility.Vector3dVector([[0.0, 0.4, 1.0] for _ in range(len(lines))])
        
        points_cloud = o3d.geometry.PointCloud()
        points_cloud.points = o3d.utility.Vector3dVector(pts_orig)
        points_cloud.colors = o3d.utility.Vector3dVector([[1.0, 0.2, 0.2] for _ in range(len(pts_orig))])
        
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Truss3D - Modal Analysis", width=1200, height=800)
        
        vis.add_geometry(line_set_orig)
        vis.add_geometry(line_set_deformed)
        vis.add_geometry(points_cloud)

        
        
        def create_axis_arrow(direction, color):
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=axis_len*0.02, cone_radius=axis_len*0.04,
                cylinder_height=axis_len*0.8, cone_height=axis_len*0.2
            )
            arrow.paint_uniform_color(color)
            if direction == 'x':
                arrow.rotate(arrow.get_rotation_matrix_from_axis_angle([0, np.pi/2, 0]), center=(0, 0, 0))
            elif direction == 'y':
                arrow.rotate(arrow.get_rotation_matrix_from_axis_angle([-np.pi/2, 0, 0]), center=(0, 0, 0))
            return arrow

        vis.add_geometry(create_axis_arrow('x', [1.0, 0.0, 0.0]))
        vis.add_geometry(create_axis_arrow('y', [0.0, 1.0, 0.0]))
        vis.add_geometry(create_axis_arrow('z', [0.0, 0.0, 1.0]))
        
        opt = vis.get_render_option()
        opt.line_width = 10.0; opt.point_size = 8.0; opt.background_color = np.array([1, 1, 1])

        def get_displacement(m_idx):
            mode = self.eig_vec[:, int(m_idx)]
            disp = np.column_stack((mode[0::3], mode[1::3], mode[2::3]))
            max_val = np.max(np.linalg.norm(disp, axis=1))
            return disp * (model_size * 0.15 / max_val) * scale if max_val > 1e-12 else disp
    
        state = {'t': 0.0, 'animate': True, 'current_mode': 0, 'active_disp': get_displacement(0)}
        
        def update_title():
            print(f"\rMode: {state['current_mode']} | Freq: {self.eig_freq[state['current_mode']]:.2f} Hz", end="", flush=True)

        vis.register_key_callback(ord(' '), lambda v: state.update({'animate': not state['animate']}) or False)
        vis.register_key_callback(262, lambda v: state.update({'current_mode': (state['current_mode']+1)%len(self.eig_freq), 'active_disp': get_displacement((state['current_mode']+1)%len(self.eig_freq))}) or update_title() or False)
        vis.register_key_callback(263, lambda v: state.update({'current_mode': (state['current_mode']-1)%len(self.eig_freq), 'active_disp': get_displacement((state['current_mode']-1)%len(self.eig_freq))}) or update_title() or False)

        update_title()
        try:
            while vis.poll_events():
                if state['animate']:
                    state['t'] += 0.15
                    deformed_pts = pts_orig + state['active_disp'] * np.sin(state['t'])
                    line_set_deformed.points = o3d.utility.Vector3dVector(deformed_pts)
                    points_cloud.points = o3d.utility.Vector3dVector(deformed_pts)
                    vis.update_geometry(line_set_deformed); vis.update_geometry(points_cloud)
                vis.update_renderer(); time.sleep(0.01)
        finally:
            vis.destroy_window()

    def display(self):
        pts = self.nodes.copy()
        model_size = np.max(np.ptp(pts, axis=0))
        axis_len = model_size * 0.2
        
        lines = [[e[0], e[1]] for e in self.elements]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts); line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0.1, 0.1, 0.1] for _ in range(len(lines))])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts); pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(pts))])
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Truss3D Display", width=1200, height=800)
        vis.add_geometry(line_set); vis.add_geometry(pcd)

        def create_axis_arrow(direction, color):
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=axis_len*0.02,
                cone_radius=axis_len*0.04,
                cylinder_height=axis_len*0.8,
                cone_height=axis_len*0.2
            )
            arrow.paint_uniform_color(color)
            
            if direction == 'x':
                R = arrow.get_rotation_matrix_from_axis_angle([0, np.pi/2, 0])
                arrow.rotate(R, center=(0, 0, 0))
            elif direction == 'y':
                R = arrow.get_rotation_matrix_from_axis_angle([-np.pi/2, 0, 0])
                arrow.rotate(R, center=(0, 0, 0))
            return arrow

        arrow_x = create_axis_arrow('x', [1.0, 0.0, 0.0])
        arrow_y = create_axis_arrow('y', [0.0, 1.0, 0.0]) 
        arrow_z = create_axis_arrow('z', [0.0, 0.0, 1.0])

        vis.add_geometry(arrow_x)
        vis.add_geometry(arrow_y)
        vis.add_geometry(arrow_z)
        
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1]); opt.point_size = 12.0; opt.line_width = 5.0
        vis.run(); vis.destroy_window()

        

    def edit_constraints_3d(self):
        '''Open a UI to edit constraints of a Truss3D object in 3D.
        
        Supports:
        - [1] Pinned support (ux=uy=uz=0)
        - [2] Roller support (nastavljivi koti s sliderji)
        - [3] Remove constraints at node
        '''
        import pyvista as pv
        import numpy as np

        self.font_size = 15
        self.y_pos = 0.03
        
        element_lengths = []
        for e in self.elements:
            p1 = self.nodes[e[0]]
            p2 = self.nodes[e[1]]
            dist = np.linalg.norm(p2 - p1)
            if dist > 1e-6:
                element_lengths.append(dist)
        
        if element_lengths:
            min_len = np.min(element_lengths)
            axis_len = min_len / 3.0
        else:
            axis_len = 1.0

        self.r_size = np.clip(axis_len, 0, 0.1)

        if hasattr(self, 'constraints') and self.constraints is not None:
            self.temp_rows = self.constraints.tolist()
        else:
            self.temp_rows = []

        pv.set_jupyter_backend(None)
        p = pv.Plotter(notebook=False, title="Truss3D - Edit Constraints", 
                        window_size=[1400, 900])
        p.set_background("white")
        p.enable_parallel_projection()

        pts = self.nodes.copy()
        cells = np.hstack([[2, e[0], e[1]] for e in self.elements])
        mesh = pv.UnstructuredGrid(cells, [pv.CellType.LINE]*len(self.elements), pts)

        p.add_mesh(mesh, color="#555555", line_width=2, render_lines_as_tubes=True)
        p.add_point_labels(pts, [f"{i}" for i in range(len(self.nodes))], 
                        point_size=10, font_size=18, text_color="black", 
                        always_visible=True, name="node_labels", shadow=True)

        # koordinatne osi
        origin = [0, 0, 0]
        p.add_mesh(pv.Line(origin, [axis_len, 0, 0]), color='red', line_width=5, name="axis_x")
        p.add_mesh(pv.Line(origin, [0, axis_len, 0]), color='green', line_width=5, name="axis_y")
        p.add_mesh(pv.Line(origin, [0, 0, axis_len]), color='blue', line_width=5, name="axis_z")
        p.add_point_labels([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]], 
                          ["X", "Y", "Z"], font_size=15, text_color='black', always_visible=True)

        p.add_text("[1] Pinned", position=(0.02, self.y_pos), 
                color='firebrick', font_size=self.font_size, name="st_1", 
                shadow=True, viewport=True)

        p.add_text("[2] Roller", position=(0.15, self.y_pos), 
                color='royalblue', font_size=self.font_size, name="st_2", 
                shadow=True, viewport=True)

        p.add_text("[3] Remove", position=(0.28, self.y_pos), 
                color='#333333', font_size=self.font_size, name="st_3", 
                shadow=True, viewport=True)

        p.add_text("Right click to select a node", position='upper_left', 
                font_size=self.font_size, color='black', name="instruction_text")

        self.roller_angles = {'x': -1, 'y': -1, 'z': -1}
        
        def update_angle_texts():
            p.remove_actor("angle_info")
            info = []
            for ax in ['x', 'y', 'z']:
                val = self.roller_angles[ax]
                status = "FIXED" if val == -1 else f"{val}°"
                info.append(f"{ax.upper()}: {status}")
            
            angle_text = "Roller status:  " + "  ".join(info)
            p.add_text(angle_text, position=(0.45, self.y_pos), 
                    color='royalblue', font_size=self.font_size, 
                    name="angle_info", viewport=True, shadow=True)

        def slider_x_callback(value):
            self.roller_angles['x'] = int(round(value))
            update_angle_texts()
        def slider_y_callback(value):
            self.roller_angles['y'] = int(round(value))
            update_angle_texts()
        def slider_z_callback(value):
            self.roller_angles['z'] = int(round(value))
            update_angle_texts()

        for i, (label, cb) in enumerate([("X angle:", slider_x_callback), 
                                        ("Y angle:", slider_y_callback), 
                                        ("Z angle:", slider_z_callback)]):
            p.add_slider_widget(callback=cb, rng=[-1, 180], value=-1,
                                pointa=(0.75, 0.35 - i*0.115), pointb=(0.95, 0.35 - i*0.115), 
                                style='modern', color="royalblue", title=label, 
                                slider_width=0.01, tube_width=0.002, fmt="{:.0f}")

        p.camera_position = 'iso'
        p.camera.zoom(0.8)

        self.last_picked_idx = None
        n_dof_total = 3 * len(self.nodes)

        def draw_pinned_icon(node_idx):
            p.add_mesh(pv.Sphere(radius=self.r_size, center=pts[node_idx]), 
                    color="firebrick", name=f"pinned_{node_idx}")
        
        def draw_roller_icon(node_idx):
            p.add_mesh(pv.Sphere(radius=self.r_size, center=pts[node_idx]), 
                    color="royalblue", name=f"roller_{node_idx}")
            

            ax = np.radians(max(0, self.roller_angles['x']))
            ay = np.radians(max(0, self.roller_angles['y']))
            az = np.radians(max(0, self.roller_angles['z']))

            # x slider vrti v ravnini yz (začnemo na y, vrtimo proti z)
            vec_x = np.array([0.0, np.cos(ax), np.sin(ax)])
            
            # y slider vrti v ravnini xz (začnemo na z, vrtimo proti x)
            vec_y = np.array([np.sin(ay), 0.0, np.cos(ay)])
            
            # z slider vrti v ravnini xy (začnemo na x, vrtimo proti y)
            vec_z = np.array([np.cos(az), np.sin(az), 0.0])

            # ce je slider na -1 je to fixed
            direction = np.array([0.0, 0.0, 0.0])
            if self.roller_angles['x'] != -1: direction += vec_x
            if self.roller_angles['y'] != -1: direction += vec_y
            if self.roller_angles['z'] != -1: direction += vec_z

            # ce so vsi -1 je to x smer
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction /= norm

            # puscica za prikaz smeri 
            arr_len = axis_len * 0.8
            a1 = pv.Arrow(start=pts[node_idx], direction=direction, scale=arr_len, 
                        shaft_radius=0.015, tip_radius=0.04)
            a2 = pv.Arrow(start=pts[node_idx], direction=-direction, scale=arr_len, 
                        shaft_radius=0.015, tip_radius=0.04)
            
            p.add_mesh(a1 + a2, color="royalblue", name=f"arrow_{node_idx}", lighting=True)
            
            offset = pts[node_idx] + [axis_len*0.2]*3
            cfg = "/".join(["F" if self.roller_angles[a]==-1 else f"{self.roller_angles[a]}°" for a in ['x','y','z']])
            p.add_point_labels([offset], [cfg], font_size=12, text_color="royalblue", 
                            name=f"angle_label_{node_idx}", always_visible=True, shadow=True)
        
        def update_status_text():
            p.remove_actor("st_node")
            if self.last_picked_idx is not None:
                p.add_text(f"Node: {self.last_picked_idx}", position=(0.02, 0.92), 
                        color='black', font_size=self.font_size+10, name="st_node", 
                        shadow=True, viewport=True)

        def pick_callback(point_data, *args):
            idx = mesh.find_closest_point(point_data)
            self.last_picked_idx = idx
            p.add_mesh(pv.Sphere(radius=self.r_size*1.5, center=pts[idx]), 
                    color="yellow", opacity=0.3, name="selection_glow")
            p.add_mesh(pv.Sphere(radius=self.r_size*0.3, center=pts[idx]), 
                    color="black", name="selection_center")
            update_status_text()

        def remove_at_node():
            if self.last_picked_idx is not None:
                idx = self.last_picked_idx
                # 3 ps pobrisemo
                dofs = [3*idx + i for i in range(3)]
                self.temp_rows = [r for r in self.temp_rows if all(np.abs(r[d]) < 1e-6 for d in dofs)]
                
                for prefix in ["pinned_", "roller_"]:
                    p.remove_actor(f"{prefix}{idx}")
                p.remove_actor(f"angle_label_{idx}")

        def set_pinned():
            if self.last_picked_idx is not None:
                remove_at_node()
                idx = self.last_picked_idx
                for i in range(3):
                    r = np.zeros(n_dof_total)
                    r[3*idx + i] = 1.0
                    self.temp_rows.append(r)
                draw_pinned_icon(idx)

        def set_roller():
            if self.last_picked_idx is not None:
                remove_at_node()
                idx = self.last_picked_idx
                for i, axis in enumerate(['x', 'y', 'z']):
                    val = self.roller_angles[axis]
                    if val == -1:
                        r = np.zeros(n_dof_total)
                        r[3*idx + i] = 1.0
                        self.temp_rows.append(r)
                    elif val > 0: 
                        phi = np.radians(val)
                        r = np.zeros(n_dof_total)
                        r[3*idx + i] = np.cos(phi)
                        if i < 2: r[3*idx + i + 1] = np.sin(phi)
                        self.temp_rows.append(r)
                draw_roller_icon(idx)

        p.enable_point_picking(callback=pick_callback, left_clicking=False, show_message=False, pickable_window=True)
        
        p.add_key_event('1', set_pinned)
        p.add_key_event('2', set_roller)
        p.add_key_event('3', remove_at_node)

        update_angle_texts()
        p.show()

        if self.temp_rows:
            self.constraints = np.array(self.temp_rows)
            self._update_solver()
        else:
            self.constraints = None

def calculate_K_glob_truss_3d(nodes, elements, A, E):
    n_ps = 3 * len(nodes)
    K_glob = np.zeros((n_ps, n_ps))
    for n1, n2 in elements:
        diff = nodes[n2] - nodes[n1]; L = np.linalg.norm(diff)
        l, m, n = diff / L
        ke_val = E * A / L
        T = np.array([l, m, n, -l, -m, -n])
        ke_g = np.outer(T, T) * ke_val
        dofs = np.concatenate([np.arange(n1*3, n1*3+3), np.arange(n2*3, n2*3+3)])
        K_glob[np.ix_(dofs, dofs)] += ke_g
    return K_glob

def calculate_M_glob_truss_3d(nodes, elements, A, rho):
    n_ps = 3 * len(nodes)
    M_glob = np.zeros((n_ps, n_ps))
    for n1, n2 in elements:
        L = np.linalg.norm(nodes[n2] - nodes[n1])
        m_konst = (rho * A * L) / 6
        me = np.array([[2,0,0,1,0,0],[0,2,0,0,1,0],[0,0,2,0,0,1],[1,0,0,2,0,0],[0,1,0,0,2,0],[0,0,1,0,0,2]]) * m_konst
        dofs = np.concatenate([np.arange(n1*3, n1*3+3), np.arange(n2*3, n2*3+3)])
        M_glob[np.ix_(dofs, dofs)] += me
    return M_glob