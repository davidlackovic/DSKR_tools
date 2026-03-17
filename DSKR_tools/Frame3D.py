import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from typing import Optional, Union
import pyvista as pv
import open3d as o3d
import time



class Frame3D():
    def __init__(
        self, 
        nodes: np.ndarray, 
        elements: np.ndarray, 
        A: float | np.ndarray, 
        E: float | np.ndarray, 
        G: float | np.ndarray, 
        Iy: float | np.ndarray, 
        Iz: float | np.ndarray, 
        J: float | np.ndarray, 
        rho: float | np.ndarray, 
        constraints: np.ndarray | None = None, 
        n_mesh: int | None = None
    ):
        '''
        Initialization of Frame3D object.

        Parameters
        ----------
        nodes : np.ndarray, shape (n_nodes, 3)
            Array of nodal coordinates [x, y, z]. 

        elements : np.ndarray, shape (n_ele, 2)
            Array of node index pairs [node_start, node_end].

        A, E, G, Iy, Iz, J, rho : float or np.ndarray
            Cross-sectional and material properties. Can be scalar or vectors.
            - G: Shear modulus
            - Iy, Iz: Second moments of area (bending)
            - J: Torsional constant
            - rho: Density

        constraints : np.ndarray, shape (n_constraints, n_dof), optional
            Constraint matrix. n_dof = nodes * 6.
        
        n_mesh : int, optional
            Subdivision factor for elements (linear interpolation in 3D).
        '''

        self.nodes = nodes
        self.elements = elements
        self.A = A
        self.E = E
        self.G = G
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.rho = rho

        if n_mesh is not None and n_mesh > 1:
            all_nodes = list(self.nodes)
            all_elements = []

            def get_node_idx(point):
                for i, n in enumerate(all_nodes):
                    if np.allclose(n, point, atol=1e-6): return i
                all_nodes.append(point)
                return len(all_nodes) - 1

            for start_idx, end_idx in elements:
                p1, p2 = nodes[start_idx], nodes[end_idx]
                curr_start = start_idx
                for i in range(1, n_mesh + 1):
                    new_point = p1 + (i / n_mesh) * (p2 - p1)
                    curr_end = get_node_idx(new_point)
                    all_elements.append([curr_start, curr_end])
                    curr_start = curr_end

            self.nodes = np.array(all_nodes)
            self.elements = np.array(all_elements)
            print(f"3D Mesh: {len(self.nodes)} nodes, {len(self.elements)} elements.")

        # 6 prostostnih stopenj na vozlisce
        self.n_dof = 6 * len(self.nodes)
        self.constraints = constraints if constraints is not None else np.empty((0, self.n_dof))
        self._update_solver()


    def _update_solver(self):
        # s skalarnimi parametri
        # TODO: dodaj vektorske parametre
        M_glob = calculate_M_glob_frame3d(self.nodes, self.elements, self.A, self.rho, self.J)
        K_glob = calculate_K_glob_frame3d(self.nodes, self.elements, self.A, self.E, self.G, self.Iy, self.Iz, self.J)

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
        ''' Animacija lastnih oblik z Open3D - s pikicami na vozliščih in koordinatnimi osmi '''
        pts_orig = self.nodes.copy()

        ranges = np.ptp(pts_orig, axis=0)
        model_size = np.max(ranges)   

        axis_len = model_size * 0.2
        
        lines = []
        for e in self.elements:
            lines.append([e[0], e[1]])
        
        # nedeformirana mreza
        line_set_orig = o3d.geometry.LineSet()
        line_set_orig.points = o3d.utility.Vector3dVector(pts_orig)
        line_set_orig.lines = o3d.utility.Vector2iVector(lines)
        
        # deformirana mreža
        line_set_deformed = o3d.geometry.LineSet()
        line_set_deformed.points = o3d.utility.Vector3dVector(pts_orig)
        line_set_deformed.lines = o3d.utility.Vector2iVector(lines)
        
        # barve
        colors_orig = [[0.7, 0.7, 0.7] for _ in range(len(lines))]  # siva
        colors_deformed = [[0.0, 0.4, 1.0] for _ in range(len(lines))]  # modra
        
        line_set_orig.colors = o3d.utility.Vector3dVector(colors_orig)
        line_set_deformed.colors = o3d.utility.Vector3dVector(colors_deformed)
        
        # vozlisca
        points_cloud = o3d.geometry.PointCloud()
        points_cloud.points = o3d.utility.Vector3dVector(pts_orig)
        colors_points = [[1.0, 0.2, 0.2] for _ in range(len(pts_orig))]  # rdeče
        points_cloud.colors = o3d.utility.Vector3dVector(colors_points)
        
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Frame3D - Modal Analysis with Nodes and Axes", width=1200, height=800)
        
        vis.add_geometry(line_set_orig)
        vis.add_geometry(line_set_deformed)
        vis.add_geometry(points_cloud)
        
        # ustvarjanje puscic
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
        
        render_option = vis.get_render_option()
        render_option.line_width = 16.0 
        render_option.point_size = 10.0 
        render_option.background_color = np.array([1.0, 1.0, 1.0])

        # izracun deformacije
        def get_displacement(m_idx):
            m_idx = int(round(m_idx))
            mode = self.eig_vec[:, m_idx]
            disp = np.column_stack((mode[0::6], mode[1::6], mode[2::6]))
            model_size = np.max(np.ptp(pts_orig, axis=0)) if np.any(pts_orig) else 1.0
            max_val = np.max(np.linalg.norm(disp, axis=1))
            return disp * (model_size * 0.15 / max_val) * scale if max_val > 1e-12 else disp
    
        state = {
            't': 0.0,
            'animate': True,
            'current_mode': 0,
            'active_disp': get_displacement(0)
        }
        
        def update_window_title():
            anim_status = "ON" if state['animate'] else "OFF"
            print(f"\rMode: {state['current_mode']} | Freq: {self.eig_freq[state['current_mode']]:.2f} Hz | Animacija: {anim_status}", end="", flush=True)
        
        update_window_title()
        
        # tipke
        def toggle_animation(vis):
            state['animate'] = not state['animate']
            if state['animate']:
                state['t'] = 0.0
            update_window_title()
            return False
        
        def next_mode(vis):
            state['current_mode'] = (state['current_mode'] + 1) % len(self.eig_freq)
            state['active_disp'] = get_displacement(state['current_mode'])
            update_window_title()
            return False
        
        def prev_mode(vis):
            state['current_mode'] = (state['current_mode'] - 1) % len(self.eig_freq)
            state['active_disp'] = get_displacement(state['current_mode'])
            update_window_title()
            return False
        
        def exit_visualizer(vis):
            vis.close()
            return True
        
        vis.register_key_callback(ord(' '), toggle_animation)
        vis.register_key_callback(ord('D'), next_mode)
        vis.register_key_callback(ord('d'), next_mode)
        vis.register_key_callback(ord('A'), prev_mode)
        vis.register_key_callback(ord('a'), prev_mode)
        vis.register_key_callback(262, next_mode)  # desno puscica
        vis.register_key_callback(263, prev_mode)  # levo puscica
        vis.register_key_callback(ord('Q'), exit_visualizer)
        vis.register_key_callback(ord('q'), exit_visualizer)
        vis.register_key_callback(256, exit_visualizer)  # ESC
        
        print("\n" + "="*60)
        print("Frame3D - Modal shapes animation")
        print("="*60)
        print("NAVODILA:")
        print("  SPACE: vklop/izklop animacije")
        print("  A/←: prejšnji mode")
        print("  D/→: naslednji mode")
        print("  Q/ESC: izhod")
        print("\nKOORDINATNE OSI:")
        print("  Rdeča: X os")
        print("  Zelena: Y os")
        print("  Modra: Z os")
        print("="*60 + "\n")
        
        # zanka
        try:
            while True:
                if state['animate']:
                    state['t'] += 0.15
                    
                    factor = np.sin(state['t'])
                    deformed_pts = pts_orig + state['active_disp'] * factor
                    
                    line_set_deformed.points = o3d.utility.Vector3dVector(deformed_pts)
                    vis.update_geometry(line_set_deformed)
                    
                    points_cloud.points = o3d.utility.Vector3dVector(deformed_pts)
                    vis.update_geometry(points_cloud)
                
                if not vis.poll_events():
                    break
                vis.update_renderer()
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            pass
        finally:
            vis.destroy_window()
            print("\n" + "="*60)
            print("Animacija končana")
            print("="*60)


    def display(self, title="Frame3D - Paličje"):
        ''' Display undeformed nodes and elements with visible node points. '''
        import open3d as o3d
        import numpy as np

        pts = self.nodes.copy()

        # razpon modela po vseh treh oseh
        ranges = np.ptp(pts, axis=0)
        model_size = np.max(ranges)   

        axis_len = model_size * 0.2
        
        lines = [[e[0], e[1]] for e in self.elements]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[0.1, 0.1, 0.1] for _ in range(len(lines))])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0] for _ in range(len(pts))]) # Rdeča
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1200, height=800)
        
        vis.add_geometry(line_set)
        vis.add_geometry(pcd)

         # ustvarjanje puscic
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
        opt.background_color = np.array([1, 1, 1]) 
        opt.point_size = 12.0                      
        opt.line_width = 5.0                  
        
        vis.run()
        vis.destroy_window()
    

    def edit_constraints_3d(self):
        '''Open a UI to edit constraints of a Frame3D object in 3D.
        
        Supports:
        - [1] Pinned support (ux=uy=uz=0)
        - [2] Roller support (nastavljivi koti s sliderji)
        - [3] Fixed support (all 6 DOFs=0)
        - [4] Remove constraints at node
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


        self.r_size = axis_len  # radij sfer
        self.r_size = np.clip(self.r_size, 0, 0.1)

        if hasattr(self, 'constraints') and self.constraints is not None:
            self.temp_rows = self.constraints.tolist()
        else:
            self.temp_rows = []

        pv.set_jupyter_backend(None)
        p = pv.Plotter(notebook=False, title="Frame3D - Edit Constraints", 
                    window_size=[1400, 900])
        p.set_background("white")
        p.enable_parallel_projection()

        pts = self.nodes.copy()
        cells = np.hstack([[2, e[0], e[1]] for e in self.elements])
        mesh = pv.UnstructuredGrid(cells, [pv.CellType.LINE]*len(self.elements), pts)

        # palicje
        p.add_mesh(mesh, color="#555555", line_width=2, render_lines_as_tubes=True)
        p.add_point_labels(pts, [f"{i}" for i in range(len(self.nodes))], 
                        point_size=10, font_size=18, text_color="black", 
                        always_visible=True, name="node_labels", shadow=True)


        origin = [0, 0, 0]
        
        line_x = pv.Line(origin, [axis_len, 0, 0])
        line_y = pv.Line(origin, [0, axis_len, 0])
        line_z = pv.Line(origin, [0, 0, axis_len])


        p.add_mesh(line_x, color='red', line_width=5, name="axis_x")
        p.add_mesh(line_y, color='green', line_width=5, name="axis_y")
        p.add_mesh(line_z, color='blue', line_width=5, name="axis_z")

        # oznake na konce linij
        p.add_point_labels(
            [[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]], 
            ["X", "Y", "Z"], 
            font_size=15, 
            text_color='black', 
            always_visible=True,
            shadow=False
        )
        

        # UI
        p.add_text("[1] Pinned", position=(0.02, self.y_pos), 
                color='firebrick', font_size=self.font_size, name="st_1", 
                shadow=True, viewport=True)

        p.add_text("[2] Roller", position=(0.15, self.y_pos), 
                color='royalblue', font_size=self.font_size, name="st_2", 
                shadow=True, viewport=True)

        p.add_text("[3] Fixed", position=(0.28, self.y_pos), 
                color='forestgreen', font_size=self.font_size, name="st_3", 
                shadow=True, viewport=True)

        p.add_text("[4] Remove", position=(0.38, self.y_pos), 
                color='#333333', font_size=self.font_size, name="st_4", 
                shadow=True, viewport=True)

        p.add_text("Right click to select a node", position='upper_left', 
                font_size=self.font_size, color='black', name="instruction_text")

        # sliderji za kote roller
        self.roller_angles = {'x': -1, 'y': -1, 'z': -1}
        
        def update_angle_texts():
            p.remove_actor("angle_info")
            info = []
            for ax in ['x', 'y', 'z']:
                val = self.roller_angles[ax]
                status = "FIXED" if val == -1 else f"{val}°"
                info.append(f"{ax.upper()}: {status}")
            
            angle_text = "Roller status:  " + "  ".join(info)
            p.add_text(angle_text, position=(0.50, self.y_pos), 
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

        # -1 pomeni fiksirano, 0+ pomeni kot drsenja
        for i, (label, cb) in enumerate([("X angle:", slider_x_callback), 
                                        ("Y angle:", slider_y_callback), 
                                        ("Z angle:", slider_z_callback)]):
            p.add_slider_widget(callback=cb, rng=[-1, 180], value=-1,
                                pointa=(0.75, 0.35 - i*0.115), pointb=(0.95, 0.35 - i*0.115), 
                                style='modern', color="royalblue", title=label, slider_width=0.01, tube_width=0.002, fmt="{:.0f}")

        p.camera_position = 'iso'
        p.camera.zoom(0.8)

        self.last_picked_idx = None
        n_dof = 6 * len(self.nodes)

        def draw_pinned_icon(node_idx):
            p.add_mesh(pv.Sphere(radius=self.r_size, center=pts[node_idx]), 
                    color="firebrick", name=f"pinned_{node_idx}")
        
        def draw_roller_icon(node_idx):
            p.add_mesh(pv.Sphere(radius=self.r_size, center=pts[node_idx]), 
                    color="royalblue", name=f"roller_{node_idx}")
            
            offset = pts[node_idx] + [axis_len*0.15, axis_len*0.15, axis_len*0.15]
            angle_text = f"{self.roller_angles['x']}°/{self.roller_angles['y']}°/{self.roller_angles['z']}°"
            p.add_point_labels([offset], [angle_text], 
                            font_size=14, text_color="royalblue", 
                            name=f"angle_label_{node_idx}", 
                            shape=None, always_visible=True, shadow=True)
        
        def draw_fixed_icon(node_idx):
            p.add_mesh(pv.Sphere(radius=self.r_size*1.2, center=pts[node_idx]), 
                    color="forestgreen", name=f"fixed_{node_idx}")

        # naris obstojece podpore
        if self.temp_rows:
            node_to_dofs = {}
            for row in self.temp_rows:
                active_dofs = np.where(np.abs(row) > 1e-6)[0]
                for dof in active_dofs:
                    n_idx = dof // 6
                    d_idx = dof % 6
                    if n_idx not in node_to_dofs:
                        node_to_dofs[n_idx] = set()
                    node_to_dofs[n_idx].add(d_idx)

            for node_idx, dofs in node_to_dofs.items():
                trans_dofs = {d for d in dofs if d < 3}
                rot_dofs = {d for d in dofs if d >= 3}

                if dofs == {0, 1, 2, 3, 4, 5}:
                    # FIXED
                    draw_fixed_icon(node_idx)
                elif trans_dofs == {0, 1, 2} and not rot_dofs:
                    # PINNED
                    draw_pinned_icon(node_idx)
                else:
                    # ROLLER
                    draw_roller_icon(node_idx)

        def update_status_text():
            p.remove_actor("st_node")
            if self.last_picked_idx is not None:
                p.add_text(f"Node: {self.last_picked_idx}", position=(0.02, 0.92), 
                        color='black', font_size=self.font_size+10, name="st_node", 
                        shadow=True, viewport=True)

        def pick_callback(point_data, *args):
            idx = mesh.find_closest_point(point_data)
            self.last_picked_idx = idx
            # izbere voz
            p.add_mesh(pv.Sphere(radius=self.r_size*1.5, center=pts[idx]), 
                    color="yellow", opacity=0.3, name="selection_glow")
            p.add_mesh(pv.Sphere(radius=self.r_size*0.3, center=pts[idx]), 
                    color="black", name="selection_center")
            update_status_text()

        def remove_at_node():
            if self.last_picked_idx is not None:
                idx = self.last_picked_idx
                dofs = [6*idx + i for i in range(6)]

                self.temp_rows = [r for r in self.temp_rows 
                                if all(np.abs(r[d]) < 1e-6 for d in dofs)]
                
                prefixes = ["pinned_", "roller_", "fixed_"]
                for prefix in prefixes:
                    p.remove_actor(f"{prefix}{idx}")
                p.remove_actor(f"angle_label_{idx}")

        def set_pinned():
            if self.last_picked_idx is not None:
                remove_at_node()
                r1, r2, r3 = np.zeros(n_dof), np.zeros(n_dof), np.zeros(n_dof)
                r1[6*self.last_picked_idx] = 1      # ux
                r2[6*self.last_picked_idx + 1] = 1  # uy
                r3[6*self.last_picked_idx + 2] = 1  # uz
                self.temp_rows.extend([r1, r2, r3])
                draw_pinned_icon(self.last_picked_idx)

        def set_roller():
            if self.last_picked_idx is not None:
                remove_at_node()
                idx = self.last_picked_idx
                
                any_fixed = False
                for i, axis in enumerate(['x', 'y', 'z']):
                    # ce je slider na -1 je ta smer fiksna (constraint = 1)
                    if self.roller_angles[axis] == -1:
                        r = np.zeros(n_dof)
                        r[6*idx + i] = 1  # constraint na 1 za to os
                        self.temp_rows.append(r)
                        any_fixed = True
                
                draw_roller_icon(idx)


        def set_fixed():
            if self.last_picked_idx is not None:
                remove_at_node()
                for i in range(6):
                    row = np.zeros(n_dof)
                    row[6*self.last_picked_idx + i] = 1
                    self.temp_rows.append(row)
                draw_fixed_icon(self.last_picked_idx)

        # izbiranje vozlišč
        p.enable_point_picking(
            callback=pick_callback, 
            show_message=False,
            left_clicking=False,
            pickable_window=True
        )
        # tipke
        p.add_key_event('1', set_pinned)
        p.add_key_event('2', set_roller)
        p.add_key_event('3', set_fixed)
        p.add_key_event('4', remove_at_node)

        p.show()

        if self.temp_rows:
            self.constraints = np.array(self.temp_rows)
            print("Constraints updated. Solver will be updated.")
            self._update_solver()
        else:
            self.constraints = None
            print("All constraints removed.")

def calculate_K_glob_frame3d(nodes, elements, A, E, G, Iy, Iz, J):
    n_nodes = len(nodes)
    K_glob = np.zeros((6*n_nodes, 6*n_nodes))

    for i, (n1, n2) in enumerate(elements):
        p1, p2 = nodes[n1], nodes[n2]
        L = np.linalg.norm(p2 - p1)
        
        ke = np.zeros((12, 12))
        
        # osna togost
        ae_l = E * A / L
        ke[0,0] = ke[6,6] = ae_l
        ke[0,6] = ke[6,0] = -ae_l
        
        # torzijska togost
        gj_l = G * J / L
        ke[3,3] = ke[9,9] = gj_l
        ke[3,9] = ke[9,3] = -gj_l
        
        # upogib okoli lokalne osi Z
        iz_l = E * Iz / L**3
        ke[1,1] = ke[7,7] = 12 * iz_l
        ke[1,7] = ke[7,1] = -12 * iz_l
        ke[1,5] = ke[5,1] = ke[1,11] = ke[11,1] = 6 * E * Iz / L**2
        ke[5,7] = ke[7,5] = ke[11,7] = ke[7,11] = -6 * E * Iz / L**2
        ke[5,5] = ke[11,11] = 4 * E * Iz / L
        ke[5,11] = ke[11,5] = 2 * E * Iz / L

        # upogib okoli lokalne osi Y
        iy_l = E * Iy / L**3
        ke[2,2] = ke[8,8] = 12 * iy_l
        ke[2,8] = ke[8,2] = -12 * iy_l
        ke[2,4] = ke[4,2] = ke[2,10] = ke[10,2] = -6 * E * Iy / L**2
        ke[4,8] = ke[8,4] = ke[10,8] = ke[8,10] = 6 * E * Iy / L**2
        ke[4,4] = ke[10,10] = 4 * E * Iy / L
        ke[4,10] = ke[10,4] = 2 * E * Iy / L

        T = get_transformation_matrix_3d(p1, p2)
        ke_g = T.T @ ke @ T
        
        dofs = np.concatenate([np.arange(n1*6, n1*6+6), np.arange(n2*6, n2*6+6)])
        M_idx = np.ix_(dofs, dofs)
        K_glob[M_idx] += ke_g

    return K_glob




def calculate_M_glob_frame3d(nodes, elements, A, rho, J):
    n_nodes = len(nodes)
    M_glob = np.zeros((6*n_nodes, 6*n_nodes))

    for i, (n1, n2) in enumerate(elements):
        p1, p2 = nodes[n1], nodes[n2]
        L = np.linalg.norm(p2 - p1)
        
        me = np.zeros((12, 12))
        c1 = (rho * A * L) / 420
        
        # osna vztrajnost
        me[0,0] = me[6,6] = 140 * c1; me[0,6] = me[6,0] = 70 * c1
        
        # rotacijska vztrajnost torzije
        c_tor = (rho * J * L) / 3
        me[3,3] = me[9,9] = c_tor; me[3,9] = me[9,3] = c_tor / 2
        
        # upogibna vztrajnost
        for off in [0, 1]: # 0 za ravnino xy, 1 za ravnino xz
            d = off + 1
            r = 5 if off == 0 else 4 # phi_z ali phi_y
            s = 1 if off == 0 else -1 # predznak zaradi desnega KS
            
            me[d, d] = me[d+6, d+6] = 156 * c1
            me[d, r] = me[r, d] = s * 22 * L * c1
            me[d, r+6] = me[r+6, d] = -s * 13 * L * c1
            me[r, r] = me[r+6, r+6] = 4 * L**2 * c1
            me[r, d+6] = me[d+6, r] = s * 13 * L * c1
            me[r, r+6] = me[r+6, r] = -3 * L**2 * c1
            me[d+6, r+6] = me[r+6, d+6] = -s * 22 * L * c1
            me[d, d+6] = me[d+6, d] = 54 * c1

        T = get_transformation_matrix_3d(p1, p2)
        me_g = T.T @ me @ T
        
        dofs = np.concatenate([np.arange(n1*6, n1*6+6), np.arange(n2*6, n2*6+6)])
        M_glob[np.ix_(dofs, dofs)] += me_g

    return M_glob

def get_transformation_matrix_3d(p1, p2):
    L = np.linalg.norm(p2 - p1)
    ex = (p2 - p1) / L  # lokalna os x
    
    # lokalna os z 
    # ce je navpična, uporabimo os y, drugace os z.
    if abs(ex[0]) < 1e-6 and abs(ex[1]) < 1e-6: # nvpična palica
        up = np.array([0, 1, 0])
    else:
        up = np.array([0, 0, 1])
        
    ez = np.cross(ex, up)
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    
    R = np.vstack([ex, ey, ez]) # 3x3 rotacijska matrika
    T = np.zeros((12, 12))
    for j in range(4):
        T[j*3:(j+1)*3, j*3:(j+1)*3] = R
    return T