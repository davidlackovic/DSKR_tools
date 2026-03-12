import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from typing import Optional, Union
import pyvista as pv



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

        # 6 prostostnih stopenj na vozlišče
        self.n_dof = 6 * len(self.nodes)
        self.constraints = constraints if constraints is not None else np.empty((0, self.n_dof))
        self._update_solver()


    def _update_solver(self):
        # Klic 3D funkcij s skalarnimi parametri
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
        import open3d as o3d
        import numpy as np
        import time

        pts_orig = self.nodes.copy()
        
        # Pripravi podatke za linije
        lines = []
        for e in self.elements:
            lines.append([e[0], e[1]])
        
        # Ustvari geometrijo za nedeformirano mrežo (siva, tanka)
        line_set_orig = o3d.geometry.LineSet()
        line_set_orig.points = o3d.utility.Vector3dVector(pts_orig)
        line_set_orig.lines = o3d.utility.Vector2iVector(lines)
        
        # Ustvari geometrijo za deformirano mrežo (modra, debela)
        line_set_deformed = o3d.geometry.LineSet()
        line_set_deformed.points = o3d.utility.Vector3dVector(pts_orig)
        line_set_deformed.lines = o3d.utility.Vector2iVector(lines)
        
        # Pripravi barve za linije
        colors_orig = [[0.7, 0.7, 0.7] for _ in range(len(lines))]  # siva
        colors_deformed = [[0.0, 0.4, 1.0] for _ in range(len(lines))]  # modra
        
        line_set_orig.colors = o3d.utility.Vector3dVector(colors_orig)
        line_set_deformed.colors = o3d.utility.Vector3dVector(colors_deformed)
        
        # USTVARI PIKICE NA VOZLIŠČIH (rdeče)
        points_cloud = o3d.geometry.PointCloud()
        points_cloud.points = o3d.utility.Vector3dVector(pts_orig)
        colors_points = [[1.0, 0.2, 0.2] for _ in range(len(pts_orig))]  # rdeče
        points_cloud.colors = o3d.utility.Vector3dVector(colors_points)
        
        # === USTVARI VIZUALIZER ===
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Frame3D - Modal Analysis with Nodes and Axes", width=1200, height=800)
        
        # Dodaj osnovno geometrijo
        vis.add_geometry(line_set_orig)
        vis.add_geometry(line_set_deformed)
        vis.add_geometry(points_cloud)
        
        # === KOORDINATNE OSI ===
        # TODO: fix arrows on coordinate frame

        axis_len = np.max(np.abs(pts_orig)) * 0.3  # dolžina osi (30% velikosti modela)

        # Ustvari puščice za koordinatne osi
        # Opomba: create_arrow() privzeto kaže v smer +Y

        # X os - rdeča (smer +x) - rotacija okrog Z za -90°
        arrow_x = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=axis_len*0.02,
            cone_radius=axis_len*0.04,
            cylinder_height=axis_len*0.8,
            cone_height=axis_len*0.2
        )
        R_x = arrow_x.get_rotation_matrix_from_xyz((0, 0, -np.pi/2))  # rotacija okrog Z za -90° (Y->X)
        arrow_x.rotate(R_x, center=(0, 0, 0))
        arrow_x.paint_uniform_color([1.0, 0.0, 0.0])  # rdeča

        # Y os - zelena (smer +y) - privzeta smer, ni rotacije
        arrow_y = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=axis_len*0.02,
            cone_radius=axis_len*0.04,
            cylinder_height=axis_len*0.8,
            cone_height=axis_len*0.2
        )
        arrow_y.paint_uniform_color([0.0, 1.0, 0.0])  # zelena

        # Z os - modra (smer +z) - rotacija okrog X za +90°
        arrow_z = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=axis_len*0.02,
            cone_radius=axis_len*0.04,
            cylinder_height=axis_len*0.8,
            cone_height=axis_len*0.2
        )
        R_z = arrow_z.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))  # rotacija okrog X za +90° (Y->Z)
        arrow_z.rotate(R_z, center=(0, 0, 0))
        arrow_z.paint_uniform_color([0.0, 0.0, 1.0])  # modra

        # Dodaj jih v vizualizer
        vis.add_geometry(arrow_x)
        vis.add_geometry(arrow_y)
        vis.add_geometry(arrow_z)

        # Dodaj tanke črte za podaljške (te so pravilne)
        line_x = o3d.geometry.LineSet()
        line_x.points = o3d.utility.Vector3dVector([[-axis_len*0.2, 0, 0], [axis_len*1.2, 0, 0]])
        line_x.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_x.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])

        line_y = o3d.geometry.LineSet()
        line_y.points = o3d.utility.Vector3dVector([[0, -axis_len*0.2, 0], [0, axis_len*1.2, 0]])
        line_y.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_y.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]])

        line_z = o3d.geometry.LineSet()
        line_z.points = o3d.utility.Vector3dVector([[0, 0, -axis_len*0.2], [0, 0, axis_len*1.2]])
        line_z.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_z.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 1.0]])

        vis.add_geometry(line_x)
        vis.add_geometry(line_y)
        vis.add_geometry(line_z)

        
        # Nastavitve izgleda
        render_option = vis.get_render_option()
        render_option.line_width = 8.0  # debele črte za palice
        render_option.point_size = 10.0  # velikost pikic
        render_option.background_color = np.array([1.0, 1.0, 1.0])  # belo ozadje

        
        # Pripravi funkcijo za izračun deformacije
        def get_displacement(m_idx):
            m_idx = int(round(m_idx))
            mode = self.eig_vec[:, m_idx]
            disp = np.column_stack((mode[0::6], mode[1::6], mode[2::6]))
            model_size = np.max(np.ptp(pts_orig, axis=0)) if np.any(pts_orig) else 1.0
            max_val = np.max(np.linalg.norm(disp, axis=1))
            return disp * (model_size * 0.15 / max_val) * scale if max_val > 1e-12 else disp
        
        # Stanje animacije
        state = {
            't': 0.0,
            'animate': False,
            'current_mode': 0,
            'active_disp': get_displacement(0)
        }
        
        # Funkcija za posodobitev naslova
        def update_window_title():
            anim_status = "ON" if state['animate'] else "OFF"
            print(f"\rMode: {state['current_mode']} | Freq: {self.eig_freq[state['current_mode']]:.2f} Hz | Animacija: {anim_status}", end="", flush=True)
        
        update_window_title()
        
        # Callbacki za tipke
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
        
        # Registriraj tipke
        vis.register_key_callback(ord(' '), toggle_animation)
        vis.register_key_callback(ord('D'), next_mode)
        vis.register_key_callback(ord('d'), next_mode)
        vis.register_key_callback(ord('A'), prev_mode)
        vis.register_key_callback(ord('a'), prev_mode)
        vis.register_key_callback(262, next_mode)  # Desna puščica
        vis.register_key_callback(263, prev_mode)  # Leva puščica
        vis.register_key_callback(ord('Q'), exit_visualizer)
        vis.register_key_callback(ord('q'), exit_visualizer)
        vis.register_key_callback(256, exit_visualizer)  # ESC
        
        print("\n" + "="*60)
        print("Frame3D - Modal Analysis with Nodes and Axes")
        print("="*60)
        print("NAVODILA:")
        print("  PRESLEDNICA: vklop/izklop animacije")
        print("  A/←: prejšnji mode")
        print("  D/→: naslednji mode")
        print("  Q/ESC: izhod")
        print("\nKOORDINATNE OSI:")
        print("  Rdeča: X os")
        print("  Zelena: Y os")
        print("  Modra: Z os")
        print("="*60 + "\n")
        
        # Glavna zanka
        try:
            while True:
                if state['animate']:
                    state['t'] += 0.15
                    
                    # Izračunaj deformirane točke
                    factor = np.sin(state['t'])
                    deformed_pts = pts_orig + state['active_disp'] * factor
                    
                    # Posodobi linije
                    line_set_deformed.points = o3d.utility.Vector3dVector(deformed_pts)
                    vis.update_geometry(line_set_deformed)
                    
                    # Posodobi pikice na vozliščih
                    points_cloud.points = o3d.utility.Vector3dVector(deformed_pts)
                    vis.update_geometry(points_cloud)
                
                # Posodobi vizualizacijo
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
        ''' Prikaz nedeformiranega paličja z Open3D - najbolj preprosta verzija '''
        import open3d as o3d
        import numpy as np

        pts = self.nodes.copy()
        
        # Pripravi podatke za linije
        lines = []
        for e in self.elements:
            lines.append([e[0], e[1]])
        
        # Ustvari geometrijo za paličje
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pts)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Črne črte
        colors = [[0.0, 0.0, 0.0] for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        # Vizualizacija
        o3d.visualization.draw_geometries(
            [line_set],
            window_name=title,
            width=1200,
            height=800,
            point_show_normal=False
        )
    

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

        self.font_size = 21
        self.y_pos = 0.03
        axis_len = np.max(np.abs(self.nodes)) * 0.25 if np.any(self.nodes) else 1.0
        self.r_size = axis_len * 0.03  # radij sfer

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

        # Prikaz paličja
        p.add_mesh(mesh, color="#555555", line_width=2, render_lines_as_tubes=True)
        p.add_point_labels(pts, [f"{i}" for i in range(len(self.nodes))], 
                        point_size=10, font_size=18, text_color="black", 
                        always_visible=True, name="node_labels", shadow=True)

        # Prikaz koordinatnih osi (3D)
        axes = pv.Axes(show_actor=True, actor_scale=axis_len, line_width=3)
        p.add_actor(axes.actor)

        # Uporabniški vmesnik - tekst
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

        # Sliderji za kote roller podpore
        self.roller_angles = {'x': 0, 'y': 0, 'z': 0}
        
        def update_angle_texts():
            p.remove_actor("angle_info")
            angle_text = f"Roller angles: X={self.roller_angles['x']}°  Y={self.roller_angles['y']}°  Z={self.roller_angles['z']}°"
            p.add_text(angle_text, position=(0.75, 0.25), 
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
        
        # Dodaj sliderje
        p.add_slider_widget(callback=slider_x_callback, rng=[0, 180], value=0,
                            pointa=(0.75, 0.20), pointb=(0.95, 0.20), 
                            style='modern', color="royalblue",
                            tube_width=0.003, slider_width=0.02, 
                            title="X angle:")
        
        p.add_slider_widget(callback=slider_y_callback, rng=[0, 180], value=0,
                            pointa=(0.75, 0.15), pointb=(0.95, 0.15), 
                            style='modern', color="royalblue",
                            tube_width=0.003, slider_width=0.02, 
                            title="Y angle:")
        
        p.add_slider_widget(callback=slider_z_callback, rng=[0, 180], value=0,
                            pointa=(0.75, 0.10), pointb=(0.95, 0.10), 
                            style='modern', color="royalblue",
                            tube_width=0.003, slider_width=0.02, 
                            title="Z angle:")
        
        update_angle_texts()

        # Nastavitev pogleda
        p.camera_position = 'iso'
        p.camera.zoom(0.8)

        self.last_picked_idx = None
        n_dof = 6 * len(self.nodes)

        # Poenostavljene ikone - samo krogle!
        def draw_pinned_icon(node_idx):
            p.add_mesh(pv.Sphere(radius=self.r_size, center=pts[node_idx]), 
                    color="firebrick", name=f"pinned_{node_idx}")
        
        def draw_roller_icon(node_idx):
            p.add_mesh(pv.Sphere(radius=self.r_size, center=pts[node_idx]), 
                    color="royalblue", name=f"roller_{node_idx}")
            
            # Prikaži kote poleg
            offset = pts[node_idx] + [axis_len*0.15, axis_len*0.15, axis_len*0.15]
            angle_text = f"{self.roller_angles['x']}°/{self.roller_angles['y']}°/{self.roller_angles['z']}°"
            p.add_point_labels([offset], [angle_text], 
                            font_size=14, text_color="royalblue", 
                            name=f"angle_label_{node_idx}", 
                            shape=None, always_visible=True, shadow=True)
        
        def draw_fixed_icon(node_idx):
            p.add_mesh(pv.Sphere(radius=self.r_size*1.2, center=pts[node_idx]), 
                    color="forestgreen", name=f"fixed_{node_idx}")

        # Nariši obstoječe podpore
        if self.temp_rows:
            processed_nodes = set()
            for row in self.temp_rows:
                active_dofs = np.where(np.abs(row) > 1e-6)[0]
                if len(active_dofs) == 0: continue
                
                node_idx = active_dofs[0] // 6
                if node_idx in processed_nodes: continue
                
                dofs_at_node = [dof % 6 for dof in active_dofs if dof // 6 == node_idx]
                
                if set(dofs_at_node) == {0, 1, 2}:  # pinned (ux, uy, uz)
                    draw_pinned_icon(node_idx)
                elif len(dofs_at_node) == 1 and dofs_at_node[0] in [0, 1, 2]:  # roller (ena os prosta)
                    draw_roller_icon(node_idx)
                elif set(dofs_at_node) == {0, 1, 2, 3, 4, 5}:  # fixed
                    draw_fixed_icon(node_idx)
                
                processed_nodes.add(node_idx)

        def update_status_text():
            p.remove_actor("st_node")
            if self.last_picked_idx is not None:
                p.add_text(f"Node: {self.last_picked_idx}", position=(0.02, 0.92), 
                        color='black', font_size=self.font_size+10, name="st_node", 
                        shadow=True, viewport=True)

        def pick_callback(point_data, *args):
            idx = mesh.find_closest_point(point_data)
            self.last_picked_idx = idx
            # Označi izbrano vozlišče
            p.add_mesh(pv.Sphere(radius=self.r_size*1.5, center=pts[idx]), 
                    color="yellow", opacity=0.3, name="selection_glow")
            p.add_mesh(pv.Sphere(radius=self.r_size*0.3, center=pts[idx]), 
                    color="black", name="selection_center")
            update_status_text()

        def remove_at_node():
            if self.last_picked_idx is not None:
                idx = self.last_picked_idx
                dofs = [6*idx + i for i in range(6)]
                
                # Odstrani vrstice
                self.temp_rows = [r for r in self.temp_rows 
                                if all(np.abs(r[d]) < 1e-6 for d in dofs)]
                
                # Odstrani vse ikone
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
                
                # Ustvari roller constraint glede na kote
                # Tukaj definiramo, katera os je prosta glede na kote
                # Za primer: če so vsi koti 0, je prosta X os
                angles = self.roller_angles
                
                # Poenostavimo: za zdaj vedno naredimo roller v smeri X
                # (lahko prilagodiš glede na želeno logiko)
                r = np.zeros(n_dof)
                r[6*self.last_picked_idx] = 1  # ux prost
                self.temp_rows.append(r)
                
                draw_roller_icon(self.last_picked_idx)

        def set_fixed():
            if self.last_picked_idx is not None:
                remove_at_node()
                for i in range(6):
                    row = np.zeros(n_dof)
                    row[6*self.last_picked_idx + i] = 1
                    self.temp_rows.append(row)
                draw_fixed_icon(self.last_picked_idx)

        # Omogoči izbiranje vozlišč
        p.enable_point_picking(
            callback=pick_callback, 
            show_message=False,
            left_clicking=False,
            pickable_window=True
        )

        # Dodaj tipkovne bližnjice
        p.add_key_event('1', set_pinned)
        p.add_key_event('2', set_roller)
        p.add_key_event('3', set_fixed)
        p.add_key_event('4', remove_at_node)

        # Prikaži
        p.show()

        # Shrani spremembe
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
        
        # --- Lokalna togostna matrika (12x12) ---
        ke = np.zeros((12, 12))
        
        # Osna togost (u)
        ae_l = E * A / L
        ke[0,0] = ke[6,6] = ae_l
        ke[0,6] = ke[6,0] = -ae_l
        
        # Torzijska togost (phi_x)
        gj_l = G * J / L
        ke[3,3] = ke[9,9] = gj_l
        ke[3,9] = ke[9,3] = -gj_l
        
        # Upogib okoli lokalne osi Z (v, phi_z)
        iz_l = E * Iz / L**3
        ke[1,1] = ke[7,7] = 12 * iz_l
        ke[1,7] = ke[7,1] = -12 * iz_l
        ke[1,5] = ke[5,1] = ke[1,11] = ke[11,1] = 6 * E * Iz / L**2
        ke[5,7] = ke[7,5] = ke[11,7] = ke[7,11] = -6 * E * Iz / L**2
        ke[5,5] = ke[11,11] = 4 * E * Iz / L
        ke[5,11] = ke[11,5] = 2 * E * Iz / L

        # Upogib okoli lokalne osi Y (w, phi_y)
        iy_l = E * Iy / L**3
        ke[2,2] = ke[8,8] = 12 * iy_l
        ke[2,8] = ke[8,2] = -12 * iy_l
        ke[2,4] = ke[4,2] = ke[2,10] = ke[10,2] = -6 * E * Iy / L**2
        ke[4,8] = ke[8,4] = ke[10,8] = ke[8,10] = 6 * E * Iy / L**2
        ke[4,4] = ke[10,10] = 4 * E * Iy / L
        ke[4,10] = ke[10,4] = 2 * E * Iy / L

        # --- Transformacija ---
        T = get_transformation_matrix_3d(p1, p2)
        ke_g = T.T @ ke @ T
        
        # --- Sestavljanje ---
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
        
        # --- Lokalna konsistentna masna matrika (12x12) ---
        me = np.zeros((12, 12))
        c1 = (rho * A * L) / 420
        
        # Osni del (u)
        me[0,0] = me[6,6] = 140 * c1; me[0,6] = me[6,0] = 70 * c1
        
        # Torzijski del (phi_x) - rotacijska vztrajnost okoli osi
        c_tor = (rho * J * L) / 3
        me[3,3] = me[9,9] = c_tor; me[3,9] = me[9,3] = c_tor / 2
        
        # Upogibni del (v, phi_z) in (w, phi_y)
        # Za oba upogiba so koeficienti v konsistentni matriki enaki
        for off in [0, 1]: # 0 za ravnino xy, 1 za ravnino xz
            d = off + 1
            r = 5 if off == 0 else 4 # phi_z ali phi_y
            s = 1 if off == 0 else -1 # predznak za določene člene zaradi desnosučnega sistema
            
            me[d, d] = me[d+6, d+6] = 156 * c1
            me[d, r] = me[r, d] = s * 22 * L * c1
            me[d, r+6] = me[r+6, d] = -s * 13 * L * c1
            me[r, r] = me[r+6, r+6] = 4 * L**2 * c1
            me[r, d+6] = me[d+6, r] = s * 13 * L * c1
            me[r, r+6] = me[r+6, r] = -3 * L**2 * c1
            me[d+6, r+6] = me[r+6, d+6] = -s * 22 * L * c1
            me[d, d+6] = me[d+6, d] = 54 * c1

        # --- Transformacija ---
        T = get_transformation_matrix_3d(p1, p2)
        me_g = T.T @ me @ T
        
        # --- Sestavljanje ---
        dofs = np.concatenate([np.arange(n1*6, n1*6+6), np.arange(n2*6, n2*6+6)])
        M_glob[np.ix_(dofs, dofs)] += me_g

    return M_glob

def get_transformation_matrix_3d(p1, p2):
    L = np.linalg.norm(p2 - p1)
    ex = (p2 - p1) / L  # Lokalna os x (smer palice)
    
    # Določitev lokalne osi z (up vector). 
    # Če je palica navpična, uporabimo os Y, sicer os Z.
    if abs(ex[0]) < 1e-6 and abs(ex[1]) < 1e-6: # Navpična palica
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