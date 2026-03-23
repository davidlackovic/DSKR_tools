import numpy as np
import scipy as sp


class Combine2D():
    def __init__(self, *objects, connection_type="hinged"):
        self.objects = objects
        self.connection_type = connection_type
        
        self.update_solver()

    def update_solver(self):
        self.skupna_vozlisca = self.find_shared_nodes()
        
        self.M_cela, self.K_cela, self.C, self.offsets = self.combine()

        eig_val, eig_vec_q = sp.linalg.eig(self.K_cela, self.M_cela)
        
        # sortiramo
        idx = np.argsort(np.abs(eig_val))
        eig_val = np.real(eig_val[idx])
        eig_vec_q = eig_vec_q[:, idx]
        
        # frekvence v Hz
        self.eig_freq = np.sqrt(np.maximum(eig_val, 0)) / (2 * np.pi)
        self.eig_freq = self.eig_freq.round(3)
        
        # preslikava nazaj
        self.eig_vec = self.L @ eig_vec_q


    def find_shared_nodes(self, tol=1e-6):
        """
        Find nodes across objects that share the same coordinates.
        Returns a list of tuples: [{'global_id': 0, 'nodes': [(object_index, connected_node), (object_index, connected_node)]}]
        """
        vsa_vozlisca = []
        for obj_idx, obj in enumerate(self.objects):
            for node_idx, nodes in enumerate(obj.nodes):
                vsa_vozlisca.append((obj_idx, node_idx, nodes)) # (iz katere konstrukcije, indeks vozlisca, koordinate)


        used = [False] * len(vsa_vozlisca)
        global_id_counter = 0
        
        node_groups = []

        for i in range(len(vsa_vozlisca)):
            if used[i] == True:
                continue
                
            current_group = [vsa_vozlisca[i][:2]] # samo (obj_idx, node_idx)
            used[i] = True
            
            # primerjamo koordinate z ostalimi vozlisci, ki se niso True v used
            for j in range(i + 1, len(vsa_vozlisca)):
                if not used[j]:
                    dist = np.linalg.norm(vsa_vozlisca[i][2] - vsa_vozlisca[j][2])
                    if dist < tol:
                        current_group.append(vsa_vozlisca[j][:2]) # dodamo (obj_idx, node_idx) vozlisca, ki se prekriva
                        used[j] = True # to vozlisce je ze porabljeno
            
            # ce je v current group vec kot eno vozlisce, ki ima iste koordinate
            if len(current_group) > 1:
                node_groups.append({
                    'global_id': global_id_counter,
                    'nodes': current_group # list vseh (obj_idx, node_idx) ki se prekrivajo
                })
                global_id_counter += 1
                
        return node_groups


    def combine(self):
        """
        Combines multiple structural objects using shared node groups from find_shared_nodes().
        
        """
        # koliko PS ima vsak objekt (dimenzija masne matrike)
        dofs_per_obj = [obj.M_glob.shape[0] for obj in self.objects]
        offsets = np.insert(np.cumsum(dofs_per_obj), 0, 0)
        total_dofs = offsets[-1]

        # matrike C iz vsakega objekta spravimo v blok diagonalno matriko
        individual_constraints = [obj.constraints for obj in self.objects]
        C_internal = sp.linalg.block_diag(*individual_constraints)

        coupling_rows = []
        for group in self.skupna_vozlisca:
            nodes = group['nodes']
            master_obj_idx, master_node_idx = nodes[0] # master je prvi v listu, naslednji slave
            
            # ce je frame (ux, uy, phi), za truss (ux, uy)
            master_dpa = 3 if self.objects[master_obj_idx].type == 'frame' else 2
            master_base_idx = offsets[master_obj_idx] + master_node_idx * master_dpa
            
            # vse slave vozlisca blokiramo, isto 3 za frame, 2 za truss
            for slave_obj_idx, slave_node_idx in nodes[1:]:
                slave_dpa = 3 if self.objects[slave_obj_idx].type == 'frame' else 2
                slave_base_idx = offsets[slave_obj_idx] + slave_node_idx * slave_dpa
                
                # ux
                row_u = np.zeros(total_dofs)
                row_u[master_base_idx] = 1
                row_u[slave_base_idx] = -1
                coupling_rows.append(row_u)
                
                # uy
                row_v = np.zeros(total_dofs)
                row_v[master_base_idx + 1] = 1
                row_v[slave_base_idx + 1] = -1
                coupling_rows.append(row_v)
                
                # phi, samo ce sta oba frame
                if self.connection_type == "rigid" and master_dpa == 3 and slave_dpa == 3:
                    row_t = np.zeros(total_dofs)
                    row_t[master_base_idx + 2] = 1
                    row_t[slave_base_idx + 2] = -1
                    coupling_rows.append(row_t)

        C_AB = np.array(coupling_rows) if coupling_rows else np.empty((0, total_dofs))

        # sestavimo C
        C_cela = np.vstack([C_internal, C_AB])
        self.L = sp.linalg.null_space(C_cela)

        # zdruzimo masne in togostne matrike
        M_block = sp.linalg.block_diag(*[obj.M_glob for obj in self.objects])
        K_block = sp.linalg.block_diag(*[obj.K_glob for obj in self.objects])
        
        M_glob_AB = self.L.T @ M_block @ self.L
        K_glob_AB = self.L.T @ K_block @ self.L
        
        return M_glob_AB, K_glob_AB, C_cela, offsets
    


    def animate_mode_shapes(self, scale=1.0):
        """
        Animates mode shapes of a truss structure according to eigenvectors of the truss.

        """
        import open3d as o3d
        import numpy as np
        import time

        # sestavimo geometrijo nazaj iz locenih objektov
        all_nodes = []
        all_elements = []
        node_offset = 0

        for obj in self.objects:
            all_nodes.append(obj.nodes)
            all_elements.extend(obj.elements + node_offset)
            node_offset += len(obj.nodes)

        nodes_combined = np.vstack(all_nodes)
        elements_combined = all_elements

        pts_orig = np.zeros((len(nodes_combined), 3))
        pts_orig[:, :2] = nodes_combined
        
        # ostala logika ista kot pri Frame2D
        
        lines = [[e[0], e[1]] for e in elements_combined]
        
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
        vis.create_window(window_name="Frame2D - Modal Analysis (Open3D Engine)", width=1200, height=800)
        
        vis.add_geometry(line_set_orig)
        vis.add_geometry(line_set_deformed)
        vis.add_geometry(points_cloud)
        
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])
        opt.line_width = 10.0
        opt.point_size = 12.0

        def get_displacement(m_idx):
            mode = self.eig_vec[:, int(m_idx)]
            disp_list = []
            for i, obj in enumerate(self.objects):
                start, end = self.offsets[i], self.offsets[i+1]
                dpa = 3 if obj.type == 'frame' else 2
                obj_mode = mode[start:end]
                disp_list.append(np.column_stack((obj_mode[0::dpa], obj_mode[1::dpa], np.zeros(len(obj_mode)//dpa))))
            
            disp = np.vstack(disp_list)
            model_size = np.max(np.ptp(pts_orig, axis=0)) if np.any(pts_orig) else 1.0
            max_val = np.max(np.linalg.norm(disp, axis=1))
            if max_val > 1e-12:
                return disp * (model_size * 0.15 / max_val) * scale
            return disp

        state = {
            't': 0.0,
            'animate': True,
            'current_mode': 0,
            'active_disp': get_displacement(0)
        }

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

        vis.register_key_callback(ord(' '), toggle_anim)
        vis.register_key_callback(262, next_mode) 
        vis.register_key_callback(263, prev_mode) 
        vis.register_key_callback(ord('D'), next_mode)
        vis.register_key_callback(ord('A'), prev_mode)

        print("\nKONTROLE: [Preslednica] Start/Stop | [A/D] ali [<- / ->] Menjava načina | [Q] Izhod\n")
        update_info()

        try:
            while True:
                if state['animate']:
                    state['t'] += 0.15
                    deformed_pts = pts_orig + state['active_disp'] * np.sin(state['t'])
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