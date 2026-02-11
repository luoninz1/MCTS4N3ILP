
import numpy as np
from numba import njit
from src.envs.collinear_for_mcts import N3il_with_symmetry, filter_top_priority_moves
from src.envs.n3il_symmetric_action import N3il_with_symmetry_and_symmetric_actions, get_d4_orbit_nb
from src.rewards.n3il_rewards import get_value_nb

@njit(cache=True)
def are_collinear_3_nb(r1, c1, r2, c2, r3, c3):
    return (r2 - r1) * (c3 - c2) == (r3 - r2) * (c2 - c1)

@njit(cache=True)
def check_concyclic_strict_nb(r1, c1, r2, c2, r3, c3, r4, c4):
    """
    Returns True if the 4 points are on a circle. 
    Strictly: returns False if adjacent to a line (3 collinear) or all 4 collinear.
    """
    s1 = r1 * r1 + c1 * c1
    s2 = r2 * r2 + c2 * c2
    s3 = r3 * r3 + c3 * c3
    s4 = r4 * r4 + c4 * c4
    
    # Calculate minors corresponding to the last row (r4, c4) of the 4x4 determinant
    # Actually, we just need the full determinant.
    # We use cofactor expansion logic on the fly or just the full formula.
    # Let's trust the formula derived previously.
    
    # Determinant of:
    # | s1 r1 c1 1 |
    # | s2 r2 c2 1 |
    # | s3 r3 c3 1 |
    # | s4 r4 c4 1 |
    
    # Cofactors for first column s1, s2, s3, s4:
    # C1 =  | r2 c2 1 | = r2(c3-c4) - c2(r3-r4) + (r3*c4 - r4*c3)
    #       | r3 c3 1 |
    #       | r4 c4 1 |
    d1 = r2 * (c3 - c4) + c2 * (r4 - r3) + (r3 * c4 - r4 * c3)
    
    # C2 = -| r1 c1 1 | = -( r1(c3-c4) - c1(r3-r4) + (r3*c4 - r4*c3) )
    #       | r3 c3 1 |
    #       | r4 c4 1 |
    d2 = r1 * (c3 - c4) + c1 * (r4 - r3) + (r3 * c4 - r4 * c3)
    
    # C3 =  | r1 c1 1 | = r1(c2-c4) - c1(r2-r4) + (r2*c4 - r4*c2)
    #       | r2 c2 1 |
    #       | r4 c4 1 |
    d3 = r1 * (c2 - c4) + c1 * (r4 - r2) + (r2 * c4 - r4 * c2)
    
    # C4 = -| r1 c1 1 | = -( r1(c2-c3) - c1(r2-r3) + (r2*c3 - r3*c2) )
    #       | r2 c2 1 |
    #       | r3 c3 1 |
    d4 = r1 * (c2 - c3) + c1 * (r3 - r2) + (r2 * c3 - r3 * c2)
    
    det = s1 * d1 - s2 * d2 + s3 * d3 - s4 * d4
    
    if det != 0:
        return False
        
    # If det == 0, it is either a circle or a line (generalized circle).
    # If it is a circle, no 3 points can be collinear.
    # If 3 points ARE collinear, then it must be the line case (or degenerate line cases).
    # Since we only want to ban "On a Circle", and allow "On a Line" (unless n4il is also active, but this is pure No4Circle),
    # we return False (safe/valid) if it looks like a line.
    # Checking if 3 points are collinear is sufficient because:
    #   - If it's a circle, 3 points CANNOT be collinear.
    #   - So if 3 points ARE collinear, it's not a circle.
    if are_collinear_3_nb(r1, c1, r2, c2, r3, c3):
        return False
        
    return True

@njit(cache=True)
def get_valid_moves_no4circle_nb(state, n, m):
    # state: n x m
    mask = np.ones(n * m, dtype=np.int8)
    
    # Get occupied indices
    occupied_indices = []
    flat_state = state.flatten()
    for i in range(n * m):
        if flat_state[i] == 1:
            occupied_indices.append(i)
            
    num_occ = len(occupied_indices)
    
    # If fewer than 3 points, all moves valid (except occupied)
    if num_occ < 3:
        for idx in occupied_indices:
            mask[idx] = 0
        return mask
        
    # Iterate all triplets of existing points
    # For each triplet, check against all other grid points
    # Optimization: Calculate circle params once per triplet
    
    for i in range(num_occ):
        idx1 = occupied_indices[i]
        r1, c1 = idx1 // m, idx1 % m
        s1 = r1*r1 + c1*c1
        
        for j in range(i + 1, num_occ):
            idx2 = occupied_indices[j]
            r2, c2 = idx2 // m, idx2 % m
            s2 = r2*r2 + c2*c2
            
            for k in range(j + 1, num_occ):
                idx3 = occupied_indices[k]
                r3, c3 = idx3 // m, idx3 % m
                
                # Check 3 collinear first - if so, skip (no circle constraint from this triplet)
                if are_collinear_3_nb(r1, c1, r2, c2, r3, c3):
                    continue
                    
                s3 = r3*r3 + c3*c3
                
                # Calculate circle equation coefficients a(x^2+y^2) + bx + cy + d = 0
                # Using 4th point as variable (r, c)
                # det = s4 * (coeff_s4) + r4 * (coeff_r4) + c4 * (coeff_c4) + 1 * (coeff_1) = 0
                # But det expansion above was s1*d1 - s2*d2 ...
                # Let's reuse logic.
                # In det = s1*d1 - s2*d2 + s3*d3 - s4*d4
                # s4 is the variable s term. coeff is -d4.
                # But d4 depended only on 1, 2, 3? 
                # Wait, in the function `check_concyclic_strict_nb`:
                # d4 was calculated using points 1, 2, 3.
                # d4 = r1(c2-c3) ...
                # So `a = -d4`.
                # What about b, c, d coefficients?
                # This is more complex to derive manually correctly.
                # Simpler approach: check every empty spot.
                
                for r_test in range(n):
                    for c_test in range(m):
                        idx_test = r_test * m + c_test
                        if mask[idx_test] == 0: 
                            continue # Already invalid or occupied
                            
                        # Only strictly empty points? 
                        # get_valid_moves should return 0 for occupied too.
                        # occupied are handled by logic or at end.
                        
                        if check_concyclic_strict_nb(r1, c1, r2, c2, r3, c3, r_test, c_test):
                            mask[idx_test] = 0
                            
    # Ensure occupied are 0
    for idx in occupied_indices:
        mask[idx] = 0
        
    return mask

@njit(cache=True)
def get_valid_moves_subset_no4circle_nb(state, current_mask, action, n, m):
    # state has old points.
    # action is the new point being added.
    mask = current_mask.copy()
    
    if mask[action] == 0:
        # Should not happen in standard flow if action was valid
        pass
        
    # Mark action as occupied
    mask[action] = 0
    
    new_r, new_c = action // m, action % m
    
    # Get existing points
    points_rows = []
    points_cols = []
    flat_state = state.flatten()
    for i in range(n * m):
        if flat_state[i] == 1:
            points_rows.append(i // m)
            points_cols.append(i % m)
            
    num_occ = len(points_rows)
    
    # Iterate pairs of existing points
    # new point + 2 existing points -> triplet
    for i in range(num_occ):
        r1, c1 = points_rows[i], points_cols[i]
        
        for j in range(i + 1, num_occ):
            r2, c2 = points_rows[j], points_cols[j]
            
            # Check if triplet (new, 1, 2) is collinear
            if are_collinear_3_nb(new_r, new_c, r1, c1, r2, c2):
                continue
            
            # Check all potentially valid moves against this new circle constraint
            # We can scan the whole grid or just current valid moves
            # Scanning valid moves is likely faster if mask is sparse, 
            # but scanning grid is constant memory access.
            
            for r_test in range(n):
                for c_test in range(m):
                    idx_test = r_test * m + c_test
                    if mask[idx_test] == 0:
                        continue
                        
                    if check_concyclic_strict_nb(new_r, new_c, r1, c1, r2, c2, r_test, c_test):
                        mask[idx_test] = 0
                        
    return mask

@njit(cache=True)
def get_next_state_no4circle_logic_nb(state, action, n, op_codes, current_mask):
    orbit = get_d4_orbit_nb(action, n, op_codes)
    
    # It is expected that current_mask[action] == 1 roughly, 
    # but we might be applying symmetric actions where we haven't checked availability?
    # The MCTS logic usually checks children validity.
    # But for "symmetric actions", we apply the whole orbit.
    # We must check if the whole orbit is valid sequentially.
    
    temp_state = state.copy()
    temp_mask = current_mask.copy()
    possible = True
    
    # Apply each action in orbit
    for act in orbit:
        if temp_state.flatten()[act] == 1:
             # Already occupied by previous action in same orbit?
             # If acts are distinct, and it was 0, it is fine.
             # If acts are same (fixed point), we just set it once.
             continue
             
        if temp_mask[act] == 0:
            possible = False
            break
            
        # Update mask for this action
        temp_mask = get_valid_moves_subset_no4circle_nb(temp_state, temp_mask, act, n, n)
        temp_state[act // n, act % n] = 1
        
    if possible:
        return temp_state
    else:
        # Fallback: Just the single action
        # Note: calling function expects a state return.
        # If the symmetric group add failed, we default to adding JUST the primary action?
        # This is the behavior in N3il_with_symmetry_and_symmetric_actions.
        res = state.copy()
        res[action // n, action % n] = 1
        return res

@njit(cache=True)
def simulate_no4circle_nb(state, n, pts_upper_bound, op_codes):
    current_state = state.copy()
    valid_moves = get_valid_moves_no4circle_nb(current_state, n, n)
    total_valid = np.sum(valid_moves)
    
    while total_valid > 0:
        # Random choice logic
        rand_count = np.random.randint(0, total_valid)
        action = -1
        c = 0
        
        # Find the index of the rand_count-th set bit
        # Flattened iteration
        counter = 0
        found = False
        for i in range(n*n):
            if valid_moves[i] == 1:
                if counter == rand_count:
                    action = i
                    found = True
                    break
                counter += 1
        
        if not found:
             break # Should not happen
             
        # Try orbit application
        orbit = get_d4_orbit_nb(action, n, op_codes)
        
        temp_state = current_state.copy()
        temp_mask = valid_moves.copy()
        orbit_possible = True
        
        for act in orbit:
            if temp_state.flatten()[act] == 1:
                continue
                
            if temp_mask[act] == 0:
                orbit_possible = False
                break
                
            temp_mask = get_valid_moves_subset_no4circle_nb(temp_state, temp_mask, act, n, n)
            temp_state[act // n, act % n] = 1
            
        if orbit_possible:
            current_state = temp_state
            valid_moves = temp_mask
        else:
            # Fallback
            valid_moves = get_valid_moves_subset_no4circle_nb(current_state, valid_moves, action, n, n)
            current_state[action // n, action % n] = 1
            
        total_valid = np.sum(valid_moves)
        
    return get_value_nb(current_state, pts_upper_bound), current_state

class No_4_on_circle(N3il_with_symmetry_and_symmetric_actions):
    def __init__(self, grid_size, args, priority_grid=None):
        super().__init__(grid_size, args, priority_grid)
        self.name = "No_4_on_circle"

    def get_valid_moves(self, state):
        state = np.array(state).reshape(self.row_count, self.column_count)
        valid_moves = get_valid_moves_no4circle_nb(state, self.row_count, self.column_count)
        if self.priority_grid is not None:
             return filter_top_priority_moves(
                valid_moves, 
                self.priority_grid, 
                self.row_count, 
                self.column_count, 
                top_N=self.args['TopN'])
        return valid_moves

    def get_valid_moves_subset(self, parent_state, parent_valid_moves, action_taken, current_state):
        diff = current_state - parent_state
        added_indices = np.argwhere(diff == 1)
        
        valid_moves = parent_valid_moves.copy()
        temp_state = parent_state.copy()
        
        for idx in added_indices:
            r, c = idx
            action = r * self.column_count + c
            
            valid_moves = get_valid_moves_subset_no4circle_nb(
                temp_state, 
                valid_moves, 
                action, 
                self.row_count, 
                self.column_count
            )
            temp_state[r, c] = 1
            
        if self.priority_grid is not None:
            return filter_top_priority_moves(
                valid_moves, 
                self.priority_grid, 
                self.row_count, 
                self.column_count, 
                top_N=self.args['TopN'])
        else:
            return valid_moves

    def get_next_state(self, state, action, action_space_of_state=None):
        s_arr = np.array(state, dtype=np.int64).reshape(self.row_count, self.column_count)
        if action_space_of_state is None:
            action_space_of_state = get_valid_moves_no4circle_nb(s_arr, self.row_count, self.column_count)
            
        new_state = get_next_state_no4circle_logic_nb(s_arr, action, self.row_count, self.op_codes, action_space_of_state)
        return new_state

    def simulate(self, state):
        s_arr = np.array(state, dtype=np.int64).reshape(self.row_count, self.column_count)
        return simulate_no4circle_nb(s_arr, self.row_count, self.pts_upper_bound, self.op_codes)
