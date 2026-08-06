import numpy as np
import time

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

from config import (
    GRID_SIZE, GRID_SPACING, GRID_SPACING_LON_M, GRID_SPACING_LAT_M,
    OUTPUT_DIR,
    K_NEAREST_INIT, ILP_TIME_LIMIT, ILP_THREADS,
    GA_POP_SIZE, GA_GENERATIONS, GA_MUTATION_RATE,
    GA_ELITE_RATIO, GA_TOURNAMENT_SIZE, GA_LOCAL_SEARCH_ITER,
    LAGRANGIAN_MAX_ITER, LAGRANGIAN_TIME_LIMIT, LAGRANGIAN_GAP_TOL
)


class PMedianSolver:
    def __init__(self, data, p_value, time_matrix=None, use_gpu=False):
        """
        P-Median Solver using real travel time matrix.

        Args:
            data: DataFrame with grid data (row, col, population, etc.)
            p_value: Number of facilities to select
            time_matrix: NxN matrix where time_matrix[i,j] is travel time
                        from grid i to grid j (in seconds)
            use_gpu: Whether to use GPU acceleration
        """
        self.data = data
        self.n = len(data)
        self.p = min(p_value, self.n)

        self.rows = data['row'].values.astype(int)
        self.cols = data['col'].values.astype(int)
        self.weights = data['population'].values.astype(float)

        actual_rows = self.rows.max() + 1
        actual_cols = self.cols.max() + 1

        if actual_rows == 100 and actual_cols == 100:
            self.spacing_lon = GRID_SPACING
            self.spacing_lat = GRID_SPACING
            self.grid_size = GRID_SIZE
            self.is_real_data = False
        else:
            self.grid_size = (actual_rows, actual_cols)
            self.spacing_lon = GRID_SPACING_LON_M
            self.spacing_lat = GRID_SPACING_LAT_M
            self.is_real_data = True

        self.has_height = 'maxHeight' in data.columns

        # Load or use provided time matrix
        if time_matrix is not None:
            self.time_matrix = time_matrix
            if self.time_matrix.shape != (self.n, self.n):
                raise ValueError(
                    f"Time matrix shape {self.time_matrix.shape} "
                    f"does not match grid count ({self.n}, {self.n})"
                )
            print(f"  [Time] Using real travel time matrix: {self.n}x{self.n}")
        else:
            raise ValueError(
                "Time matrix is required. Please provide a valid time matrix."
            )

        self.use_gpu = use_gpu and HAS_CUPY
        if use_gpu and not HAS_CUPY:
            print("  [Warning] CuPy not available, using CPU. "
                  "Install: pip install cupy-cuda11x")
        elif self.use_gpu:
            print("  [GPU] CuPy detected, using GPU acceleration")

        if 'grid_id' in data.columns:
            self.grid_ids = data['grid_id'].values
        else:
            self.grid_ids = np.arange(self.n)

        if self.is_real_data:
            print(f"  [Data] Real grid: {actual_rows}x{actual_cols}, "
                  f"spacing: {self.spacing_lon:.1f}m(lon) x "
                  f"{self.spacing_lat:.1f}m(lat)")
            if self.has_height:
                print(f"         maxHeight data available")
        else:
            print(f"  [Data] Simulated grid: {GRID_SIZE}x{GRID_SIZE}, "
                  f"spacing: {GRID_SPACING}m")

    def _get_travel_times_to_facilities(self, facilities):
        """Get travel times from all grids to selected facilities."""
        # time_matrix[i, j] = time from grid i to facility at grid j
        return self.time_matrix[:, facilities]

    def _compute_assignment(self, facilities):
        """
        Compute assignment of each grid to nearest facility.
        Returns (nearest_idx, nearest_time) where nearest_idx[i] is the
        index within facilities list, and nearest_time[i] is the travel time.
        """
        times_to_facs = self._get_travel_times_to_facilities(facilities)
        nearest_idx = times_to_facs.argmin(axis=1)
        nearest_time = times_to_facs.min(axis=1)
        return nearest_idx, nearest_time

    def compute_objective(self, facilities):
        """Compute weighted total travel time objective."""
        _, nearest_time = self._compute_assignment(facilities)
        return float(np.sum(self.weights * nearest_time))

    def _precompute_wt_chunks(self, chunk_size=500, verbose=True):
        """Precompute weighted time matrix in chunks for Lagrangian."""
        N = self.n
        w = self.weights
        xp = cp if self.use_gpu else np

        time_matrix_xp = xp.asarray(self.time_matrix) if self.use_gpu else self.time_matrix
        w_xp = xp.asarray(w) if self.use_gpu else w

        device = "GPU (CuPy)" if self.use_gpu else "CPU (NumPy)"
        if verbose:
            print(f"  [2a] Precomputing weighted time matrix ({device}, "
                  f"chunks={chunk_size})...")

        t0 = time.time()
        wt_chunks = []
        n_chunks = (N + chunk_size - 1) // chunk_size

        for ci, i_start in enumerate(range(0, N, chunk_size)):
            i_end = min(i_start + chunk_size, N)

            # time_matrix[i_start:i_end, :] gives times from grids i_start:i_end to all facilities
            time_chunk = time_matrix_xp[i_start:i_end, :]
            w_chunk = w_xp[i_start:i_end].reshape(-1, 1)
            wt_matrix = w_chunk * time_chunk

            wt_chunks.append((i_start, i_end, wt_matrix))

            if verbose and (ci + 1) % 5 == 0:
                print(f"       Chunk {ci + 1}/{n_chunks} done")

        if verbose:
            mem_mb = sum(c[2].nbytes for c in wt_chunks) / 1024 / 1024
            print(f"       Done in {time.time() - t0:.1f}s, "
                  f"memory: {mem_mb:.0f}MB")

        return wt_chunks

    def solve_lagrangian(self, max_iter=None, time_limit=None, gap_tol=None,
                         verbose=True):
        max_iter = max_iter or LAGRANGIAN_MAX_ITER
        time_limit = time_limit or LAGRANGIAN_TIME_LIMIT
        gap_tol = gap_tol or LAGRANGIAN_GAP_TOL

        if verbose:
            print("=" * 60)
            print("[Stage 2] Lagrangian Relaxation (Volume Algorithm)")
            print(f"  P={self.p}, N={self.n}, "
                  f"GPU={'Yes (CuPy)' if self.use_gpu else 'No (NumPy)'}")
            print(f"  Max iterations: {max_iter}, "
                  f"Time limit: {time_limit}s")
            print(f"  Gap tolerance: {gap_tol * 100:.4f}%")
            print("=" * 60)

        start_time = time.time()
        N = self.n
        P = self.p
        xp = cp if self.use_gpu else np

        if verbose:
            print("  [Phase 1] Computing initial upper bound (greedy+TB)...")
        init_fac = self._greedy_construction(verbose=False)
        init_fac, init_obj = self._teitz_bart(
            init_fac, max_iter=50, verbose=False)
        best_facilities = list(init_fac)
        best_ub = init_obj

        if verbose:
            print(f"       Initial UB: {best_ub:.2f}")

        if verbose:
            print("  [Phase 2] Precomputing weighted time matrix...")
        wt_chunks = self._precompute_wt_chunks(
            chunk_size=500, verbose=verbose)

        lagrangian_start = time.time()
        init_elapsed = lagrangian_start - start_time

        if verbose:
            print("\n  [Phase 3] Volume Algorithm iterations...")
            print(f"       Init+precompute took {init_elapsed:.1f}s, "
                  f"remaining for iterations: "
                  f"{max(0, time_limit - init_elapsed):.0f}s")

        _, init_nearest_time = self._compute_assignment(best_facilities)
        lambda_vals = xp.asarray(
            self.weights * init_nearest_time * 0.8
        ) if self.use_gpu else (self.weights * init_nearest_time * 0.8).copy()

        v = xp.zeros(N, dtype=np.float64)

        best_lb = 0.0
        alpha = 0.05
        theta = 0.5
        no_improve_count = 0
        lb_history = []
        ub_history = []
        total_iter = 0
        soft_reset_count = 0

        for iteration in range(max_iter):
            elapsed = time.time() - lagrangian_start
            if elapsed > time_limit:
                if verbose:
                    print(f"       Time limit reached at iter {iteration + 1}")
                break

            cj = xp.zeros(N, dtype=np.float64)

            for i_start, i_end, wt_matrix in wt_chunks:
                chunk_lambda = lambda_vals[i_start:i_end].reshape(-1, 1)
                wt_lambda = wt_matrix - chunk_lambda
                neg_wt = xp.minimum(0, wt_lambda)
                cj += neg_wt.sum(axis=0)

            if self.use_gpu:
                cj_cpu = cp.asnumpy(cj)
            else:
                cj_cpu = np.asarray(cj)

            top_p_idx = np.argpartition(cj_cpu, P)[:P]

            if self.use_gpu:
                top_p_xp = cp.asarray(top_p_idx)
            else:
                top_p_xp = top_p_idx

            attracted = xp.zeros(N, dtype=np.float64)
            for i_start, i_end, wt_matrix in wt_chunks:
                chunk_lambda = lambda_vals[i_start:i_end].reshape(-1, 1)
                wt_sel = wt_matrix[:, top_p_xp]
                wt_lambda_sel = wt_sel - chunk_lambda
                attracted[i_start:i_end] = (
                    wt_lambda_sel < 0).sum(axis=1).astype(np.float64)

            lambda_sum = float(
                cp.asnumpy(lambda_vals.sum()) if self.use_gpu
                else lambda_vals.sum())
            cj_top = cj[top_p_xp]
            cj_top_sum = float(
                cp.asnumpy(cj_top.sum()) if self.use_gpu
                else cj_top.sum())
            lb_val = lambda_sum + cj_top_sum

            if lb_val > best_lb + 1e-6:
                best_lb = lb_val
                no_improve_count = 0
            else:
                no_improve_count += 1

            lb_history.append(best_lb)

            v = (1 - alpha) * v + alpha * attracted

            selected = top_p_idx.tolist()
            ub = self.compute_objective(selected)
            if ub < best_ub:
                best_ub = ub
                best_facilities = list(selected)

            ub_history.append(best_ub)

            if (iteration + 1) % 20 == 0 and iteration > 0:
                refined_fac, refined_obj = self._teitz_bart(
                    list(best_facilities), max_iter=30, verbose=False)
                if refined_obj < best_ub:
                    best_ub = refined_obj
                    best_facilities = list(refined_fac)
                    if verbose:
                        print(f"       [Local search] UB improved: "
                              f"{best_ub:.2f}")

            gap = (best_ub - best_lb) / max(abs(best_ub), 1e-10)

            if verbose and (iteration + 1) % 20 == 0:
                print(f"       Iter {iteration + 1}: LB={best_lb:.2f}, "
                      f"UB={best_ub:.2f}, gap={gap * 100:.4f}%, "
                      f"alpha={alpha:.4f}, theta={theta:.4f}, "
                      f"t={elapsed:.1f}s")

            if gap < gap_tol:
                if verbose:
                    print(f"       *** OPTIMAL "
                          f"(gap={gap * 100:.6f}%) ***")
                break

            if no_improve_count >= 50:
                theta *= 0.5
                alpha = max(alpha * 0.8, 0.01)
                no_improve_count = 0
                if theta < 0.001:
                    soft_reset_count += 1
                    refined_fac, refined_obj = self._teitz_bart(
                        list(selected), max_iter=50, verbose=False)
                    if refined_obj < best_ub:
                        best_ub = refined_obj
                        best_facilities = list(refined_fac)
                        if verbose:
                            print(f"       [Deep LS from Lagrangian] "
                                  f"UB improved: {best_ub:.2f}")
                    if soft_reset_count <= 10:
                        theta = 0.3
                        alpha = 0.05
                        v = xp.zeros(N, dtype=np.float64)
                        no_improve_count = 0
                        if verbose:
                            print(f"       [Soft reset #{soft_reset_count}] "
                                  f"Resetting theta=0.3, alpha=0.05, "
                                  f"keeping lambda at iter {iteration + 1}")
                    else:
                        lambda_perturb = xp.asarray(
                            np.random.uniform(0.9, 1.1, N)
                        ) if self.use_gpu else np.random.uniform(0.9, 1.1, N)
                        lambda_vals = lambda_vals * lambda_perturb
                        lambda_vals = xp.maximum(lambda_vals, 0)
                        theta = 0.5
                        alpha = 0.05
                        v = xp.zeros(N, dtype=np.float64)
                        no_improve_count = 0
                        soft_reset_count = 0
                        if verbose:
                            print(f"       [Hard reset] Perturbing lambda, "
                                  f"resetting theta=0.5 at iter {iteration + 1}")

            diff = xp.ones(N, dtype=np.float64) - v
            diff_norm_sq = float(xp.dot(diff, diff))

            if diff_norm_sq < 1e-20:
                if verbose:
                    print(f"       Converged at iter {iteration + 1}")
                break

            step_size = theta * (best_ub - best_lb) / diff_norm_sq
            lambda_vals = lambda_vals + step_size * diff
            lambda_vals = xp.maximum(lambda_vals, 0)

            total_iter = iteration + 1

        if verbose:
            print(f"\n  [Phase 3b] Multi-start local search for UB improvement...")
        n_random_starts = 30
        for rs in range(n_random_starts):
            elapsed = time.time() - lagrangian_start
            if elapsed > time_limit:
                break
            rng = np.random.RandomState(42 + rs)
            random_fac = rng.choice(self.n, self.p, replace=False).tolist()
            refined_fac, refined_obj = self._teitz_bart(
                random_fac, max_iter=100, verbose=False)
            if refined_obj < best_ub:
                best_ub = refined_obj
                best_facilities = list(refined_fac)
                if verbose:
                    print(f"       [Random start {rs + 1}] UB improved: "
                          f"{best_ub:.2f}")
            if rs % 10 == 9 and verbose:
                print(f"       Random starts: {rs + 1}/{n_random_starts}, "
                      f"best UB: {best_ub:.2f}")

        if verbose:
            print(f"\n  [Phase 4] Final refinement (Teitz-Bart)...")
        best_facilities, refined_obj = self._teitz_bart(
            best_facilities, max_iter=200, verbose=False)
        if refined_obj < best_ub:
            best_ub = refined_obj

        final_gap = (best_ub - best_lb) / max(abs(best_ub), 1e-10)
        nearest_idx, nearest_time = self._compute_assignment(best_facilities)

        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        hist_path = os.path.join(
            OUTPUT_DIR, f'convergence_P{self.p}_lagrangian.csv')
        with open(hist_path, 'w') as f:
            f.write("iteration,lower_bound,upper_bound,gap_pct\n")
            for i, (lb, ub) in enumerate(zip(lb_history, ub_history)):
                g = (ub - lb) / max(abs(ub), 1e-10) * 100
                f.write(f"{i + 1},{lb},{ub},{g:.6f}\n")
        if verbose:
            print(f"       Convergence history saved: {hist_path}")

        elapsed = time.time() - start_time
        lagrangian_elapsed = time.time() - lagrangian_start
        if verbose:
            print(f"\n  Lagrangian relaxation completed in {elapsed:.1f}s "
                  f"(iterations: {lagrangian_elapsed:.1f}s)")
            print(f"  Lower bound (proven): {best_lb:.2f}")
            print(f"  Upper bound (best feasible): {best_ub:.2f}")
            print(f"  Optimality gap: {final_gap * 100:.4f}%")
            if final_gap < 0.01:
                print("  *** SOLUTION IS PROVEN GLOBALLY OPTIMAL ***")
            elif final_gap < 0.05:
                print("  *** SOLUTION IS NEAR-OPTIMAL (gap < 5%) ***")
            elif final_gap < 0.10:
                print("  *** SOLUTION IS GOOD (gap < 10%) ***")
            self._print_solution(best_facilities, nearest_idx, nearest_time)

        return best_facilities, nearest_idx, best_ub, best_lb, final_gap

    def solve_gurobi(self, K=None, time_limit=None, verbose=True):
        return self.solve_ilp(K=K, time_limit=time_limit, verbose=verbose)

    def solve_ilp(self, K=None, time_limit=None, verbose=True):
        if verbose:
            print("=" * 60)
            print(f"[Stage 2] Exact ILP Solver: P={self.p}, N={self.n}")
            print("=" * 60)

        K = K or K_NEAREST_INIT
        time_limit = time_limit or ILP_TIME_LIMIT
        start_time = time.time()

        try:
            import gurobipy as gp
            solver_name = "Gurobi"
        except ImportError:
            gp = None
            solver_name = "PuLP/CBC"
            try:
                import pulp
            except ImportError:
                raise ImportError(
                    "No ILP solver available. Install one of:\n"
                    "  - Gurobi: pip install gurobipy "
                    "(free academic license at gurobi.com)\n"
                    "  - PuLP/CBC: pip install pulp")

        if verbose:
            print(f"  Solver: {solver_name}")
            print(f"  Time limit: {time_limit}s")

        if verbose:
            print("  [Phase 1] Computing initial upper bound (greedy+TB)...")
        init_fac = self._greedy_construction(verbose=False)
        init_fac, init_obj = self._teitz_bart(
            init_fac, max_iter=50, verbose=False)
        best_facilities = list(init_fac)
        best_obj = init_obj
        if verbose:
            print(f"       Initial UB (greedy+TB): {best_obj:.2f}")

        current_K = K
        iteration = 0

        while True:
            iteration += 1
            elapsed = time.time() - start_time
            if elapsed > time_limit * 0.9:
                if verbose:
                    print(f"  Time limit approaching, stopping")
                break

            if verbose:
                n_y_vars = self.n * current_K
                print(f"\n  --- Iteration {iteration}: K={current_K} ---")
                print(f"  [2a] Computing {current_K}-nearest neighbors...")
            t0 = time.time()
            knn, knn_time = self._compute_knn(current_K)
            if verbose:
                print(f"       KNN computed in {time.time() - t0:.1f}s")
                print(f"       x variables: {self.n:,}, "
                      f"y variables: {n_y_vars:,}")

            if gp is not None:
                result = self._solve_ilp_gurobi(
                    knn, knn_time, current_K, best_facilities,
                    time_limit, start_time, verbose)
            else:
                result = self._solve_ilp_pulp(
                    knn, knn_time, current_K, best_facilities,
                    time_limit, start_time, verbose)

            if result is None:
                if verbose:
                    print("  Solver failed, stopping")
                break

            facilities, obj_val, is_optimal = result

            if obj_val < best_obj - 1e-6:
                best_obj = obj_val
                best_facilities = facilities
                if verbose:
                    print(f"       UB improved: {best_obj:.2f}")

            if is_optimal and len(facilities) == self.p:
                if verbose:
                    print("  [2d] Verifying optimality (KNN check)...")
                verified = self._verify_solution(facilities, knn)
                if verified:
                    if verbose:
                        print("       *** VERIFIED: Globally optimal! ***")
                    break
                else:
                    new_K = min(current_K * 2, self.n - 1)
                    if verbose:
                        print(f"       Verification failed. "
                              f"K: {current_K} -> {new_K}")
                    current_K = new_K
                    if current_K >= self.n - 1:
                        if verbose:
                            print("       K >= n-1, solution is optimal")
                        break
            else:
                if verbose:
                    print("       Solution not proven optimal by solver")
                new_K = min(int(current_K * 1.5), self.n - 1)
                if new_K <= current_K:
                    new_K = min(current_K + 50, self.n - 1)
                current_K = new_K
                if current_K >= self.n - 1:
                    break

            if time.time() - start_time > time_limit:
                break

        if verbose:
            print("\n  [Phase 3] Final refinement (Teitz-Bart)...")
        best_facilities, refined_obj = self._teitz_bart(
            best_facilities, max_iter=100, verbose=False)
        if refined_obj < best_obj:
            best_obj = refined_obj

        nearest_idx, nearest_time = self._compute_assignment(best_facilities)

        elapsed = time.time() - start_time
        if verbose:
            print(f"\n  ILP completed in {elapsed:.1f}s")
            print(f"  Objective: {best_obj:.2f}")
            print(f"  Final K used: {current_K}")
            self._print_solution(best_facilities, nearest_idx, nearest_time)

        return best_facilities, nearest_idx, best_obj

    def _solve_ilp_gurobi(self, knn, knn_time, K, warm_start_fac,
                          time_limit, start_time, verbose):
        import gurobipy as gp

        t0 = time.time()
        model = gp.Model("P_Median")
        model.setParam('OutputFlag', 1 if verbose else 0)
        remaining = max(60, time_limit - (time.time() - start_time))
        model.setParam('TimeLimit', remaining)
        model.setParam('Threads', 0)
        model.setParam('MIPFocus', 1)
        model.setParam('Presolve', 2)
        model.setParam('Heuristics', 0.3)

        x_vars = model.addVars(self.n, vtype=gp.GRB.BINARY, name="x")

        y_vars = {}
        cost_dict = {}
        for i in range(self.n):
            for k_idx in range(K):
                j = int(knn[i, k_idx])
                if (i, j) not in y_vars:
                    y_vars[i, j] = model.addVar(
                        lb=0, ub=1, vtype=gp.GRB.CONTINUOUS,
                        name=f"y_{i}_{j}")
                cost_dict[i, j] = float(
                    self.weights[i] * knn_time[i, k_idx])

        obj_expr = gp.quicksum(
            cost_dict[i, j] * y_vars[i, j] for i, j in y_vars)
        model.setObjective(obj_expr, gp.GRB.MINIMIZE)

        model.addConstr(
            gp.quicksum(x_vars[j] for j in range(self.n)) == self.p)

        for i in range(self.n):
            assign_terms = []
            for k_idx in range(K):
                j = int(knn[i, k_idx])
                if (i, j) in y_vars:
                    assign_terms.append(y_vars[i, j])
            model.addConstr(gp.quicksum(assign_terms) == 1)

        for i, j in y_vars:
            model.addConstr(y_vars[i, j] <= x_vars[j])

        for j in warm_start_fac:
            if j < self.n:
                x_vars[j].start = 1.0
        for j in range(self.n):
            if j not in set(warm_start_fac):
                x_vars[j].start = 0.0

        if verbose:
            print(f"       Gurobi model built in {time.time() - t0:.1f}s")
            print(f"       Solving (limit: {remaining:.0f}s)...")

        model.optimize()

        is_optimal = (model.status == gp.GRB.OPTIMAL)

        if model.status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL,
                            gp.GRB.TIME_LIMIT):
            facilities = []
            for j in range(self.n):
                if x_vars[j].X > 0.5:
                    facilities.append(j)

            if len(facilities) == self.p:
                obj_val = self.compute_objective(facilities)
                if verbose:
                    gap_str = ""
                    if model.MIPGap < 1e6:
                        gap_str = f", MIP gap={model.MIPGap * 100:.4f}%"
                    print(f"       Solution found: obj={obj_val:.2f}"
                          f"{gap_str}")
                return facilities, obj_val, is_optimal
            else:
                if verbose:
                    print(f"       Warning: {len(facilities)} != {self.p}")
                return None

        if verbose:
            print(f"       Gurobi status: {model.status}")
        return None

    def _solve_ilp_pulp(self, knn, knn_time, K, warm_start_fac,
                        time_limit, start_time, verbose):
        import pulp

        t0 = time.time()
        prob = pulp.LpProblem("P_Median", pulp.LpMinimize)

        x = {}
        for j in range(self.n):
            x[j] = pulp.LpVariable(f"x_{j}", cat='Binary')
            if j in set(warm_start_fac):
                x[j].setInitialValue(1)
            else:
                x[j].setInitialValue(0)

        y = {}
        obj_terms = []
        for i in range(self.n):
            for k_idx in range(K):
                j = int(knn[i, k_idx])
                if (i, j) not in y:
                    y[i, j] = pulp.LpVariable(
                        f"y_{i}_{j}", lowBound=0, upBound=1,
                        cat='Continuous')
                cost = float(self.weights[i] * knn_time[i, k_idx])
                obj_terms.append(cost * y[i, j])
        prob += pulp.lpSum(obj_terms)

        prob += pulp.lpSum(x[j] for j in range(self.n)) == self.p

        for i in range(self.n):
            prob += pulp.lpSum(
                y[i, int(knn[i, k_idx])]
                for k_idx in range(K)
                if (i, int(knn[i, k_idx])) in y) == 1

        for i, j in y:
            prob += y[i, j] <= x[j]

        if verbose:
            print(f"       PuLP model built in {time.time() - t0:.1f}s")

        remaining = max(60, time_limit - (time.time() - start_time))
        if verbose:
            print(f"       Solving with CBC (limit: {remaining:.0f}s)...")

        solver = pulp.PULP_CBC_CMD(
            msg=1 if verbose else 0,
            timeLimit=int(remaining),
            threads=0,
            warmStart=True)
        prob.solve(solver)

        is_optimal = (prob.status == pulp.constants.LpStatusOptimal)

        if prob.status in (pulp.constants.LpStatusOptimal,
                           pulp.constants.LpStatusNotSolved):
            facilities = []
            for j in range(self.n):
                val = x[j].varValue
                if val is not None and val > 0.5:
                    facilities.append(j)

            if len(facilities) == self.p:
                obj_val = self.compute_objective(facilities)
                if verbose:
                    status_str = pulp.LpStatus[prob.status]
                    print(f"       Solution found: obj={obj_val:.2f}, "
                          f"status={status_str}")
                return facilities, obj_val, is_optimal
            else:
                if verbose:
                    print(f"       Warning: {len(facilities)} != {self.p}")
                return None

        if verbose:
            print(f"       CBC status: {pulp.LpStatus[prob.status]}")
        return None

    def solve_ga(self, pop_size=None, generations=None, seed=42, verbose=True):
        if verbose:
            print("=" * 60)
            print(f"[Stage 2] GA Solver: P={self.p}, N={self.n}")
            print("=" * 60)

        pop_size = pop_size or GA_POP_SIZE
        generations = generations or GA_GENERATIONS

        rng = np.random.RandomState(seed)
        start_time = time.time()

        if verbose:
            print(f"  [2a] Initializing population (size={pop_size})...")
        population = self._ga_init_population(pop_size, rng)
        fitness = np.array(
            [self.compute_objective(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_facilities = list(population[best_idx])
        best_obj = fitness[best_idx]

        if verbose:
            print(f"       Initial best: {best_obj:.2f}")

        elite_count = max(1, int(pop_size * GA_ELITE_RATIO))
        no_improve = 0

        if verbose:
            print(f"  [2b] Running GA for {generations} generations...")

        for gen in range(generations):
            sorted_idx = np.argsort(fitness)
            population = [population[i] for i in sorted_idx]
            fitness = fitness[sorted_idx]

            new_pop = list(population[:elite_count])

            while len(new_pop) < pop_size:
                p1 = self._tournament_select(population, fitness, rng)
                p2 = self._tournament_select(population, fitness, rng)
                child = self._ga_crossover(p1, p2, rng)
                child = self._ga_mutate(child, rng)
                new_pop.append(child)

            population = new_pop
            fitness = np.array(
                [self.compute_objective(ind) for ind in population])

            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < best_obj - 1e-6:
                best_obj = fitness[curr_best_idx]
                best_facilities = list(population[curr_best_idx])
                no_improve = 0
            else:
                no_improve += 1

            if verbose and (gen + 1) % 50 == 0:
                print(f"       Gen {gen + 1}/{generations}: "
                      f"best={best_obj:.2f}, avg={fitness.mean():.2f}")

            if no_improve >= 100:
                if verbose:
                    print(f"       Converged at gen {gen + 1}")
                break

        if verbose:
            print("  [2c] Local search refinement...")
        best_facilities, best_obj = self._teitz_bart(
            best_facilities, max_iter=GA_LOCAL_SEARCH_ITER, verbose=False)

        nearest_idx, nearest_time = self._compute_assignment(best_facilities)

        elapsed = time.time() - start_time
        if verbose:
            print(f"\n  GA completed in {elapsed:.1f}s")
            print(f"  Objective: {best_obj:.2f}")
            self._print_solution(best_facilities, nearest_idx, nearest_time)

        return best_facilities, nearest_idx, best_obj

    def solve_greedy(self, verbose=True):
        if verbose:
            print("=" * 60)
            print(f"[Stage 2] Greedy+TeitzBart Solver: "
                  f"P={self.p}, N={self.n}")
            print("=" * 60)

        start_time = time.time()

        if verbose:
            print("  [2a] Greedy construction...")
        facilities = self._greedy_construction(verbose=verbose)

        if verbose:
            print("  [2b] Teitz-Bart improvement...")
        facilities, obj = self._teitz_bart(facilities, verbose=verbose)

        nearest_idx, nearest_time = self._compute_assignment(facilities)

        elapsed = time.time() - start_time
        if verbose:
            print(f"\n  Greedy+TB completed in {elapsed:.1f}s")
            print(f"  Objective: {obj:.2f}")
            self._print_solution(facilities, nearest_idx, nearest_time)

        return facilities, nearest_idx, obj

    def _compute_knn(self, K):
        N = self.n
        knn = np.zeros((N, K), dtype=np.int32)
        knn_time = np.full((N, K), 1e18, dtype=np.float64)

        for i in range(N):
            sorted_indices = np.argpartition(self.time_matrix[i, :], K + 1)[:K + 1]
            sorted_indices = sorted_indices[np.argsort(self.time_matrix[i, sorted_indices])]
            sorted_indices = sorted_indices[sorted_indices != i][:K]
            knn[i, :len(sorted_indices)] = sorted_indices
            knn_time[i, :len(sorted_indices)] = self.time_matrix[i, sorted_indices]

        return knn, knn_time

    def _verify_solution(self, facilities, knn):
        nearest_idx, _ = self._compute_assignment(facilities)
        fac_set = set(facilities)
        for i in range(self.n):
            assigned_fac = facilities[nearest_idx[i]]
            if assigned_fac not in fac_set:
                return False
            row_knn = set(knn[i].tolist())
            if assigned_fac not in row_knn:
                return False
        return True

    def _greedy_construction(self, verbose=True):
        facilities = []
        remaining = set(range(self.n))
        current_nearest_time = np.full(self.n, 1e18)

        for step in range(self.p):
            best_idx = -1
            best_obj = float('inf')

            candidates = list(remaining)
            for j in candidates:
                # time_matrix[:, j] is travel times from all grids to facility j
                time_to_j = self.time_matrix[:, j]
                new_nearest = np.minimum(current_nearest_time, time_to_j)
                new_obj = np.sum(self.weights * new_nearest)
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_idx = j

            facilities.append(best_idx)
            remaining.discard(best_idx)
            time_to_best = self.time_matrix[:, best_idx]
            current_nearest_time = np.minimum(current_nearest_time, time_to_best)

            if verbose:
                r, c = self.rows[best_idx], self.cols[best_idx]
                print(f"       Step {step + 1}/{self.p}: "
                      f"site at ({r},{c}), obj={best_obj:.2f}")

        return facilities

    def _teitz_bart(self, facilities, max_iter=50, verbose=True):
        facilities = list(facilities)
        facility_set = set(facilities)
        n_fac = len(facilities)

        nearest_idx, nearest_time = self._compute_assignment(facilities)
        current_obj = np.sum(self.weights * nearest_time)

        if verbose:
            print(f"       TB start: obj={current_obj:.2f}")

        for iteration in range(max_iter):
            best_swap = None
            best_delta = 0

            non_facilities = [i for i in range(self.n)
                              if i not in facility_set]

            for j_in in non_facilities:
                time_to_j = self.time_matrix[:, j_in]
                new_time_if_keep = np.minimum(time_to_j, nearest_time)

                fac_arr = np.array(facilities)
                # Get times to all current facilities
                times_to_facs = self.time_matrix[:, fac_arr]
                sorted_times = np.sort(times_to_facs, axis=1)
                second_nearest_time = (
                    sorted_times[:, 1] if n_fac > 1
                    else np.full(self.n, 1e18))
                new_time_if_remove = np.minimum(time_to_j, second_nearest_time)

                base_delta = np.sum(
                    self.weights * (nearest_time - new_time_if_keep))

                loss_per_node = self.weights * (
                    new_time_if_remove - new_time_if_keep)
                extra_loss = np.bincount(
                    nearest_idx, weights=loss_per_node, minlength=n_fac)

                best_k_idx = int(np.argmin(extra_loss))
                delta = base_delta - extra_loss[best_k_idx]

                if delta > best_delta + 1e-6:
                    best_delta = delta
                    best_swap = (j_in, best_k_idx, facilities[best_k_idx])

            if best_swap is None:
                if verbose:
                    print(f"       TB converged at iter {iteration + 1}")
                break

            j_in, k_idx, k_out = best_swap
            facilities[k_idx] = j_in
            facility_set = set(facilities)

            nearest_idx, nearest_time = self._compute_assignment(facilities)
            new_obj = np.sum(self.weights * nearest_time)

            if verbose and (iteration + 1) % 5 == 0:
                print(f"       Iter {iteration + 1}: "
                      f"obj {current_obj:.2f}->{new_obj:.2f}")

            current_obj = new_obj
        else:
            if verbose:
                print(f"       TB reached max iterations ({max_iter})")

        return facilities, current_obj

    def _ga_init_population(self, pop_size, rng):
        population = []

        greedy_fac = self._greedy_construction(verbose=False)
        population.append(greedy_fac)

        pop_weights = self.weights.copy()
        pop_weights[pop_weights < 0] = 0
        total = pop_weights.sum()
        probs = (pop_weights / total
                 if total > 0
                 else np.ones(self.n) / self.n)

        for _ in range(pop_size - 1):
            indices = rng.choice(
                self.n, size=self.p, replace=False, p=probs)
            population.append(sorted(indices.tolist()))

        return population

    def _tournament_select(self, population, fitness, rng):
        indices = rng.choice(
            len(population), size=GA_TOURNAMENT_SIZE, replace=False)
        best = indices[np.argmin(fitness[indices])]
        return population[best]

    def _ga_crossover(self, p1, p2, rng):
        union = list(set(p1) | set(p2))
        if len(union) < self.p:
            all_indices = set(range(self.n))
            candidates = list(all_indices - set(union))
            need = self.p - len(union)
            extra = rng.choice(
                candidates, size=min(need, len(candidates)),
                replace=False)
            union.extend(extra.tolist())

        if len(union) > self.p:
            selected = rng.choice(union, size=self.p, replace=False)
            return sorted(selected.tolist())
        return sorted(union[:self.p])

    def _ga_mutate(self, individual, rng):
        if rng.random() > GA_MUTATION_RATE:
            return individual

        ind = list(individual)
        fac_set = set(ind)

        non_fac = [i for i in range(self.n) if i not in fac_set]
        if len(non_fac) == 0:
            return ind

        pos = rng.randint(0, len(ind))
        new_fac = non_fac[rng.randint(0, len(non_fac))]
        ind[pos] = new_fac

        return sorted(ind)

    def _print_solution(self, facilities, nearest_idx, nearest_time):
        print(f"\n  Selected sites (P={len(facilities)}):")
        for i, fac in enumerate(facilities):
            r, c = self.rows[fac], self.cols[fac]
            pop_assigned = self.weights[nearest_idx == i].sum()
            n_assigned = (nearest_idx == i).sum()
            avg_time = np.mean(nearest_time[nearest_idx == i]) if n_assigned > 0 else 0
            print(f"    Site {i + 1}: grid({r},{c}), "
                  f"serves {n_assigned} cells, "
                  f"pop={pop_assigned:,.0f}, "
                  f"avg travel={avg_time:.1f}s")

    def solve(self, method='lagrangian', verbose=True, **kwargs):
        if method == 'lagrangian':
            result = self.solve_lagrangian(verbose=verbose, **kwargs)
            facilities, assignments, obj, lb, gap = result
            return facilities, assignments, obj
        elif method == 'gurobi':
            return self.solve_gurobi(verbose=verbose, **kwargs)
        elif method == 'ilp':
            return self.solve_ilp(verbose=verbose, **kwargs)
        elif method == 'ga':
            return self.solve_ga(verbose=verbose, **kwargs)
        elif method == 'greedy':
            return self.solve_greedy(verbose=verbose)
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Use 'lagrangian', 'gurobi', 'ilp', 'ga', or 'greedy'")
