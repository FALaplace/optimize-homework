```python
def problem_3(Sr, Cr, Ur, s0, st):
    U = cp.Variable(Ur.shape)
    S = dict.fromkeys(product(range(Ns), range(Tk + 1)))
    for j in range(Ns):
        S[(j, 0)] = s0[j]
    constraints = []
    objective = 0.0
    for j in range(Ns):
        for k in range(1, Tk + 1):
            Jac = mm.Jacboi_matrix(Sr[j, 6 * (k - 1): 6 * k], Cr[6 * (k - 1):6 * k])
            err = mm.ds_fsat(Sr[j, 6 * (k - 1): 6 * k], Cr[6 * (k - 1):6 * k]) - 
            	  Jac @ Sr[j, 6 * (k - 1):6 * k]
            Ad = expm(t_sam * Jac)
            Bd = t_sam * (expm(t_sam * Jac) @ mm.B + 4 * expm(t_sam * Jac / 2) @ mm.B + mm.B) / 6
            erd = t_sam * (expm(t_sam * Jac) @ err + 4 * expm(t_sam * Jac / 2) @ err + err) / 6
            S[(j, k)] = Ad @ S[(j, k - 1)] + Bd @ U[j, 3 * (k - 1): 3 * k] + erd
            objective += cp.norm(U[j, 3 * (k - 1): 3 * k]) * t_sam
            objective += cp.maximum(0.0, cp.norm(S[(j, k)][:3]) - scale) * r_pun
            constraints.append(cp.norm(S[(j, k)] - Sr[j, 6 * k: 6 * (k + 1)], p="inf") <= tau)
            constraints.append(cp.norm(U[j, 3 * (k - 1): 3 * k]) <= umax)
            for m in range(j):
                Ef = Sr[j, 6 * k: 6 * k + 3] - Sr[m, 6 * k: 6 * k + 3]
                if norm(Ef) < dmin + 2 * tau:
             		objective += cp.maximum(dmin*norm(Ef) - Ef@(S[(j, k)][:3]-S[(m, k)][:3]), 0.0)*c_pun
                constraints.append(cp.norm(S[(j, k)][:3] - S[(m, k)][:3]) <= dmax)
            for o in range(r_obs.size):
			   po, ro = pos_obs[o], r_obs[o]
                Df = Sr[j, 6 * k: 6 * k + 3] - po
                if norm(Df) < ro + tau:
                     objective += cp.maximum(ro * norm(Df) - Df @ (S[(j, k)][:3] - po), 0.0) * c_pun
        objective += cp.norm(S[(j, Tk)] - st) * t_pun
    opt = cp.Minimize(objective)
    prob = cp.Problem(opt, constraints)
    prob.solve(solver=cp.MOSEK)
    return U.value, prob.value
```

```python
    while True:
        U_, obj_ = problem_3(S_ref, C_ref, U_ref, s_ini, s_tar)
        counter += 1
        S_, tau_real = cal_reference_state(U_, S_ref, C_ref, s_ini)
        with open('data/iteration_log.txt', 'a') as file:
            file.write(f'The {counter} iteration, the cost is {obj_}\n')
        U_ref = U_.copy()
        S_ref = S_.copy()
        tau = tau_real * 0.99
        if tau_real <= 1.0 or abs(obj_ - obj) <= 0.01:
            break
        obj = obj_
```

```python
def problem_4(Sr, Cr, s0, st, j, Ns, Tk):
    Uj = cp.Variable(Tk * 3)
    Sj = dict.fromkeys(range(Tk + 1))
    Sj[0] = s0[j]
    constraints = []
    objective = 0.0
    for k in range(1, Tk + 1):
        Sj[k] = Ad @ Sj[k - 1] + Bd @ Uj[3 * (k - 1): 3 * k] + erd
        objective += cp.norm(Uj[3*(k-1):3*k])*t_sam + cp.maximum(0.0, cp.norm(Sj[k][:3])-scale)*r_pun
        constraints.append(cp.norm(Sj[k] - Sr[j, 6 * k: 6 * (k + 1)], p="inf") <= tau) 
        constraints.append(cp.norm(Uj[3 * (k - 1): 3 * k]) <= umax)
        for m in range(Ns):
            if m != j:
                Ef = Sr[j, 6 * k: 6 * k + 3] - Sr[m, 6 * k: 6 * k + 3]
                objective += cp.maximum(dmin*norm(Ef)-Ef@(Sj[k][:3]-Sr[m, 6*k:6*k+3]), 0.0)*c_pun/2
                constraints.append(cp.norm(Sj[k][:3] - Sr[m, 6 * k: 6 * k + 3]) <= (dmax - 0))
        for o in range(r_obs.size):
            po, ro = pos_obs[o], r_obs[o]
            Df = Sr[j, 6 * k: 6 * k + 3] - po
            if norm(Df) < ro + tau:
                objective += cp.maximum(ro * norm(Df) - Df @ (Sj[k][:3] - po), 0.0) * c_pun
    objective += cp.norm(Sj[Tk] - st) * t_pun
    opt = cp.Minimize(objective)
    prob = cp.Problem(opt, constraints)
    prob.solve(solver=cp.MOSEK)
    return Uj.value, prob.value

def thread_fun(Sr, Cr, s0, st, j, Ns, Tk, res):
    result = problem_4(Sr, Cr, s0, st, j, Ns, Tk)
    res[j] = result
    return 0

while True:
    outputs = [None] * N
    threads = []
    for j in range(N):
        thread = threading.Thread(target=thread_fun, args=(S_ref,C_ref,s_ini,s_tar,j,N,K,outputs))
        threads.append(thread)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    outputs_T = list(zip(*outputs))
    obj_, U_ = sum(outputs_T[1]), np.array(outputs_T[0])
    S_, tau_real = cal_reference_state(U_, S_ref, C_ref, s_ini)
    U_ref = U_.copy()
    S_ref = S_.copy()
    tau = tau_real * 0.99
    if tau_real <= 1.0 or abs(obj_ - obj) <= 0.01:
        break
    obj = obj_
```

