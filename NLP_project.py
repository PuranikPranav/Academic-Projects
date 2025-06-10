import numpy as np
import time

#Get the objective function value at a specific point
def objective_function(x):
    return (x[0] - x[1])**2 + (x[1] - x[2])**2 + (x[2] - x[3])**4 + (x[3] - x[4])**2

def error_function(x,y):
    der_lag = dLag(x,y)
    first_term = np.linalg.norm(der_lag)
    H_of_x = constraints(x)
    second_term = np.linalg.norm(H_of_x)
    return max(first_term, second_term)
    
#Get the gradient of the function at a specific point
def gradient_of_objective_function(x):
    return np.array([2*(x[0]-x[1]), 2*(x[1]-x[2]) - 2*(x[0] - x[1]), 4*(x[2]-x[3])**3 - 2*(x[1] - x[2]), 2*(x[3]-x[4]) - 4*(x[2]-x[3])**3,-2*(x[3]-x[4])]).reshape(-1,1)

#Get the Jacobian
def Jacobian():
    return np.array([[1,0,0],[2,1,0],[3,2,1],[0,3,2],[0,0,3]])

#Get the LHS of the constraints at a specific point
def constraints(x):
    return np.array([x[0] + 2*x[1] + 3*x[2] - 6, x[1] + 2*x[2] + 3*x[3] - 6, x[2] + 2*x[3] + 3*x[4] - 6]).reshape(-1,1)

#Get the vector valued function to check for the norm
def F_of_y(x,mu):
    A = Jacobian()
    q = gradient_of_objective_function(x) - A @ mu
    H = np.array([0,0,0]).reshape(-1,1)
    return np.concatenate((q,H))

#Get the gradient of Lagrangian
def dLag(x,mu):
    A = Jacobian()
    return gradient_of_objective_function(x) - A @ mu

#hessian of lagrangian
def hessian(x):

    H = np.zeros((5, 5))

    H[0, 0] = 2
    H[0, 1] = -2
    H[1, 0] = -2
    H[1, 1] = 4
    H[1, 2] = -2
    H[2, 1] = -2
    H[2, 2] = 12 * (x[2] - x[3])**2 + 2
    H[2, 3] = -12 * (x[2] - x[3])**2
    H[3, 2] = -12 * (x[2] - x[3])**2
    H[3, 3] = 2 + 12 * (x[2] - x[3])**2
    H[3, 4] = -2
    H[4, 3] = -2
    H[4, 4] = 2

    return H

def JF_of_k(x):
    H = hessian(x)
    J = Jacobian()
    m = J.shape
    # print(m)
    return np.vstack((np.hstack((H,J)), np.hstack((J.T,np.zeros((m[1],m[1]))))))


#Get the merit function value to check for step length
def merit_function(x, pen_param=100):
    return objective_function(x) + pen_param*np.linalg.norm(constraints(x),1)

def IPM(x, y, k=0):
    E_of_k = error_function(x,y)

    while E_of_k > 1e-8:
        f_of_x = objective_function(x)
        jacobian = JF_of_k(x)
        der_lag = dLag(x,y)
        H_of_x = constraints(x)
        RHS = np.vstack((der_lag, H_of_x ))
        d = np.linalg.solve(jacobian, RHS)
        n = x.size
        dx_k = d[:n].reshape(-1,1)
        dy_k = d[n:].reshape(-1,1)
        grad_f = gradient_of_objective_function(x)

        alpha = 1
        sigma = 1e-4
        beta = 0.5
        #Get the step length using the merit function
        while merit_function(x + alpha * dx_k) > merit_function(x) + sigma * alpha * grad_f.T @ dx_k:
            alpha *= beta
        
        print(f"\nIteration {k}\n")
        print(f"x = {x}\n")
        print(f"f(x) = {f_of_x}\n")
        print("\n--------------------------------------------------------------------------------\n")
        k=k+1
        x = x + alpha*dx_k
        y = y + dy_k
        E_of_k = error_function(x,y)
    print("\n--------------------------------------------------------------------------------\n")
    print(f"\nThe program terminates and converges successfully with total of {k} iterations\n")
    #Note the end time
    # end = time.time()
    # total_time = end - start
    # print(f"The CPU time is {total_time} seconds\n")
    return x, y

def constraint_violation(x):
    return np.linalg.norm(constraints(x), 1)


def SQP(x_0, mu_0, k=0, use_filter = False):
    """This method takes a starting point and starting lagrange multipliers
    and solves a QP subproblem in each iteration to get the final optimal
    solution
    """
    #Note the starting time
    start = time.time()
    print("\nSQP algorithm")
    x_k = x_0
    mu_k = mu_0
    m = mu_0.size
    n = x_0.size
    B_k = np.eye(n,n)
    filter_set = []
    gamma = 1e-4
    
    #Iteratively solve the EQP to get the search direction
    while np.linalg.norm(F_of_y(x_k,mu_k)) > 1e-10:
        f_of_x = objective_function(x_k)[0]
        A = Jacobian()  
        KKT_matrix = np.vstack((np.hstack((B_k, -A)), np.hstack((A.T, np.zeros((m, m))))))
        grad_f = gradient_of_objective_function(x_k)
        c_k = constraints(x_k)
        RHS = np.concatenate((-grad_f, -c_k))
        solution = np.linalg.solve(KKT_matrix, RHS)
        dx_k = solution[:n].reshape(-1,1)
        dmu_k = solution[n:].reshape(-1,1)

        alpha = 1
        sigma = 1e-4
        beta = 0.5
        x_trial = x_k + alpha * dx_k
        f_trial = objective_function(x_trial)[0]
        c_trial = constraint_violation(x_trial)
        if use_filter:
            # Filter acceptance condition
            while any(
                f_trial >= f_i - gamma and c_trial >= c_i - gamma
                for (f_i, c_i) in filter_set
            ):
                alpha *= beta
                x_trial = x_k + alpha * dx_k
                f_trial = objective_function(x_trial)
                c_trial = constraint_violation(x_trial)

            # Add to filter
            filter_set.append((f_trial, c_trial))
        else:
            # Merit function line search
            while merit_function(x_k + alpha * dx_k) > merit_function(x_k) + sigma * alpha * grad_f.T @ dx_k:
                alpha *= beta


        #BFGS update
        s_k = alpha*dx_k
        x_kplus1 = x_k + s_k
        mu_plus1 = dmu_k
        y_k = dLag(x_kplus1,mu_plus1) - dLag(x_k, mu_plus1)
        if s_k.T @ y_k >= 0.2*s_k.T @ B_k @ s_k:
            theta_k = 1
        else:
            theta_k = (0.8*s_k.T @ B_k @ s_k)/(s_k.T @ B_k @ s_k - s_k.T @ y_k)
        r_k = theta_k*y_k + (1-theta_k)*B_k @ s_k
        B_k = B_k - ((B_k@s_k@s_k.T@B_k)/(s_k.T@B_k@s_k)) + ((r_k @ r_k.T)/(s_k.T@r_k))
        
        print(f"\nIteration {k}\n")
        print(f"x = {x_k}\n")
        print(f"f(x) = {f_of_x}\n")

        print("\n--------------------------------------------------------------------------------\n")

        #Next iteration values
        k=k+1
        x_k = x_kplus1
        mu_k = mu_plus1
        # Stop if converged
        if np.linalg.norm(F_of_y(x_k, mu_k)) <= 1e-8:
            break
    print("\n--------------------------------------------------------------------------------\n")
    print(f"\nThe program terminates and converges successfully with total of {k} iterations\n")

    #Note the end time
    end = time.time()
    total_time = end - start
    print(f"The CPU time is {total_time} seconds\n")
    return x_k, mu_k

#Define the starting point
x_0 = np.array([35,-31,11,5,-5]).reshape(-1,1)
#Define the lagrange multipliers
mu_0 = np.array([1,2,3]).reshape(-1,1)
#Run the SQP solver
x,mu = SQP(x_0,mu_0)

print(f"The optimal solution is {x}\n")
f = objective_function(x)[0]
if f < 1e-18:
    f=0
print(f"The optimal objective function value {f}")


