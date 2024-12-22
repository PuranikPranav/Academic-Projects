import numpy as np #Import the numpy library

#Define constants for the IPM method
alpha = 0.995 #Step feasibility fraction
beta = 0.1 #Step complementarity fraction
tolerance = 1e-6 #complementarity tolerance

#Suppresses the runtimewarning of "divisionbyzero"
np.seterr(divide='ignore')

def row_ops(nbv, bv, r0, T):
    """Performs row operations starting with the given tableau
    param np.ndarray nbv: list of indices of starting non basic variables
    param np.ndarray bv: list of indices of starting basic variables
    :param np.ndarray ro: starting zeroth row of the tableau (reduced costs + obj value)
    :param np.ndarray T: starting main body of tableau as a matrix (constraint coefficient matrix + RHS)
    :return: optimal objective function value, bv, nbv
    :rtype:
    """

    #Initiate iteration counter
    i=1
    while True:

        #Print the tableaus in each iterations
        print("\n-------------------------\n")
        print(f"\nIteration {i}\n")
        print('\n')
        print(f"Row 0 is ---> {r0}")
        print('\n')
        print(f"The body of the tableau is ---> {T}")
        print('\n')
        print(f"The basic variables are ---> {bv}")
        print('\n')
        print(f"The non basic variables are ---> {nbv}")
        #If all values in row 0 are non positive then the current tableau is optimal
        if np.all(r0[:-1] <= 0):
            return r0[-1], bv, nbv, r0, T
        
        #Blands rule for the entering variable
        positive_indices = np.where(r0 > 0)[0] #Get the indices of all the positive reduced costs
        entering_variable = positive_indices[0] #Select the variable with least index out of all the positive indices

        #Checking whether all the values in the corresponding column of the entering variable are non positive
        #If they are all non positive, then the LP is unbounded
        y = T[:,entering_variable]
        if np.all(y <= 0):
            recession_direction = np.zeros(len(r0)-1) #Initiate the recession direction
            recession_direction[bv] = -1*T[:,entering_variable] #Get the recession direction
            recession_direction[entering_variable] = 1 #Value corresponding to entering variable is 1
            print(f"The problem is unbounded with the recession direction {recession_direction}\n")
            return -np.inf, bv, nbv, r0, T #Return with z=-infinity and other variables
        
        #Blands rule for leaving variable and ratio test
        RHS = T[:,-1] #Last column of the body
        positive_indices = np.where(y > 0)[0] #Only get the indices where y is positive
        ratios = RHS[positive_indices]/y[positive_indices] #Calculate the ratios only for those variables
        p_index = np.argmin(ratios) #Get the index with minimum ratio. argmin automatically applies blands rule
        row_index = positive_indices[p_index] #Get the row index of the leaving variable 
        leaving_variable = bv[row_index] #Get the leaving variable
        
        #Perform row operations
        T[row_index, :] = T[row_index, :]/T[row_index, entering_variable] #Make the pivot element = 1
        for row_num in range(T.shape[0]): #Loop through all the rows of the body
            if row_num != row_index: #only do this for rows other than the one corresponding to the leaving variable
                T[row_num,:] = T[row_num,:] + (-1*T[row_num, entering_variable])*T[row_index, :] #Elementary row_op to make 0
        r0 = r0 + (-1*r0[entering_variable])*T[row_index, :] # Elementary row_op for the zeroth row

        #Swap the entering and leaving variable
        bv[row_index] = entering_variable
        nbv[np.where(nbv == entering_variable)] = leaving_variable

        #Increment iteration counter
        i=i+1



def simplex(c, A, b):
    """Solves a LP with simplex method

    :param np.ndarray c: an 1Xn vector of cost coefficients
    :param np.ndarray A: an mXn matrix of constraint coefficients
    :param np.ndarray b: an mX1 vector of RHS of constraints
    :return: z, bv, optimal_solution
    :rtype: 
    """

    #If the RHS is negative, then modify the coefficient matrix A
    negative_indices = np.where(b < 0)[0]
    b[negative_indices] = -1*b[negative_indices]
    A[negative_indices] = -1*A[negative_indices]

    #Add the columns for artificial variables in coefficient matrix A
    A = np.hstack((A, np.eye(A.shape[0])))
    #Change the cost vector for Phase1
    phase1_cost = np.hstack((np.zeros(c.shape[0]), np.ones(A.shape[0])))
    #Set the basic and non basic indices for Phase 1
    bv = np.where(phase1_cost == 1)[0]
    nbv = np.where(phase1_cost == 0)[0]
    # Set Artificial variables
    av = bv.copy()

    #Cost vector for basic variables
    cb = phase1_cost[bv]
    #Cost vector for non basic variables
    cn = phase1_cost[nbv]
    #Set the B_inverse as the identitiy matrix
    B_inv = np.eye(len(bv))
    #sub-matrix for the non basic variables
    N = A[:, nbv]

    #Calculate the zeroth row for the tableau
    r0 = np.zeros(len(phase1_cost) + 1)
    r0[nbv] = (cb @ B_inv @ N) - cn
    r0[-1] = cb @ B_inv @ b

    #Calculate the main body of the initial tableau
    T = np.zeros((len(bv), len(bv) + len(nbv) + 1 ))
    T[:, bv] = np.eye(len(bv))
    T[:, nbv] = B_inv @ N
    T[:, -1] = B_inv @ b

    print("\nThe phase 1 iterations are\n")
    #Perform row operations for Phase1
    z, bv, nbv, r0, T = row_ops(nbv, bv, r0, T)
    print("\n------------------------Phase 1 ends---------------------------------------------------\n")

    #Due to precision issues, if the optimal objective value returned by Phase 1 is more than 1e-5, 
    #it will be considered significant enough to call it non zero, and hence the LP will be infeasible
    if (abs(z) > 1e-5):
        print("\nThe problem is infeasible\n")
        print(f"The optimal objective function value at the end of Phase 1 is {z}")
        return None, None, None
    
    #Calculate the indices of the redundant rows in the tableau
    redundant_rows = [row for row, value in enumerate(bv) if value in av]
    #Change the basic and non basic indices to only keep non-artificial variables
    bv = np.array([value for value in bv if value not in av])
    nbv = np.array([value for value in nbv if value not in av])
    #Delete the redundant rows
    T = np.delete(T, redundant_rows, axis = 0)
    #Calculate the inverse for the Phase 2
    av2 = np.delete(av,redundant_rows)
    B_inv = T[:,av2]
    #Delete the artificial variable columns from the tableau
    T = np.delete(T, av, axis=1)
    #Also modify the coefficient matrix so that it will be easier to calculate N further ahead
    A = np.delete(A,redundant_rows,axis=0)

    #Calculate the components for Phase 2 zeroth row
    cb = c[bv]
    cn = c[nbv]
    N = A[:,nbv]

    #Calculate the zeroth row for Phase 2
    r0 = np.zeros(len(c) + 1)
    r0[nbv] = (cb @ B_inv @ N) - cn
    r0[-1] = cb @ T[:,-1]

    print("\nThe Phase 2 iterations are:\n")
    #Perform row operations for Phase 2
    z, bv, nbv, r0, T = row_ops(nbv, bv, r0, T)
    
    #Optimal solution should consist of values for all the original variables passed to the function
    optimal_solution = np.zeros(len(c))
    optimal_solution[bv] = T[:,-1]
    return bv, z, optimal_solution


def forestry(method):
    """
    FORESTRY PROBLEM
    """
    number_of_constraints = 8
    number_of_variables = 24

    #The cost vector is negated because the original problem is of maximization and we want minimization
    c = np.zeros(number_of_variables)
    c[:16] = [-16,-12,-20,-18,-14,-13,-24,-20,-17,-10,-28,-20,-12,-11,-18,-17]

    #RHS of the inequalities
    b = np.array([1500,1700,900,600,22.5,9,4.8,3.5])

    #Filling up the coefficient matrix
    A = np.zeros((number_of_constraints, number_of_variables))
    A[0, :4], A[1, 4:8], A[2, 8:12], A[3, 12:16] = [1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]
    A[0, 16], A[1, 17], A[2, 18], A[3, 19] = 1, 1, 1, 1
    for index, arr in enumerate([[17,15,13,10], [14,16,12,11], [10,12,14,8], [9,11,8,6]], start=4):
        A[index, (index-4):16:4] = arr
    A[4, 20], A[5, 21], A[6, 22], A[7, 23] = [-1, -1, -1, -1]
    
    
    if method.lower() == 'simplex': #Run this if user typed "simplex"
        #Run the simplex algorithm
        bv,z, optimal_solution = simplex(c=c, A=A, b=b)
        #if the LP is not unbounded or infeasible then go ahead and print the results
        if z != (-np.inf) and z != None:
            print("\n-------------------------\n")
            print("\nThe final results are:\n")
            print(f"\nThe basic variables are {bv}\n")
            print(f"The optimal objective function value : {-1*z}(original problem is of maximization)")
            print(f"\nThe optimal solution is: \n{optimal_solution}")
    
    elif method.lower() == 'ipm': #Run this if user typed "ipm"
        #Run the IPM
        x, sigma, pi = IPM(c,A,b)
        print("\n-------------------------\n")
        print(f"The final answer is :\n")
        print(f"The x is {x}\n\n The sigma is {sigma}\n\n and the pi is {pi}\n")
        z = c @ x #Get the optimal objective function value from the optimal solution "x"
        print(f"The optimal objective function value is {-1*z}(original problem is of maximization)")

    else:
        print(f"Invalid input!")


def IPM(c, A, b):
    """ Solves the LP model by Interior Point Method
    :param np.ndarray c: an 1Xn vector of cost coefficients
    :param np.ndarray A: an mXn matrix of constraint coefficients
    :param np.ndarray b: an mX1 vector of RHS of constraints
    :return: x,pi,sigma
    :rtype: 
    """
    m = A.shape[0] #Get the number of rows/constraints/dual variables
    n = A.shape[1] #Get the number of columns/variables/sigmas/x

    x = np.ones(n) #Start with any point (infeasible or feasible doesnt matter)
    pi = np.ones(m) #Start with any point (infeasible or feasible doesnt matter)
    sigma = np.ones(n) #Start with any point (infeasible or feasible doesnt matter)
    e = np.ones(n) #Vector of ones

    i=1 #Initialize the counter
    while True:

        #Print the values for each iteration    
        print(f"\nIteration {i}\n")
        print(f"x --> {x}\n pi --> {pi}\n")
        print(f"The objective function value is {c@x}\n")

        mu = beta*(x @ sigma)/n #Calculate the parameter mu
        X_inv = np.diag(1/x) #Calculate the diagonal matrix with the x values
        E = np.diag(sigma) #Calculate the diagonal matrix with the sigma values
        top_row = np.hstack((-(X_inv @ E),A.T)) #Calculate the first row of the main matrix
        bottom_row = np.hstack((A,np.zeros((m,m)))) #Calulate the second row of the main matrix
        D = np.vstack((top_row,bottom_row)) #Stack the two rows vertically to make the main matrix
        RHS_1 = (((c - ((A.T) @ pi)) - mu*(X_inv @ e))).reshape(-1,1) #Calculate the first part of the RHS
        RHS_2 = ((b - (A @ x))).reshape(-1,1) #Calculate the second part for the RHS
        RHS = np.vstack((RHS_1, RHS_2)) #Stack the two parts vertically to make the RHS

        deltas = np.linalg.solve(D,RHS) #Solve the system of equations
        delta_x = deltas[:n,:].flatten() #Get the deltas for x
        delta_pi = deltas[n:,:].flatten() #Get the deltas for pi
        delta_sigma = -sigma - X_inv @ ((E @ delta_x - ((mu*e)))) #Calculate deltas for sigma using this formula

        #If the absolute value of delta_x*delta_sigma for all the variables is 
        # less than the tolerance level, we terminate
        if np.all(abs(delta_x * delta_sigma) < tolerance):
            return x, sigma, pi

        theta_x = min(np.where(delta_x < 0, -x/delta_x, np.inf)) #Calcuate the theta for x
        theta_sigma = min(np.where(delta_sigma < 0, -sigma/delta_sigma, np.inf)) #Calculate the theta for sigma
        theta = min(1, alpha*theta_x, alpha*theta_sigma) #Calculate the common theta

        #Update x,sigma and pi
        x = x + theta*delta_x
        sigma = sigma + theta*delta_sigma
        pi = pi + theta*delta_pi

        #Increment the iteration counter
        i=i+1

#Main method
def main():
    method = input(f"Enter the algorithm to be used (IPM/simplex):") #Get the user input for the method
    forestry(method) #Solve the model using the preferred method by the user

main() #Execute the main function