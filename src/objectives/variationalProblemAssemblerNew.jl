
raw"""
Helper function that builds a basis of the tangent space of M at p
"""

function build_base(M::AbstractManifold, p)
    Bl = get_basis(M,p,DefaultOrthonormalBasis())
    return get_vectors(M,p, Bl)
end

@doc raw"""
This function is called by Newton's method to compute one block of the matrix for the Newton step

Input:

M:                      Product manifold\\
y:                      iterate\\
eval:                   function that evaluates y at left and right boundary point of i-th interval, signature: eval(y, i, scaling), must return an element of M
A:                      Matrix to be written into\\
integrand:	            integrand of the functional as a struct, must have a field value and a field derivative\\
transport:	            vectortransport used to compute the connection term (as a struct, must have a field value and a field derivative)
h:                      length of interval\\
number_of_intervals:    total number of intervals\\

Keyword arguments:

row_index:                  row index of block inside system\\
column_index:               column index of block inside system\\
degree_test_function:       degree of test functions (1: linear, 0: constant)\\
degree_ansatz_function:     degree of ansatz functions (1: linear, 0: constant)\\
test_space:    				base manifold of the space of test functions\\
ansatz_space:   			base manifold of the space of ansatz functions

...
"""
function get_jacobian_block!(M::ProductManifold, y, eval, A, integrand, transport, h, number_of_intervals; row_index = nothing, column_index = nothing, degree_test_function = 1, degree_ansatz_function = 1, test_space = nothing, ansatz_space = nothing)
    isnothing(row_index) && error("Please provide the row index of the block to be assembled")
    isnothing(column_index) && error("Please provide the column index of the block to be assembled")
    isnothing(test_space) && error("Please provide the space of the test functions")
    isnothing(ansatz_space) && error("Please provide the space of the ansatz functions")

	# loop: time intervals
	for i in 1:number_of_intervals

		# Evaluation of the current iterate. This routine has to be provided from outside, because knowledge about the ansatz functions is needed
		yl=eval(y,i,0.0)
		yr=eval(y,i,1.0)

        base_ansatz_functions_left = build_base(ansatz_space, yl[M, column_index])
        base_ansatz_functions_right = build_base(ansatz_space, yr[M, column_index])
        base_test_functions_left = build_base(test_space, yl[M, row_index])
        base_test_functions_right = build_base(test_space, yr[M, row_index])

        # In the following, all combinations of test and ansatz functions have to be considered.
		
		# The case, where both test and ansatz functions are linear. We have 2x2=4 combinations, since there are two test/ansatz functions on each interval
		if degree_test_function == 1 && degree_ansatz_function == 1

    	    assemble_local_jacobian_with_connection!(M, yl, yr, A, h, i, base_ansatz_functions_left, 1, 0, base_test_functions_left, 1, 0, integrand, transport; row_index = row_index)		
            assemble_local_jacobian_with_connection!(M, yl, yr, A, h, i, base_ansatz_functions_right, 0, 1, base_test_functions_left, 1, 0, integrand, transport; row_index = row_index)	
			assemble_local_jacobian_with_connection!(M, yl, yr, A, h, i, base_ansatz_functions_left, 1, 0, base_test_functions_right, 0, 1, integrand, transport; row_index = row_index)	
            assemble_local_jacobian_with_connection!(M, yl, yr, A, h, i, base_ansatz_functions_right, 0, 1, base_test_functions_right, 0, 1, integrand, transport; row_index = row_index)	

		end
		# The case, where test functions are linear and ansatz functions are piecewise constant. We have 1x2=2 combinations, since there are are two test functions and one ansatz function on each interval
		if degree_test_function == 1 && degree_ansatz_function == 0 	

            assemble_local_jacobian_with_connection!(M, yl, yr, A, h, i, base_ansatz_functions_right, 1, 1, base_test_functions_left, 1, 0, integrand, transport; row_index = row_index)		
            assemble_local_jacobian_with_connection!(M, yl, yr, A, h, i, base_ansatz_functions_right, 1, 1, base_test_functions_right, 0, 1, integrand, transport; row_index = row_index)

		end
        # The case, where test functions are piecewise constant and ansatz functions are linear. We have 2x1=2 combinations, since there are are two ansatz functions and one test function on each interval
		if degree_test_function == 0 && degree_ansatz_function == 1 
	
            assemble_local_jacobian_with_connection!(M, yl, yr, A, h, i, base_ansatz_functions_left, 1, 0, base_test_functions_right, 1, 1, integrand, transport; row_index = row_index)		
            assemble_local_jacobian_with_connection!(M, yl, yr, A, h, i, base_ansatz_functions_right, 0, 1, base_test_functions_right, 1, 1, integrand, transport; row_index = row_index)	

		end
		# Other cases could be added here.
	end
end

@doc raw"""
This function is called by Newton's method to compute the matrix for the Newton step

Input:

M:                      Manifold\\
y:                      iterate\\
eval:                   function that evaluates y at left and right boundary point of i-th interval, signature: eval(y, i, scaling), must return an element of M
A:                      Matrix to be written into\\
integrand:	            integrand of the functional as a struct, must have a field value and a field derivative\\
transport:	            vectortransport used to compute the connection term (as a struct, must have a field value and a field derivative)
h:                      length of interval\\
number_of_intervals:    total number of intervals\\

Keyword arguments:

degree_test_function:       degree of test functions (1: linear, 0: constant)\\
degree_ansatz_function:      degree of ansatz functions (1: linear, 0: constant) (test functions and ansatz functions are the same in this case)\\
test_space:    Space of test functions\\
ansatz_space:   Space of ansatz functions (test functions and ansatz functions are the same in this case)

...
"""

function get_jacobian!(M::AbstractManifold, y, eval, A, integrand, transport, h, number_of_intervals; row_index = nothing, column_index = nothing, degree_test_function = 1, degree_ansatz_function = degree_test_function, test_space = nothing, ansatz_space = test_space)

    isnothing(test_space) && error("Please provide the space of the test functions")

	# loop: time intervals
	for i in 1:number_of_intervals
		yl=eval(y,i,0.0)
		yr=eval(y,i,1.0)

        base_test_functions_left = build_base(test_space, yl)
        base_test_functions_right = build_base(test_space, yr)

        assemble_local_jacobian_with_connection!(M, yl, yr, A, h, i, base_test_functions_left, 1, 0, base_test_functions_left, 1, 0, integrand, transport)				
        assemble_local_jacobian_with_connection!(M, yl, yr, A, h, i, base_test_functions_right, 0, 1, base_test_functions_left, 1, 0, integrand, transport)		
        assemble_local_jacobian_with_connection!(M, yl, yr, A, h, i, base_test_functions_left, 1, 0, base_test_functions_right, 0, 1, integrand, transport)			
        assemble_local_jacobian_with_connection!(M, yl, yr, A, h, i, base_test_functions_right, 0, 1, base_test_functions_right, 0, 1, integrand, transport)	
	end
end

@doc raw"""
This function is called by Newton's method to compute one block of the right hand side for the Newton step

Input:

M:                      Product manifold\\
y:                      iterate\\
eval:                   function that evaluates y at left and right boundary point of i-th interval, signature: eval(y, i, scaling), must return an element of M
b:                      vector to be written into\\
h:                      length of interval\\
number_of_intervals:    total number of intervals\\
integrand:	            integrand of the functional as a struct, must have a field value and a field derivative

Keyword arguments:

row_index:                  if M is a ProductManifold, the row index of the entry inside the right hand side must be provided\\
degree_test_function:       degree of test functions (1: linear, 0: constant)\\
test_space:    Space of test functions\\

...
"""

function get_right_hand_side_row!(M::ProductManifold, y, eval, b, h, number_of_intervals, integrand; row_index = nothing, degree_test_function = 1, test_space = nothing)
	
    isnothing(row_index) && error("Please provide the row index of the right hand side to be assembled")
    isnothing(test_space) && error("Please provide the space of the test functions")

	# loop: time intervals
	for i in 1:number_of_intervals
		y_left = eval(y,i,0.0)
		y_right = eval(y,i,1.0)

        base_test_function_left = build_base(test_space, y_left[M, row_index])
        base_test_function_right = build_base(test_space, y_right[M, row_index])

		if degree_test_function == 1
			assemble_local_right_hand_side!(M, y_left, y_right, b, h, i, base_test_function_left, 1, 0, integrand)		
        	assemble_local_right_hand_side!(M, y_left, y_right, b, h, i, base_test_function_right, 0, 1, integrand)		
		end

		if degree_test_function == 0
        	assemble_local_right_hand_side!(M, y_left, y_right, b, h, i, base_test_function_right, 1, 1, integrand)		
		end
	end
end

@doc raw"""
This function is called by Newton's method to compute the right hand side for the Newton step

Input:

M:                      manifold\\
y:                      iterate\\
eval:                   function that evaluates y at left and right boundary point of i-th interval, signature: eval(y, i, scaling), must return an element of M
b:                      vector to be written into\\
h:                      length of interval\\
number_of_intervals:    total number of intervals\\
integrand:	            integrand of the functional as a struct, must have a field value and a field derivative

Keyword arguments:

degree_test_function:       degree of test functions (1: linear, 0: constant)\\
test_space:    Space of test functions\\

...
"""

function get_right_hand_side!(M::AbstractManifold, y, eval, b, h, number_of_intervals, integrand; row_index = nothing, degree_test_function = 1, test_space = nothing)
	
    isnothing(test_space) && error("Please provide the space of the test functions")

	# loop: time intervals
	for i in 1:number_of_intervals
		y_left = eval(y,i,0.0)
		y_right = eval(y,i,1.0)

        base_test_function_left = build_base(test_space, y_left)
        base_test_function_right = build_base(test_space, y_right)

		if degree_test_function == 1
			assemble_local_right_hand_side!(M, y_left, y_right, b, h, i, base_test_function_left, 1, 0, integrand)		
        	assemble_local_right_hand_side!(M, y_left, y_right, b, h, i, base_test_function_right, 0, 1, integrand)		
		end
	end
end


@doc raw"""
This function is called by Newton's method to compute the right hand side for the simplified Newton step. This is needed for the affine-covariant damping strategy

Input:

M:                      product manifold\\
y:                      iterate\\
y_trial:				next iterate\\
eval:                   function that evaluates y at left and right boundary point of i-th interval, signature: eval(y, i, scaling), must return an element of M
b:                      vector to be written into\\
h:                      length of interval\\
number_of_intervals:    total number of intervals\\
integrand:	            integrand of the functional as a struct, must have a field value and a field derivative
transport: 				vectortransport used to transport the test functions (as a struct, must have a field value and a field derivative)

Keyword arguments:

row_index:					if M is a ProductManifold, the row index of the entry inside the right hand side must be provided
degree_test_function:       degree of test functions (1: linear, 0: constant)\\
test_space:    Space of test functions\\

...
"""


function get_right_hand_side_simplified_row!(M::ProductManifold, y, y_trial, eval, b, h, number_of_intervals, integrand, transport; row_index = nothing, degree_test_function = 1, test_space = nothing)

    isnothing(row_index) && error("Please provide the row index of the right hand side to be assembled")
    isnothing(test_space) && error("Please provide the space of the test functions")

	# loop: time intervals
	for i in 1:number_of_intervals
		y_left = eval(y,i,0.0)
		y_right = eval(y,i,1.0)

		y_trial_left = eval(y_trial,i,0.0)
		y_trial_right = eval(y_trial,i,1.0)

        base_test_functions_left = build_base(test_space, y_left[M, row_index])

	    base_test_functions_right = build_base(test_space, y_right[M, row_index])
	
		dim = manifold_dimension(test_space)

		# transport the test functions 
        for k=1:dim
			base_test_functions_left[k] = transport.value(test_space, y_left[M, row_index], base_test_functions_left[k], y_trial_left[M, row_index])
			base_test_functions_right[k] = transport.value(test_space, y_right[M, row_index], base_test_functions_right[k], y_trial_right[M, row_index])
		end

		if degree_test_function == 1
		    assemble_local_right_hand_side!(M, y_trial_left, y_trial_right, b, h, i, base_test_functions_left, 1, 0, integrand)		
            assemble_local_right_hand_side!(M, y_trial_left, y_trial_right, b, h, i, base_test_functions_right, 0, 1, integrand)		
		end

		if degree_test_function == 0
            assemble_local_right_hand_side!(M, y_trial_left, y_trial_right, b, h, i, base_test_functions_right, 1, 1, integrand)		
		end
	end
end

@doc raw"""
This function is called by Newton's method to compute the right hand side for the simplified Newton step. This is needed for the affine-covariant damping strategy

Input:

M:                      manifold\\
y:                      iterate\\
y_trial:				next iterate\\
eval:                   function that evaluates y at left and right boundary point of i-th interval, signature: eval(y, i, scaling), must return an element of M
b:                      vector to be written into\\
h:                      length of interval\\
number_of_intervals:    total number of intervals\\
integrand:	            integrand of the functional as a struct, must have a field value and a field derivative
transport: 				vectortransport used to transport the test functions (as a struct, must have a field value and a field derivative)

Keyword arguments:

degree_test_function:       degree of test functions (1: linear, 0: constant)\\
test_space:    Space of test functions\\

...
"""


function get_right_hand_side_simplified!(M::AbstractManifold, y, y_trial, eval, b, h, number_of_intervals, integrand, transport; row_index = nothing, degree_test_function = 1, test_space = nothing)

    isnothing(test_space) && error("Please provide the space of the test functions")

	# loop: time intervals
	for i in 1:number_of_intervals
		y_left = eval(y,i,0.0)
		y_right = eval(y,i,1.0)
			
		y_trial_left = eval(y_trial,i,0.0)
		y_trial_right = eval(y_trial,i,1.0)
		
        base_test_functions_left = build_base(test_space, y_left)
		base_test_functions_right = build_base(test_space, y_right)
		
		dim = manifold_dimension(test_space)
		# transport the test functions
        for k=1:dim
			base_test_functions_left[k] = transport.value(test_space, y_left, base_test_functions_left[k], y_trial_left)
			base_test_functions_right[k] = transport.value(test_space, y_right, base_test_functions_right[k], y_trial_right)
		end


		if degree_test_function == 1
		    assemble_local_right_hand_side!(M, y_trial_left, y_trial_right, b, h, i, base_test_functions_left, 1, 0, integrand)		
            assemble_local_right_hand_side!(M, y_trial_left, y_trial_right, b, h, i, base_test_functions_right, 0, 1, integrand)		
		end
	end
end


"""
This function is called to assemble (one block of) the matrix for the Newton step locally (i.e. in one interval)

M :                     manifold
y_left:                 left value of iterate\\
y_right:                right value of iterate\\
A:                      Matrix to be written into\\
h:                      length of interval\\
i:                      index of interval\\
B:                      basis vector for ansatz function\\
bfl:                    0/1 scaling factor at left boundary\\
bfr:                    0/1 scaling factor at right boundary \\
T:                      basis vector for test function\\
tfl:                    0/1 scaling factor at left boundary\\
tfr:                    0/1 scaling factor at right boundary \\
integrand:	            integrand of the functional as a struct, must have a field value and a field derivative\\
transport:	            vectortransport used to compute the connection term (as a struct, must have a field value and a field derivative)

Keyword Arguments:

row_index:                row index of block inside system\\
y_component_left:         if M is a ProductManifold this is one of the components of y_left, otherwise it is just y_left
y_component_right:         if M is a ProductManifold this is one of the components of y_right, otherwise it is just y_right


...
"""
function assemble_local_jacobian_with_connection!(M, y_left, y_right, A, h, i, B, bfl, bfr, T, tfl, tfr, integrand, transport; row_index = nothing, y_component_left = isnothing(row_index) ? y_left : y_left[M, row_index], y_component_right = isnothing(row_index) ? y_right : y_right[M, row_index], M_component = isnothing(row_index) ? M : M[row_index])
 dim_ansatz = length(B)
 dim_test = length(T)

if tfr == 1
	idxc = dim_test*(i-1)
else 
	idxc = dim_test*(i-2)
end
if bfr == 1
	idx = dim_ansatz*(i-1)
else 
	idx = dim_ansatz*(i-2)
end

 ydot = (y_right-y_left)/h # approximate time derivative of y
 quadrature_weight = 0.5*h
 nA1 = size(A,1)
 nA2 = size(A,2)

 #	loop: components of test functions
 for k in 1:dim_test
    # loop: components of ansatz functions
	for j in 1:dim_ansatz
		# ensure that indices remain in bounds
        if idx+j >= 1 && idxc+k >= 1 && idx+j <= nA2 && idxc+k <= nA1

		# approximation of time derivative of ansatz and test functions (=0 am jeweils anderen Rand)
     	Tdot = (tfr-tfl)*T[k]/h
		Bdot = (bfr-bfl)*B[j]/h


    	# derivative (using the embedding) at right and left quadrature point
        tmp = integrand.derivative(integrand,y_left,ydot,bfl*B[j],Bdot,tfl*T[k],Tdot)	
			
		tmp += integrand.derivative(integrand,y_right,ydot,bfr*B[j],Bdot,tfr*T[k],Tdot)	
			
		# modification for covariant derivative:	
		# derivative of the vector transport w.r.t. y at left quadrature point
		# P'(yl)bfl*B[j] (tfl*T(k))

		Pprime_left = transport.derivative(M_component,y_component_left,bfl*B[j],tfl*T[k])
		Pprime_right = transport.derivative(M_component,y_component_right,bfr*B[j],tfr*T[k])

		# approximation of the time derivative of the vector transport

		Pprimedot = (Pprime_right - Pprime_left)/h

		# applying rhs at right and left quadrature point
		tmp += integrand.value(integrand,y_left,ydot,bfl*Pprime_left,Pprimedot)
		tmp += integrand.value(integrand,y_right,ydot,bfr*Pprime_right,Pprimedot)
	
        # Update matrix entry		
		A[idxc+k,idx+j] += quadrature_weight*tmp
		end
	end
 end
end

"""
This function is called to assemble the right hand side for the Newton step locally (i.e. in one interval)

M :                     manifold
y_left:                 left value of iterate\\
y_right:                right value of iterate\\
b:                      Vector to be written into\\
h:                      length of interval\\
i:                      index of interval\\
base_test_functions:    basis of the space of test functions\\
tlf:                    0/1 scaling factor at left boundary\\
trf:                    0/1 scaling factor at right boundary \\
integrand:	            integrand of the functional as a struct, must have a field value and a field derivative\\

"""

function assemble_local_right_hand_side!(M, y_left, y_right, b, h, i, base_test_functions, tlf, trf, integrand)
    dim_test = length(base_test_functions)
	if trf == 1
    	idx = dim_test*(i-1)
	else 
		idx = dim_test*(i-2)
	end

	ydotl=(y_right-y_left)/h
	ydotr=(y_right-y_left)/h

	# trapezoidal rule
	quadwght = 0.5*h   
	for k in 1:dim_test
		# finite differences, taking into account values of test function at both endpoints
        if idx+k > 0 && idx+k <= length(b)
			
			Tdot = (trf-tlf)*base_test_functions[k]/h
			# left quadrature point
			tmp =  integrand.value(integrand, y_left, ydotl, tlf*base_test_functions[k], Tdot)	
			# right quadrature point
			tmp += integrand.value(integrand, y_right, ydotr, trf*base_test_functions[k], Tdot)	
			# Update rhs
		  	b[idx+k]+= quadwght*tmp	
		end
	end
end