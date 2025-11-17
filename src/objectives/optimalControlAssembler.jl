"""
 A:      Matrix to be written into\\
row_idx: row index of block inside system\\
detT:    degree of test function: 1: linear, 0: constant\\
col_idx: column index of block inside system\\
detB:    degree of basis function: 1: linear, 0: constant\\
h:       length of interval\\
nCell:    total number of intervals\\
y:       iterate\\
...
"""
function get_Jac_Lyy!(eval,A,row_idx,col_idx,h, nCells,y,integrand,transport)
	M = integrand.domain
	N = integrand.precodomain
	# Schleife über Intervalle
	for i in 1:nCells

		# Evaluation of the current iterate. This routine has to be provided from outside, because Knowledge about the basis functions is needed
		yl=eval(y,i,0.0)
		yr=eval(y,i,1.0)

		#yl=ArrayPartition(getindex.(y.x, (i-1...,)))
		#yr=ArrayPartition(getindex.(y.x, (i...,)))

		Bcl=get_basis(M,yl.x[col_idx],DefaultOrthonormalBasis())
	    Bl = get_vectors(M,yl.x[col_idx], Bcl)
		Bcr=get_basis(M,yr.x[col_idx],DefaultOrthonormalBasis())
	    Br = get_vectors(M,yr.x[col_idx], Bcr)

		Tcl=get_basis(N,yl.x[row_idx],DefaultOrthonormalBasis())
	    Tl = get_vectors(N,yl.x[row_idx], Tcl)
		Tcr=get_basis(N,yr.x[row_idx],DefaultOrthonormalBasis())
	    Tr = get_vectors(N,yr.x[row_idx], Tcr)

        # In the following, all combinations of test and basis functions have to be considered.
		
        assemble_local_Jac_Lyy!(A,row_idx,col_idx,h,i,yl,yr, Bl,1,0, Tl,1,0, integrand, transport)		
		assemble_local_Jac_Lyy!(A,row_idx,col_idx,h,i,yl,yr, Br,0,1, Tl,1,0, integrand, transport)		
		assemble_local_Jac_Lyy!(A,row_idx,col_idx,h,i,yl,yr, Bl,1,0, Tr,0,1, integrand, transport)		
		assemble_local_Jac_Lyy!(A,row_idx,col_idx,h,i,yl,yr, Br,0,1, Tr,0,1, integrand, transport)		

	end
end

"""
 A:      Matrix to be written into\\
row_idx: row index of block inside system\\
col_idx: column index of block inside system\\

h:       length of interval\\
i:       index of interval\\

yl:      left value of iterate\\
yr:      right value of iterate\\

B:       basis vector for basis function\\
bfl:     0/1 scaling factor at left boundary\\
bfr:     0/1 scaling factor at right boundary \\

T:       basis vector for test function\\
tfl:     0/1 scaling factor at left boundary\\
tfr:     0/1 scaling factor at right boundary \\
...
"""
function assemble_local_Jac_Lyy!(A,row_idx, col_idx, h, i, yl, yr, B, bfl, bfr, T, tfl, tfr, integrand, transport)
 dim = manifold_dimension(integrand.domain)
 dimc = manifold_dimension(integrand.precodomain)
if tfr == 1
	idxc=dimc*(i-1)
else 
	idxc=dimc*(i-2)
end
if bfr == 1
	idx=dim*(i-1)
else 
	idx=dim*(i-2)
end

 ydot=(yr-yl)/h
 quadwght=0.5*h
 nA1=size(A,1)
 nA2=size(A,2)
 #	Schleife über Komponenten der Testfunktion
 for k in 1:dimc
    # Schleife über Komponenten der Basisfunktion
	for j in 1:dim
		# Sicherstellen, dass wir in Indexgrenzen der Matrix bleiben
        if idx+j >= 1 && idxc+k >= 1 && idx+j <= nA2 && idxc+k <= nA1

		# Zeit-Ableitungen der Basis- und Testfunktionen (=0 am jeweils anderen Rand)
     	Tdot=(tfr-tfl)*T[k]/h
		Bdot=(bfr-bfl)*B[j]/h
			
		# Modifikation für Kovariante Ableitung:	
		# y-Ableitungen der Projektionen am linken Punkt
		# P'(yl)bfl*B[j] (tfl*T(k))

		Pprimel=transport.derivative(integrand.domain,yl,bfl*B[j],tfl*T[k])
		Pprimer=transport.derivative(integrand.domain,yr,bfr*B[j],tfr*T[k])
			
		# Zeit- und y-Ableitungen der Projektionen
		Pprimedotl=(bfr-bfl)*Pprimel/h
		Pprimedotr=(bfr-bfl)*Pprimer/h	

		Pprimedot_neu = (Pprimer - Pprimel)/h
			
		# Einsetzen in die rechte Seite am rechten und linken Quadraturpunkt

		tmp = integrand.derivative(integrand,yl,ydot,bfl*B[j],Bdot,bfl*Pprimel,Pprimedot_neu)
		#tmp+=integrand.derivative(integrand,yr,ydot,bfr*B[j],Bdot,bfr*Pprimel,Pprimedot_neu)
			
		
        # Update des Matrixeintrags

		#tmp+=integrand.derivative(integrand,yl,ydot,bfl*B[j], Bdot,bfl*Pprimer,Pprimedot_neu)
		tmp+=integrand.derivative(integrand,yr,ydot,bfr*B[j],Bdot,bfr*Pprimer,Pprimedot_neu)
			
		A[idxc+k,idx+j]+=quadwght*tmp
		end
	end
 end
end

function get_rhs_simplified_y!(eval, b,row_idx,degT,h,nCells,y,y_trial,integrand,transport)
	S = integrand.precodomain
	# loop: time intervals
	for i in 1:nCells
			yl=eval(y,i,0.0)
		    yr=eval(y,i,1.0)
			
			yl_trial=eval(y_trial,i,0.0)
			yr_trial=eval(y_trial,i,1.0)
		
			Tcl=get_basis(S,yl.x[row_idx],DefaultOrthonormalBasis())
			Tl=get_vectors(S, yl.x[row_idx],Tcl)

			Tcr=get_basis(S,yr.x[row_idx],DefaultOrthonormalBasis())
	    	Tr = get_vectors(S, yr.x[row_idx], Tcr)
		
			dim = manifold_dimension(S)
		# transport test functions to $T_{x_k}S$
            # for k=1:dim
			# 	Tl[k]=transport.value(S,yl.x[row_idx],Tl[k],yl_trial.x[row_idx])
			# 	Tr[k]=transport.value(S,yr.x[row_idx],Tr[k],yr_trial.x[row_idx])
			# end
			
			if degT == 1
			assemble_local_rhs_OC!(b,row_idx, h, i, yl_trial, yr_trial, Tl, 1, 0, integrand, transport, yl, yr)		
        	assemble_local_rhs_OC!(b,row_idx, h, i, yl_trial, yr_trial, Tr, 0, 1, integrand, transport, yl, yr)		
			end
			if degT == 0
        	assemble_local_rhs_OC!(b,row_idx, h, i, yl_trial, yr_trial, Tr, 1, 1, integrand, transport, yl, yr)		
			end
	end
end

function assemble_local_rhs_OC!(b, row_idx, h, i, yl, yr, T, tlf, trf, integrand, transport, yl_vorher, yr_vorher)
    dimc = manifold_dimension(integrand.precodomain)
	S = integrand.precodomain
	if trf == 1
     idx=dimc*(i-1)
	else 
	 idx=dimc*(i-2)
	end
	ydotl=(yr-yl)/h
	ydotr=(yr-yl)/h
	# trapezoidal rule
	quadwght = 0.5*h   
	for k in 1:dimc
		# finite differences, taking into account values of test function at both endpoints
        if idx+k > 0 && idx+k <= length(b)

			Pl = transport.value(S, yl.x[row_idx], transport.derivative(integrand.domain,yl.x[row_idx],tlf*T[k],tlf*yl.x[3]), yl.x[row_idx])
			Pr = transport.value(S, yr.x[row_idx], transport.derivative(integrand.domain,yr.x[row_idx],trf*T[k],trf*yr.x[3]), yr.x[row_idx])
			Pdot = (Pr-Pl)/h

			tmp = integrand.value(integrand, yl, ydotr, tlf*Pl, Pdot)
			tmp += integrand.value(integrand, yr, ydotr, trf*Pr, Pdot)
			# Update rhs
		  	b[idx+k]+= quadwght*tmp	
		end
	end
end


function get_Jac_simplified!(eval,A,row_idx,degT,col_idx,degB,h, nCells,y,y_trial,integrand,transport)
	M = integrand.domain
	N = integrand.precodomain
	# loop: time intervals
	for i in 1:nCells

		# Evaluation of the current iterate. This routine has to be provided from outside, because knowledge about the basis functions is needed
		yl=eval(y,i,0.0)
		yr=eval(y,i,1.0)

		yl_trial=eval(y_trial,i,0.0)
		yr_trial=eval(y_trial,i,1.0)

		Bcl=get_basis(M,yl_trial.x[col_idx],DefaultOrthonormalBasis())
	    Bl = get_vectors(M,yl_trial.x[col_idx], Bcl)
		Bcr=get_basis(M,yr_trial.x[col_idx],DefaultOrthonormalBasis())
	    Br = get_vectors(M,yr_trial.x[col_idx], Bcr)

		Tcl=get_basis(N,yl.x[row_idx],DefaultOrthonormalBasis())
	    Tl = get_vectors(N,yl.x[row_idx], Tcl)
		Tcr=get_basis(N,yr.x[row_idx],DefaultOrthonormalBasis())
	    Tr = get_vectors(N,yr.x[row_idx], Tcr)

		# for k=1:manifold_dimension(M)
		# 	Bl[k]=transport.value(M,yl.x[row_idx],Bl[k],yl_trial.x[row_idx])
		# 	Br[k]=transport.value(M,yr.x[row_idx],Br[k],yr_trial.x[row_idx])
		# end

		for k=1:manifold_dimension(N)
			Tl[k]=transport.value(N,yl.x[row_idx],Tl[k],yl_trial.x[row_idx])
			Tr[k]=transport.value(N,yr.x[row_idx],Tr[k],yr_trial.x[row_idx])
		end

        # In the following, all combinations of test and basis functions have to be considered.
		
		# The case, where both test and basis functions are linear. We have 2x2=4 combinations, since there are two test/basis functions on each interval
		if degT==1 && degB == 1
    	    assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl_trial,yr_trial, Bl,1,0, Tl,1,0, integrand, transport)		
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl_trial,yr_trial, Br,0,1, Tl,1,0, integrand, transport)		
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl_trial,yr_trial, Bl,1,0, Tr,0,1, integrand, transport)		
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl_trial,yr_trial, Br,0,1, Tr,0,1, integrand, transport)		
		end
		# The case, where both test functions are linear and basis functions are piecewies constant. We have 1x2=2 combinations, since there are are two test functions and 1 basis function on each interval
		if degT==1 && degB == 0 
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl_trial,yr_trial, Br,1,1, Tl,1,0, integrand, transport)		
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl_trial,yr_trial, Br,1,1, Tr,0,1, integrand, transport)		
		end
		if degT==0 && degB == 1 
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl_trial,yr_trial, Bl,1,0, Tr,1,1, integrand, transport)		
			assemble_local_Jac_with_connection!(A,row_idx,col_idx,h,i,yl_trial,yr_trial, Br,0,1, Tr,1,1, integrand, transport)		
		end
		# Other cases could be added here. In the rod example I did not need them, thus I havent implemented them
	end
end