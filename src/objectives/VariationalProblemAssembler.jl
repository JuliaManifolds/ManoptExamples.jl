@doc raw"""
This function is called by Newton's method to compute the rhs and the matrix for the Newton step
"""
function get_rhs_Jac!(b, A, h, y, integrand, transport)
    S = integrand.domain
    # Schleife über Intervalle
    for i in 1:(length(y) - 1)
        yl = y[i - 1]
        yr = y[i]
        Bcl = get_basis(S, yl, DefaultOrthonormalBasis())
        Bl = get_vectors(S, yl, Bcl)
        Bcr = get_basis(S, yr, DefaultOrthonormalBasis())
        Br = get_vectors(S, yr, Bcr)
        assemble_local_rhs!(b, h, i, yl, yr, Bl, Br, integrand)
        assemble_local_Jac!(A, h, i, yl, yr, Bl, Br, integrand, transport)
    end
    return
end

@doc raw"""
This function is called by Newton's method to compute the rhs for the simplified Newton step
"""
function get_rhs_simplified!(b, h, y, y_trial, integrand, transport)
    S = integrand.domain
    # Schleife über Intervalle
    for i in 1:(length(y) - 1)
        yl = y[i - 1]
        yr = y[i]
        yl_trial = y_trial[i - 1]
        yr_trial = y_trial[i]
        Bcl = get_basis(S, yl, DefaultOrthonormalBasis())
        Bl = get_vectors(S, yl, Bcl)
        Bcr = get_basis(S, yr, DefaultOrthonormalBasis())
        Br = get_vectors(S, yr, Bcr)
        dim = manifold_dimension(S)
        # Transport der Testfunktionen auf $T_{x_k}S$
        for k in 1:dim
            Bl[k] = transport.value(S, yl, Bl[k], yl_trial)
            Br[k] = transport.value(S, yr, Br[k], yr_trial)
        end
        assemble_local_rhs!(b, h, i, yl_trial, yr_trial, Bl, Br, integrand)
    end
    return
end

@doc raw"""
This is a helper function
"""
function assemble_local_rhs!(b, h, i, yl, yr, Bl, Br, integrand)
    dim = manifold_dimension(integrand.domain)
    idxl = dim * (i - 2)
    idxr = dim * (i - 1)
    ydotl = (yr - yl) / h
    ydotr = (yr - yl) / h
    # Trapezregel
    quadwght = 0.5 * h
    for k in 1:dim
        Bldot = -Bl[k] / h
        Brdot = Br[k] / h
        # linke Testfunktion
        if idxl >= 0
            #linker Quadraturpunkt
            tmp = integrand.value(integrand, yl, ydotl, Bl[k], Bldot)
            #rechter Quadraturpunkt
            tmp += integrand.value(integrand, yr, ydotr, 0.0 * Bl[k], Bldot)
            # Update der rechten Seite
            b[idxl + k] += quadwght * tmp
        end
        # rechte Testfunktion
        if idxr < length(b)
            tmp = integrand.value(integrand, yl, ydotl, 0.0 * Br[k], Brdot)
            tmp += integrand.value(integrand, yr, ydotr, Br[k], Brdot)
            b[idxr + k] += quadwght * tmp
        end
    end
    return
end

@doc raw"""
This is a helper function
"""
function assemble_local_Jac!(A, h, i, yl, yr, Bl, Br, integrand, transport)
    dim = manifold_dimension(integrand.domain)
    idxl = dim * (i - 2)
    idxr = dim * (i - 1)
    ydot = (yr - yl) / h
    quadwght = 0.5 * h
    nA = size(A, 1)
    #	Schleife über Testfunktionen
    for k in 1:dim
        Bdotlk = -Bl[k] / h
        Bdotrk = Br[k] / h
        # Schleife über Testfunktionen
        for j in 1:dim
            # Zeit-Ableitungen der Testfunktionen (=0 am jeweils anderen Rand)
            Bdotlj = (0 - 1) * Bl[j] / h
            Bdotrj = (1 - 0) * Br[j] / h

            # y-Ableitungen der Projektionen
            Pprimel = transport.derivative(integrand.domain, yl, Bl[j], Bl[k])
            Pprimer = transport.derivative(integrand.domain, yr, Br[j], Br[k])

            # Zeit- und y-Ableitungen der Projektionen
            Pprimedotl = (0 - 1) * Pprimel / h
            Pprimedotr = (1 - 0) * Pprimer / h

            # linke x linke Testfunktion
            if idxl >= 0
                # linker Quadraturpunkt
                # Ableitung in der Einbettung
                tmp = integrand.derivative(integrand, yl, ydot, Bl[j], Bdotlj, Bl[k], Bdotlk)
                # Modifikation für Kovariante Ableitung
                tmp += integrand.value(integrand, yl, ydot, Pprimel, Pprimedotl)
                # rechter Quadraturpunkt (siehe oben)
                tmp += integrand.derivative(integrand, yr, ydot, 0.0 * Bl[j], Bdotlj, 0.0 * Bl[k], Bdotlk)
                tmp += integrand.value(integrand, yr, ydot, 0.0 * Pprimel, Pprimedotl)
                # Update des Matrixeintrags
                A[idxl + k, idxl + j] += quadwght * tmp
                # TODO: Stimmt das auch bei nicht-symmetrischen Matrizen? j <-> k?
            end
            # linke x rechte Testfunktion
            if idxl >= 0 && idxr < nA
                # linker Quadraturpunkt
                # Ableitung in der Einbettung
                tmp = integrand.derivative(integrand, yl, ydot, 0.0 * Br[j], Bdotrj, Bl[k], Bdotlk)
                # Modifikation für Kovariante Ableitung fällt hier weg, da Terme = 0
                # rechter Quadraturpunkt
                tmp += integrand.derivative(integrand, yr, ydot, Br[j], Bdotrj, 0.0 * Bl[k], Bdotlk)
                # Symmetrisches Update der Matrixeinträge
                A[idxl + k, idxr + j] += quadwght * tmp
                A[idxr + j, idxl + k] += quadwght * tmp
            end
            # rechte x rechte Testfunktion (siehe oben)
            if idxr < nA
                tmp = integrand.derivative(integrand, yl, ydot, 0.0 * Br[j], Bdotrj, 0.0 * Br[k], Bdotrk)
                tmp += integrand.value(integrand, yl, ydot, 0.0 * Pprimer, Pprimedotr)
                tmp += integrand.derivative(integrand, yr, ydot, Br[j], Bdotrj, Br[k], Bdotrk)
                tmp += integrand.value(integrand, yr, ydot, Pprimer, Pprimedotr)

                A[idxr + k, idxr + j] += quadwght * tmp
                # TODO: Stimmt das auch bei nicht-symmetrischen Matrizen?  j <-> k?
            end
        end
    end
    return
end
