"""The WaveBlocks Project

This file contains code for evaluating inner products
and matrix elements by using a specially adapted
numerical steepest descent technique.

@author: R. Bourquin
@copyright: Copyright (C) 2013 R. Bourquin
@license: Modified BSD License
"""

from numpy import (zeros, ones, diag, squeeze,  conjugate, transpose, dot, einsum,
                   zeros_like, product, indices, flipud, complexfloating, imag, nan_to_num)
from scipy import exp, sqrt, pi
from scipy.linalg import inv, schur, det, sqrtm

from Quadrature import Quadrature

__all__ = ["NSDInhomogeneous"]


class NSDInhomogeneous(Quadrature):
    r"""
    """

    def __init__(self, QR=None):
        # Pure convenience to allow setting of quadrature rule in constructor
        if QR is not None:
            self.set_qr(QR)
        else:
            self._QR = None


    def __str__(self):
        return "Inhomogeneous numerical steepest descent using a " + str(self._QR)


    def get_description(self):
        r"""Return a description of this quadrature object.
        A description is a ``dict`` containing all key-value pairs
        necessary to reconstruct the current instance. A description
        never contains any data.
        """
        d = {}
        d["type"] = "NSDInhomogeneous"
        d["qr"] = self._QR.get_description()
        return d


    def initialize_packet(self, pacbra, packet=None):
        r"""Provide the wavepacket parts of the inner product to evaluate.
        Since the quadrature is inhomogeneous, different wavepackets can be
        used for the 'bra' as well as the 'ket' part.

        :param pacbra: The packet that is used for the 'bra' part.
        :param packet: The packet that is used for the 'ket' part.
        """
        # Allow to omit the ket if it is the same as the bra
        if packet is None:
            packet = pacbra

        self._pacbra = pacbra
        self._packet = packet


    def initialize_operator(self, operator=None, matrix=False, eval_at_once=False):
        r"""Provide the operator part of the inner product to evaluate.
        This function initializes the operator used for quadratures
        and for building matrices.

        Note that the operator must not have residues and can be
        maximally polynomial but not exponential.

        :param operator: The operator of the inner product.
                         If ``None`` a suitable identity is used.
        :param matrix: Set this to ``True`` (Default is ``False``) in case
                       we want to compute the matrix elements.
                       For nasty technical reasons we can not yet unify
                       the operator call syntax.
        :param eval_at_once: Flag to tell whether the operator supports the ``entry=(r,c)`` call syntax.
        :type eval_at_once: Boolean, default is ``False``.
        """
        # TODO: Make this more efficient, only compute values needed at each (r,c) step.
        #       For this, 'operator' must support the 'component=(r,c)' option.
        # Operator is None is interpreted as identity transformation
        if operator is None:
            self._operator = lambda nodes, dummy, entry=None: ones((1,nodes.shape[1])) if entry[0] == entry[1] else zeros((1,nodes.shape[1]))
        else:
            # Wrap the operator inside a lambda ignoring the dummy parameter
            # This allows to use the same nsd code for quadrature and matrix construction.
            if matrix is False:
                self._operator = lambda nodes, dummy, entry=None: operator(nodes, entry=entry)
            else:
                self._operator = operator
        self._eval_at_once = eval_at_once


    def mix_parameters(self, Pibra, Piket):
        r"""Mix the two parameter sets :math:`\Pi_i` and :math:`\Pi_j`
        from the 'bra' and the 'ket' wavepackets :math:`\Phi\left[\Pi_i\right]`
        and :math:`\Phi^\prime\left[\Pi_j\right]`.

        :param Pibra: The parameter set :math:`\Pi_i` from the bra part wavepacket.
        :param Piket: The parameter set :math:`\Pi_j` from the ket part wavepacket.
        :return: The mixed parameters :math:`q_0` and :math:`Q_S`. (See the theory for details.)
        """
        # <Pibra | ... | Piket>
        qr, pr, Qr, Pr = Pibra
        qc, pc, Qc, Pc = Piket

        # Mix the parameters
        Gr = dot(Pr, inv(Qr))
        Gc = dot(Pc, inv(Qc))

        r = imag(Gc - conjugate(Gr.T))
        s = imag(dot(Gc, qc) - dot(conjugate(Gr.T), qr))

        q0 = dot(inv(r), s)
        Q0 = 0.5 * r

        # Here we can not avoid the matrix root by using svd
        Qs = inv(sqrtm(Q0))

        return (q0, Qs)


    def build_bilinear(self, Pibra, Piket):
        r"""Convert the oscillator :math:`-\overline{g_k} + g_l`
        occuring in the integral :math:`\langle\phi_k\left[\Pi_k\right] | \phi_l\left[\Pi_l\right]\rangle`
        into a bilinear form
        :math:`g(x) = \underline{x}^{\mathrm{H}} \mathbf{A} \underline{x} + \underline{b}^{\mathrm{T}} \underline{x} + c`.

        :param Pibra: The parameters :math:`\Pi_k = (\underline{q_k}, \underline{p_k}, \mathbf{Q_k}, \mathbf{P_k})` from the 'bra' packet.
        :param Piket: The parameters :math:`\Pi_l = (\underline{q_l}, \underline{p_l}, \mathbf{Q_l}, \mathbf{P_l})` from the 'ket' packet.
        :return: Three arrays: a matrix :math:`\mathbf{A}` of shape :math:`D \times D`,
                 a vector :math:`\underline{b}` of shape :math:`D \times 1` and a scalar value :math:`c`.
        """
        qr, pr, Qr, Pr = Pibra[:4]
        qc, pc, Qc, Pc = Piket[:4]

        Gr = dot(Pr, inv(Qr))
        Gc = dot(Pc, inv(Qc))

        # Merge into a single oscillator
        A = 0.5 * (Gc - conjugate(transpose(Gr)))
        b = (0.5 * (  dot(Gr, qr)
                    - dot(conjugate(transpose(Gc)), qc)
                    + dot(transpose(Gr), conjugate(qr))
                    - dot(conjugate(Gc), conjugate(qc))
                   )
             + (pc - conjugate(pr))
            )
        b = conjugate(b)
        c = (0.5 * (  dot(conjugate(transpose(qc)), dot(Gc, qc))
                    - dot(conjugate(transpose(qr)), dot(conjugate(transpose(Gr)), qr)))
                 + (dot(conjugate(transpose(qr)),pr) - dot(conjugate(transpose(pc)),qc))
            )
        return A, b, c


    def update_oscillator(self, T, i):
        r"""Given a upper triangular matrix :math:`\mathbf{T} \in \mathbb{C}^{D \times D}` representing
        the oscillator :math:`g(x) = \underline{x}^{\mathrm{H}} \mathbf{T} \underline{x}`, update
        its entries according to:

        .. math::

            t_{j,j} & := t_{j,j} - \frac{t_{i,j}^2}{4 t_{i,i}} \\
            t_{j,k} & := t_{j,k} - \frac{t_{i,j} t_{i,k}}{2 t_{i,i}} \,, \quad k > j \,.

        :param T: The matrix :math:`\mathbf{T}` we want to update. (Note that the matrix
                  is not modified and a copy returned.)
        :param i: The row :math:`0 \leq i \leq D-1` which is taken as base for the updates.
        :return: The updated matrix :math:`\mathbf{T}`.
        """
        Ti = T.copy()
        rr, cc = T.shape

        if T[i-1,i-1] == 0:
            # TODO: Prove that this never happens or handle it correctly!
            print("Warning: 'update_oscillator' encountered a RESIDUE situation!")
            return Ti

        # Diagonal Elements
        for j in xrange(i,rr):
            Ti[j,j] = T[j,j] - T[i-1,j]**2 / (4.0*T[i-1,i-1])

        # Others
        for r in xrange(i,rr):
            for c in xrange(r+1,cc):
                Ti[r,c] = T[r,c] - T[i-1,r]*T[i-1,c] / (2*T[i-1,i-1])

        return Ti


    def prepare(self, rows, cols):
        r"""Precompute some values needed for evaluating the quadrature
        :math:`\langle \Phi_i | f(x) | \Phi^\prime_j \rangle` or the corresponding
        matrix over the basis functions of :math:`\Phi_i` and :math:`\Phi^\prime_j`.

        :param rows: A list of all :math:`i` with :math:`0 \leq i \leq N`
                     selecting the :math:`\Phi_i` for which te precompute values.
        :param cols: A list of all :math:`j` with :math:`0 \leq j \leq N`
                     selecting the :math:`\Phi^\prime_j` for which te precompute values.

        Note that the two arguments are not used in the current implementation.
        """
        # Unpack quadrature rules
        self._nodes = self._QR.get_nodes()
        self._weights = self._QR.get_weights()
        self._sqrtnodes = product(sqrt(self._nodes), axis=0)

        # Signs
        D = self._packet.get_dimension()
        x,y = indices((D, 2**D))
        self._allsigns = -2.0*(flipud((y>>x) & 1)-0.5)


    def do_nsd(self, row, col):
        r"""Evaluates by numerical steepest descent the integral
        :math:`\langle \Phi_i | f | \Phi^\prime_j \rangle` for a polynomial
        function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi^\prime_j` of :math:`\Psi^\prime`.
        :return: A complex valued matrix of shape :math:`|\mathfrak{K}_i| \times |\mathfrak{K}^\prime_j|`.
        """
        D = self._packet.get_dimension()
        N = self._packet.get_number_components()
        eps = self._packet.get_eps()
        Pibra = self._pacbra.get_parameters(component=row)
        Piket = self._packet.get_parameters(component=col)
        Pimix = self.mix_parameters(Pibra[:4], Piket[:4])

        # Combine oscillators
        A, b, c = self.build_bilinear(Pibra[:4], Piket[:4])

        # Schur decomposition of A = U^H T U
        T, U = schur(A, output="complex")
        U = conjugate(transpose(U))

        # Oscillator updates
        for i in xrange(1, D):
            T = self.update_oscillator(T, i)

        X = inv(A + transpose(A))
        ctilde = c - 0.5 * dot(transpose(b), dot(X, b))

        # Prefactor originating from constant term c
        eps = self._packet.get_eps()
        w = 1.0 / eps**2
        prefactor = exp(1.0j * w * ctilde)

        # Take out diagonals of T
        Dk = diag(T).reshape((D,1))
        # Tau (path parametrization variable)
        tk = self._nodes / w
        # Paths
        pathsqrtpart = sqrt(1.0j * tk / Dk)
        # Path derivatives
        dht = sqrt(1.0j / (4.0 * Dk * tk))
        dhtp = product(dht, axis=0)

        # The path-independent but non-constant part of the integrand
        quadrand = dhtp * self._sqrtnodes * self._weights / w**D

        # Another normalization prefactor
        # This is what differs the constant part of phi_0 from 1.
        # We loose it when dividing by phi_0 hence manually add it again.
        # TODO: Do we need mixing parameters here?
        #       Preliminary answer: no
        fr = (pi*eps**2)**(-0.25*D) * 1.0/sqrt(det(Pibra[2]))
        fc = (pi*eps**2)**(-0.25*D) * 1.0/sqrt(det(Piket[2]))
        normfactor = conjugate(fr)*fc

        # Compute global phase difference
        phase = exp(1.0j/eps**2 * (Piket[4]-conjugate(Pibra[4])))

        # Packets can also have different basis sizes
        Kbra = self._pacbra.get_basis_shapes(component=row).get_basis_size()
        Kket = self._packet.get_basis_shapes(component=col).get_basis_size()

        # Compute all contour integrals
        M = zeros((Kbra,Kket), dtype=complexfloating)

        for i in xrange(2**D):
            # Signs
            signs = self._allsigns[:,i].reshape((D,))
            # Construct full paths
            ht = zeros_like(pathsqrtpart)
            for k in reversed(xrange(D)):
                ht[k,:] = ht[k,:] + signs[k] * pathsqrtpart[k,:]
                for j in xrange(k+1, D):
                    ht[k,:] = ht[k,:] - 0.5*T[k,j]/T[k,k] * ht[j,:]
            # Back transformation
            h = dot(conjugate(transpose(U)), ht) - dot(X, b)
            # Non-oscillatory parts
            # Wavepacket
            # TODO: This is a huge hack: division by phi_0 not stable?
            basisr = self._pacbra.evaluate_basis_at(conjugate(h), row, prefactor=False)
            basisr = basisr / basisr[0,:]
            basisc = self._packet.evaluate_basis_at(h, col, prefactor=False)
            basisc = basisc / basisc[0,:]
            # Basis division by phi0 may introduce NaNs
            #basisr = nan_to_num(basisr)
            #basisc = nan_to_num(basisc)

            # Operator should support the component notation for efficiency
            if self._eval_at_once is True:
                # TODO: Sure, this is inefficient, but we can not do better right now.
                opath = self._operator(h, Pimix[0])[row*N+col]
            else:
                opath = self._operator(h, Pimix[0], entry=(row,col))
            # Do the quadrature
            factor = (opath * quadrand).reshape((-1,))
            # Sum up matrices over all quadrature nodes
            M = M + einsum("k,ik,jk", factor, conjugate(basisr), basisc)

        return phase * normfactor * prefactor * M


    def perform_quadrature(self, row, col):
        r"""Evaluates by numerical steepest descent the integral
        :math:`\langle \Phi_i | f | \Phi^\prime_j \rangle` for a polynomial
        function :math:`f(x)` with :math:`x \in \mathbb{R}^D`.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi^\prime_j` of :math:`\Psi^\prime`.
        :return: A single complex floating point number.
        """
        if not self._QR.get_dimension() == self._packet.get_dimension():
            raise ValueError("Quadrature dimension does not match the wavepacket dimension")

        M = self.do_nsd(row, col)
        # Include the coefficients as c^H M c
        cbra = self._pacbra.get_coefficients(component=row)
        cket = self._packet.get_coefficients(component=col)
        I = squeeze(dot(transpose(conjugate(cbra)), dot(M, cket)))
        # Handle NaNs if any
        I = nan_to_num(I)
        return I


    def perform_build_matrix(self, row, col):
        r"""Computes by standard quadrature the matrix elements
        :math:`\langle\Phi_i | f |\Phi^\prime_j\rangle` for a general function
        :math:`f(x)` with :math:`x \in \mathbb{R}^D`.

        :param row: The index :math:`i` of the component :math:`\Phi_i` of :math:`\Psi`.
        :param row: The index :math:`j` of the component :math:`\Phi^\prime_j` of :math:`\Psi^\prime`.
        :return: A complex valued matrix of shape :math:`|\mathfrak{K}_i| \times |\mathfrak{K}^\prime_j|`.
        """
        if not self._QR.get_dimension() == self._packet.get_dimension():
            raise ValueError("Quadrature dimension does not match the wavepacket dimension")

        M = self.do_nsd(row, col)
        # Handle NaNs if any
        M = nan_to_num(M)
        return M
