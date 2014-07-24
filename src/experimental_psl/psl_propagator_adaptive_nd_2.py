import string
from copy import copy as clone
from pylru import lrucache

from numpy import *
from scipy.linalg import pinv2
from matplotlib.pyplot import *

from WaveBlocksND import *
from WaveBlocksND.Plot import plotcf2d

# ========================================================================
# Model Parameters

dimension = 2

eps = 0.1

# Potential
potential = {}
potential["variables"] = ["x", "y"]
potential["potential"] = "x**2 + y**2"

# Basis shape of one Psi_j
Kbs = 2

latdistratio = 0.75

# Basis propagation
T = 0.05
dt = 0.005

# Endtime
Tend = 0.5

# Initial value parameters
q0 = array([[ 0.0],
            [ 0.2]])

p0 = array([[ 0.5],
            [ 0.0]])

# For the pseudoinverse of A
threshold_invert = 1e-6

# For pruning the active set J(t)
threshold_prune = 1e-2

# For the norm conservation tests
threshold_norm = 1e-6

# Maximal size of J(t)
Jsizemax = 5000

# Maximal number of iterations in adaptivity steps
maxiter = 10

# Cache size, choose >> C * max |J(t)|^2
cachesize_integrals = 2**20
cachesize_wavepackets = 2**12
cachesize_wavefunctions = 2**12

# Grid for packet evaluation
evaluate_packets = True
plot_packets = True
save_wavefunction = True

limits = [(-6.283185307179586, 6.283185307179586), (-6.283185307179586, 6.283185307179586)]
number_nodes = [1024, 1024]

# ========================================================================
# Potential setup

pot = {}
pot['potential'] = potential

BF = BlockFactory()
V = BF.create_potential(pot)
V.calculate_local_quadratic()
V.calculate_local_remainder()

# ------------------------------------------------------------------------
# Phase space grid

Id = eye(2*dimension, dtype=integer)
directions = vstack([Id, -Id])
latdist = latdistratio * eps * sqrt(pi)

def neighbours(S):
    N = set([])
    for s in S:
        ars = array(s)
        n = vsplit(ars+directions, 2*2*dimension)
        N.update(map(lambda x: tuple(squeeze(x)), n))
    return N.difference(S)

# ------------------------------------------------------------------------
# The parameters governing the semiclassical splitting

simparameters = {
    "leading_component" : 0,
    "T" : T,
    "dt" : dt,
    "matrix_exponential" : "pade",
    "splitting_method" : "Y4"
    }

TM0 = TimeManager(simparameters)
prop = SemiclassicalPropagator(simparameters, V)

# ------------------------------------------------------------------------
# Construct the families of wavepackets to be propagated semiclassically

QR = GaussHermiteQR(Kbs+4)
TPQR = TensorProductQR(dimension * [QR])
Q = DirectHomogeneousQuadrature(TPQR)
IPwpho = HomogeneousInnerProduct(Q)

WP0 = lrucache(cachesize_wavepackets)

def new_wavepacket(k):
    kq, kp = split(array(k), 2)
    q = kq * latdist
    p = kp * latdist

    HAWP = HagedornWavepacket(dimension, 1, eps)
    HAWP.set_parameters((q, p), key=('q','p'))
    HAWP.set_coefficient(0, tuple(dimension * [0]), 1.0)
    HAWP.set_innerproduct(IPwpho)

    WP0[k] = HAWP

WP1 = lrucache(cachesize_wavepackets)

def propagate_wavepacket(k):
    HAWP = WP0[k].clone()

    # Give a larger basis to the packets we propagate
    B = HyperbolicCutShape(dimension, Kbs)
    HAWP.set_basis_shapes([B])

    prop.set_wavepackets([(HAWP,0)])

    # Propagate all wavepackets
    nsteps = TM0.compute_number_timesteps()
    for i in xrange(1, nsteps+1):
        prop.propagate()

    HAWP = prop.get_wavepackets()[0]
    WP1[k] = HAWP


def get_wavepackets_0(J):
    wp0 = {}
    for j in J:
        if not j in WP0.keys():
            new_wavepacket(j)
        wp0[j] = WP0[j]
    return wp0


def get_wavepackets_1(J):
    wp1 = {}
    for j in J:
        if not j in WP1.keys():
            if not j in WP0.keys():
                new_wavepacket(j)
            propagate_wavepacket(j)
        wp1[j] = WP1[j]
    return wp1

# ------------------------------------------------------------------------
# IO
IOM = IOManager()
IOM.create_file()
IOM.create_group()
IOM.create_block()

# ------------------------------------------------------------------------
# Various quadratures needed

# Warning: NSD not suitable for potentials with exponential parts
QR = GaussHermiteOriginalQR(5)
TPQR = TensorProductQR(dimension * [QR])
Q = NSDInhomogeneous(TPQR)

IPwpih = InhomogeneousInnerProduct(Q)
IPlcih = InhomogeneousInnerProductLCWP(IPwpih)

Obs = ObservablesMixedHAWP(IPwpih)

IPcache_T = lrucache(cachesize_integrals)
IPcache_A = lrucache(cachesize_integrals)
IPcache_epot = lrucache(cachesize_integrals)
IPcache_ekin = lrucache(cachesize_integrals)

cache_A_fill = []
cache_T_fill = []

def matrix_a(Jt):
    # <LC0 | LC1>
    Jt = sorted(Jt)
    wp0 = get_wavepackets_0(Jt)
    nrow = len(Jt)
    ncol = len(Jt)
    A = zeros((nrow, ncol), dtype=complexfloating)
    # Manually build matrix entry by entry
    for kr, jr in enumerate(Jt):
        for kc, jc in enumerate(Jt):
            # Look up value in cache
            if not (jr,jc) in IPcache_A:
                # Compute it if not there
                Ijrjc = IPwpih.quadrature(wp0[jr], wp0[jc], summed=True)
                IPcache_A[(jr,jc)] = Ijrjc.copy()
            # Fill in matrix entry
            A[kr,kc] = IPcache_A[(jr,jc)].copy()

    cache_A_fill.append(len(IPcache_A))
    return A


def matrix_theta(J1, J0):
    # <LC0 | LC1>
    J0 = sorted(J0)
    J1 = sorted(J1)
    wp0 = get_wavepackets_0(J1)
    wp1 = get_wavepackets_1(J0)
    nrow = len(J1)
    ncol = len(J0)
    THETA = zeros((nrow, ncol), dtype=complexfloating)
    # Manually build matrix entry by entry
    for kr, jr in enumerate(J1):
        for kc, jc in enumerate(J0):
            # Look up value in cache
            if not (jr,jc) in IPcache_T:
                # Compute it if not there
                Ijrjc = IPwpih.quadrature(wp0[jr], wp1[jc], summed=True)
                IPcache_T[(jr,jc)] = Ijrjc.copy()
            # Fill in matrix entry
            THETA[kr,kc] = IPcache_T[(jr,jc)].copy()

    cache_T_fill.append(len(IPcache_T))
    return THETA


def matrix_epot(Jt):
    # <LC0 | V | LC1>
    Jt = sorted(Jt)
    wp0 = get_wavepackets_0(Jt)
    nrow = len(Jt)
    ncol = len(Jt)
    EPOT = zeros((nrow, ncol), dtype=complexfloating)
    # Manually build matrix entry by entry
    for kr, jr in enumerate(Jt):
        for kc, jc in enumerate(Jt):
            # Look up value in cache
            if not (jr,jc) in IPcache_epot:
                # Compute it if not there
                Ijrjc = Obs.potential_overlap_energy(wp0[jr], wp0[jc], V.evaluate_at, summed=True)
                IPcache_epot[(jr,jc)] = Ijrjc.copy()
            # Fill in matrix entry
            EPOT[kr,kc] = IPcache_epot[(jr,jc)].copy()

    return EPOT


def matrix_ekin(Jt):
    # <LC0 | T | LC1
    Jt = sorted(Jt)
    wp0 = get_wavepackets_0(Jt)
    nrow = len(Jt)
    ncol = len(Jt)
    EKIN = zeros((nrow, ncol), dtype=complexfloating)
    # Manually build matrix entry by entry
    for kr, jr in enumerate(Jt):
        for kc, jc in enumerate(Jt):
            # Look up value in cache
            if not (jr,jc) in IPcache_ekin:
                # Compute it if not there
                Ijrjc = Obs.kinetic_overlap_energy(wp0[jr], wp0[jc], summed=True)
                IPcache_ekin[(jr,jc)] = Ijrjc.copy()
            # Fill in matrix entry
            EKIN[kr,kc] = IPcache_ekin[(jr,jc)].copy()

    return EKIN

# ------------------------------------------------------------------------
# Evaluate the functions in the generating system

# Grid
G = TensorProductGrid(limits, number_nodes)
axx, axy = G.get_axes()

WF = lrucache(cachesize_wavefunctions)

def wavepackets_values(J, C):
    # Find unevaluated packets if any
    Jue = [j for j in J if not j in WF.keys()]

    # Evaluate all unevaluated packets
    wps = get_wavepackets_0(Jue)
    for k, wp in wps.iteritems():
        WF[k] = wp.evaluate_at(G, prefactor=True, component=0)

    # Compute final value of |Y>
    wf = zeros((1, G.get_number_nodes(overall=True)), dtype=complexfloating)
    for k, c in zip(J, C):
        wf += c * WF[k]

    return wf

# ------------------------------------------------------------------------
# Helper functions

def enlarge(J, steps=1):
    # Enlarge active set J(t)
    J = set(J)
    for step in xrange(steps):
        J = J.union(neighbours(J))
    return sorted(list(J))


def prune(J, c, threshold_prune=threshold_prune):
    # Prune negligible indices
    c = atleast_1d(c)
    indices = abs(c) > threshold_prune
    cpruned = c[indices]
    Jpruned = [ J[i] for i, v in enumerate(indices) if v == True ]
    return Jpruned, cpruned

# ------------------------------------------------------------------------
# Initial Value

def find_initial_J(Pi0):
    q0, p0 = Pi0
    q0i = list((q0 / latdist).round().astype(integer).reshape(dimension))
    p0i = list((p0 / latdist).round().astype(integer).reshape(dimension))
    return set([tuple(q0i + p0i)])


def initial_step(J0, IV, normiv):
    k = 0
    J0km1 = J0
    norm0km1 = 0.0

    while k <= maxiter and len(J0km1) <= Jsizemax:
        k += 1
        # Enlarge the current candidate set J(0)^(k)
        J0k = enlarge(J0km1)
        print("  Size of enlarged set J(0) is %d" % len(J0k))
        # Build linear combination Y(0)^(k) from J(0)^(k)
        wps = get_wavepackets_0(J0k)
        LC0 = LinearCombinationOfWPs(dimension, 1)
        for j in J0k:
            LC0.add_wavepacket(wps[j])
        # Project initial value to J(0)^(k)
        b0k = IPlcih.build_matrix(LC0, LCI)
        b0k = squeeze(b0k)
        # Threshold and prune J(0)^(k)
        J0k, b0k = prune(J0k, b0k)
        print("  Size of pruned set J(0) is %d" % len(J0k))
        # Backproject to obtain Psi(0)^(k)
        Ak = matrix_a(J0k)
        c0k = dot(pinv2(Ak, rcond=threshold_invert), b0k)
        # Compute norm of current candidate Psi(0)^(k)
        c0kbar = conjugate(transpose(c0k))
        norm0k = abs(dot(c0kbar, dot(Ak, c0k)))
        print("   Norm (try: %d): %f" % (k, norm0k))
        # Decide if the current solution Psi(0)^(k) is good enough
        normdiffk = abs(normiv - norm0k)
        print("   Norm difference: %f" % normdiffk)
        if normdiffk <= threshold_norm:
            print("   \033[1;32m=> success & break\033[1;m")
            break
        else:
            if abs(norm0km1 - norm0k) <= threshold_norm / 10.0:
                print("   \033[1;33mWARNING: Norm did not improve enough\033[1;m")
                print("   \033[1;33m=> give up & break\033[1;m")
                break
            else:
                J0km1 = J0k
                norm0km1 = norm0k
                print("   => not sufficient & retry")
    else:
        print("  \033[1;31mWARNING: Maximal adaptivity exhausted\033[1;m")

    Jt = J0k
    ct = c0k
    no = norm0k
    return Jt, ct, no


# Build packet from initial value
IVWP = HagedornWavepacket(dimension, 1, eps)
IVWP.set_parameters((q0, p0), key=('q','p'))
IVWP.set_coefficient(0, tuple(dimension * [0]), 1.0)

LCI = LinearCombinationOfWPs(dimension, 1)
LCI.add_wavepacket(IVWP)

# Compute norm of initial value
normiv = IVWP.norm(summed=True)

# Perform the initial step
print("Initial Step 0:")
J0 = find_initial_J([q0, p0])
Jt, ct, no = initial_step(J0, LCI, normiv)

# Observables
ctbar = conjugate(transpose(ct))
EPOT = matrix_epot(Jt)
EKIN = matrix_ekin(Jt)
ep = dot(ctbar, dot(EPOT, ct))
ek = dot(ctbar, dot(EKIN, ct))

# Trail
trail = set(Jt)

# Statistics
Jsize_hist = []
Jsize_hist.append(len(Jt))

# Output
IOM.add_norm({"ncomponents":1})
IOM.add_energy({"ncomponents":1})

IOM.save_norm(no, timestep=0)
IOM.save_energy([ek, ep], timestep=0)

# Evaluate wavepacket |Y>
if evaluate_packets:
    psi = wavepackets_values(Jt, ct)

if evaluate_packets and save_wavefunction:
    IOM.add_grid({"dimension":dimension, "number_nodes":number_nodes}, blockid="global")
    IOM.add_wavefunction({"ncomponents":1, "number_nodes":number_nodes})

    IOM.save_grid(G.get_nodes(flat=True), blockid="global")
    IOM.save_wavefunction([psi.reshape(G.get_number_nodes())], timestep=0)

# Plot frames
if evaluate_packets and plot_packets and dimension == 2:
    f = figure()
    ax = f.gca()
    plotcf2d(axx, axy, psi.reshape(G.get_number_nodes()), axes=ax, darken=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    f.savefig("frame_" + string.zfill(str(0),6)+".png")
    close(f)

# ------------------------------------------------------------------------
# Time propagation of the overall scheme

simparameters2 = {
    "T": Tend,
    "dt" : simparameters["T"],
    }

TM = TimeManager(simparameters2)
nsteps = TM.compute_number_timesteps()


def adaptive_step(Jt, ct, normt):
    # Copy input data and setup
    k = 0
    Jtkm1 = clone(Jt)
    normtkm1 = 0.0

    while k <= maxiter and len(Jtkm1) <= Jsizemax:
        k += 1
        # Enlarge the current candidate set J(t)^(k)
        Jtk = enlarge(Jtkm1)
        print("  Size of enlarged set J(t) is %d" % len(Jtk))
        # Make a theta-step
        THETAk = matrix_theta(Jtk, Jt)
        btk = dot(THETAk, ct)
        # Threshold and prune J(t)^(k)
        Jtk, btk = prune(Jtk, btk)
        print("  Size of pruned set J(t) is %d" % len(Jtk))
        # Backproject to obtain Psi(t)^(k)
        Ak = matrix_a(Jtk)
        ctk = dot(pinv2(Ak, rcond=threshold_invert), btk)
        # Compute norm of current candidate Psi(t)^(k)
        ctkbar = conjugate(transpose(ctk))
        normtk = abs(dot(ctkbar, dot(Ak, ctk)))
        print("   Norm (try: %d): %f" % (k, normtk))
        # Decide if the current solution Psi(t)^(k) is good enough
        normdiffk = abs(normt - normtk)
        print("   Norm difference: %f" % normdiffk)
        if normdiffk <= threshold_norm:
            print("   \033[1;32m=> success & break\033[1;m")
            break
        else:
            if abs(normtkm1 - normtk) <= threshold_norm / 10.0:
                print("   \033[1;33mWARNING: Norm did not improve enough\033[1;m")
                print("   \033[1;33m=> give up & break\033[1;m")
                break
            else:
                Jtkm1 = Jtk
                normtkm1 = normtk
                print("   => not sufficient & retry")
    else:
        print("  \033[1;31mWARNING: Maximal adaptivity exhausted\033[1;m")

    Jt = Jtk
    ct = ctk
    no = normtk
    return Jt, ct, no


# Perform the time iteration
for n in xrange(1, nsteps+1):
    print("Step %d:" % n)

    # Loop
    Jt, ct, no = adaptive_step(Jt, ct, no)

    # Observables
    EPOT = matrix_epot(Jt)
    EKIN = matrix_ekin(Jt)
    ctbar = conjugate(transpose(ct))
    ep = dot(ctbar, dot(EPOT, ct))
    ek = dot(ctbar, dot(EKIN, ct))

    IOM.save_norm(no, timestep=n)
    IOM.save_energy([ek, ep], timestep=n)

    # Trail
    trail = trail.union(Jt)

    # Statistics
    Jsize_hist.append(len(Jt))

    # Evaluate wavepacket |Y>
    if evaluate_packets:
        psi = wavepackets_values(Jt, ct)

    if evaluate_packets and save_wavefunction:
        IOM.save_wavefunction([psi.reshape(G.get_number_nodes())], timestep=n)

    # Plot frames
    if evaluate_packets and plot_packets and dimension == 2:
        f = figure()
        ax = f.gca()
        plotcf2d(axx, axy, psi.reshape(G.get_number_nodes()), axes=ax, darken=0.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        f.savefig("frame_" + string.zfill(str(n),6)+".png")
        close(f)


# ------------------------------------------------------------------------
# Output and plotting

print("-------------------------")
print("Cache Sizes:")
print(" Maximal size for wavepackets:   %d" % cachesize_wavepackets)
print(" Maximal size for wavefunctions: %d" % cachesize_wavefunctions)
print(" Maximal size for integrals:     %d" % cachesize_integrals)
print(" WP(0):  %d" % len(WP0))
print(" WP(dt): %d" % len(WP1))
print(" A:      %d" % len(IPcache_A))
print(" Theta:  %d" % len(IPcache_T))
print(" Ekin:   %d" % len(IPcache_ekin))
print(" Epot:   %d" % len(IPcache_epot))
print(" WF:     %d" % len(WF))
print("-------------------------")


# Plot statistics
figure()
plot(Jsize_hist)
grid(True)
xlabel(r"$t$")
ylabel(r"$|J(t)|$")
savefig("Jsize_hist.png")

figure()
plot(cache_A_fill, label=r"$A$")
plot(cache_T_fill, label=r"$\Theta$")
grid(True)
legend(loc="lower right")
xlabel(r"$t$")
ylabel(r"cache fill")
savefig("cache_fill.png")


# Plot Observables
timegrid = IOM.load_norm_timegrid()
time = TM.compute_time(timegrid)
norms = squeeze(IOM.load_norm())

figure()
plot(time, norms)
grid(True)
xlim(time[0], time[-1])
xlabel(r"$t$")
ylabel(r"$|\Psi(t)|$")
savefig("norms.png")

figure()
semilogy(time, abs(norms[0] - norms))
grid(True)
xlim(time[0], time[-1])
xlabel(r"$t$")
ylabel(r"$||\Psi(0)| - |\Psi(t)||$")
savefig("norms_drift_log.png")

timegridek, timegridep = IOM.load_energy_timegrid()
time = TM.compute_time(timegridek)
ekin, epot = map(squeeze, IOM.load_energy())

figure()
plot(time, epot)
grid(True)
xlim(time[0], time[-1])
xlabel(r"$t$")
ylabel(r"$E_p(t)$")
savefig("epot.png")

figure()
plot(time, ekin)
grid(True)
xlim(time[0], time[-1])
xlabel(r"$t$")
ylabel(r"$E_k(t)$")
savefig("ekin.png")

figure()
plot(time, ekin, label=r"$E_\mathrm{kin}$")
plot(time, epot, label=r"$E_\mathrm{pot}$")
plot(time, ekin+epot, label=r"$E_\mathrm{tot}$")
grid(True)
xlim(time[0], time[-1])
xlabel(r"$t$")
ylabel(r"$E(t)$")
legend(loc="upper right")
savefig("etot.png")

figure()
plot(time, (ekin+epot)[0] - (ekin+epot) )
grid(True)
xlim(time[0], time[-1])
xlabel(r"$t$")
ylabel(r"$E(0) - E(t)$")
savefig("etot_drift.png")

figure()
semilogy(time, abs((ekin+epot)[0] - (ekin+epot)) )
grid(True)
xlim(time[0], time[-1])
xlabel(r"$t$")
ylabel(r"$|E(0) - E(t)|$")
savefig("etot_drift_log.png")

IOM.finalize()
