from pylru import lrucache

from numpy import *
from scipy.linalg import pinv2
from matplotlib.pyplot import *

from WaveBlocksND import *

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

# For the pseudoinverse
threshold_invert = 1e-6

# For pruning the active set
threshold_prune = 1e-2

# Cache size, choose >> C * max |J(t)|^2
cachesize_integrals = 2**20
cachesize_wavepackets = 2**12
cachesize_wavefunctions = 2**12

# Grid for packet evaluation
evaluate = False
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


def superset(J):
    J = set(J)
    Jn = neighbours(J)
    return sorted(list(J.union(Jn)))

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

    # Give a larger basis to the propagated packets
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

WF = lrucache(cachesize_wavefunctions)

def wavepackets_values(J, C):
    # Find unevaluated packets if any
    Jue = [j for j in J if not j in WF.keys()]

    # Evaluate all unevaluated packets
    wps = get_wavepackets_0(Jue)
    for k, wp in wps.iteritems():
        WF[k] = wp.evaluate_at(G, prefactor=True, component=0)

    # Compute final value of |Y>
    wf = zeros_like(G.get_nodes(), dtype=complexfloating)
    for k, c in zip(J, C):
        wf += c * WF[k]

    return wf

# ------------------------------------------------------------------------
# Initial Value

def find_initial_J(Pi0, distance=2):
    q0, p0 = Pi0
    q0i = list((q0 / latdist).round().astype(integer).reshape(dimension))
    p0i = list((p0 / latdist).round().astype(integer).reshape(dimension))
    J0 = set([tuple(q0i + p0i)])
    for i in xrange(distance):
        J0 = superset(J0)
    return sorted(list(J0))


# Build packet from initial value
IVWP = HagedornWavepacket(dimension, 1, eps)
IVWP.set_parameters((q0, p0), key=('q','p'))
IVWP.set_coefficient(0, tuple(dimension * [0]), 1.0)

LCI = LinearCombinationOfWPs(dimension, 1)
LCI.add_wavepacket(IVWP)

# Project to phasespace grid
J0 = find_initial_J([q0, p0])
wps = get_wavepackets_0(J0)
LC0 = LinearCombinationOfWPs(dimension, 1)
for j in J0:
    LC0.add_wavepacket(wps[j])

b = IPlcih.build_matrix(LC0, LCI)
b = squeeze(b)

# Prune irrelevant indices
ib = abs(b) > threshold_prune
b = b[ib]
J0c = [ J0[i] for i, v in enumerate(ib) if v == True ]

# Backproject to grid
A = matrix_a(J0c)

ct = dot(pinv2(A, rcond=threshold_invert), b)
Jt = J0c

# Observables
ctbar = conjugate(transpose(ct))
no = abs(dot(ctbar, dot(A, ct)))
print(" Norm: %f" % no)

EPOT = matrix_epot(Jt)
EKIN = matrix_ekin(Jt)
ep = dot(ctbar, dot(EPOT, ct))
ek = dot(ctbar, dot(EKIN, ct))

# Trail
trail = set(Jt)

# Statistics
Jsize_hist = []
Jsize_hist.append(len(Jt))
print(" Size of pruned J(t) is %d" % len(Jt))

# Output
IOM.add_norm({"ncomponents":1})
IOM.add_energy({"ncomponents":1})

IOM.save_norm(no, timestep=0)
IOM.save_energy([ek, ep], timestep=0)

# Evaluate wavepacket |Y>
#psi = wavepackets_values(Jt, ct)

#IOM.add_wavefunction({"ncomponents":1, "number_nodes":number_nodes})
#IOM.save_wavefunction([psi], timestep=0)

# ------------------------------------------------------------------------
# Time propagation of the overall scheme

simparameters2 = {
    "T": Tend,
    "dt" : simparameters["T"],
    }

TM = TimeManager(simparameters2)
nsteps = TM.compute_number_timesteps()

# Go
for n in xrange(1, nsteps+1):
    print("Step: "+str(n))

    # Enlarge Einzugsgebiete
    Jtn = superset(Jt)
    print(" Size of unpruned J(t) is %d" % len(Jtn))

    # Make a theta-step
    THETA = matrix_theta(Jtn, Jt)
    btn = dot(THETA, ct)

    # Prune irrelevant indices
    ibn = abs(btn) > threshold_prune
    btnc = btn[ibn]
    Jtnc = [ Jtn[i] for i, v in enumerate(ibn) if v == True ]

    # Backproject to grid
    A = matrix_a(Jtnc)
    ctn = dot(pinv2(A, rcond=threshold_invert), btnc)

    # Loop
    ct = ctn
    Jt = Jtnc

    # Observables
    ctbar = conjugate(transpose(ct))
    no = abs(dot(ctbar, dot(A, ct)))
    print(" Norm: %f" % no)

    EPOT = matrix_epot(Jtnc)
    EKIN = matrix_ekin(Jtnc)
    ep = dot(ctbar, dot(EPOT, ct))
    ek = dot(ctbar, dot(EKIN, ct))

    IOM.save_norm(no, timestep=n)
    IOM.save_energy([ek, ep], timestep=n)

    # Trail
    trail = trail.union(Jt)

    # Statistics
    Jsize_hist.append(len(Jt))
    print(" Size of pruned J(t) is %d" % len(Jt))

    # Evaluate wavepacket |Y>
    #psi = wavepackets_values(Jt, ct)

    #IOM.save_wavefunction([psi], timestep=n)

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
