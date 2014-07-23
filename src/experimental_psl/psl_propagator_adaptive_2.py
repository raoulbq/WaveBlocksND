import string
from copy import copy as clone
from pylru import lrucache

from numpy import *
from scipy.linalg import pinv2
from scipy.special import erf
from matplotlib.pyplot import *

from WaveBlocksND import *
from WaveBlocksND.Plot import plotcf

# ========================================================================
# Model Parameters

eps = 0.1

# Potential
potential = {}
potential["variables"] = ["x"]
potential["potential"] = "x**4"

# Basis shape of one Psi_j
L = 7 #13

latdistratio = 0.75

# Basis propagation
T = 0.05
dt = 0.005

# Endtime
Tend = 0.5 * 4.4

# Initial value parameters
q0 = 0.0
p0 = 1.0

# For the pseudoinverse of A
threshold_invert = 1e-6

# For pruning the active set J(t)
threshold_prune = 1e-6

# For the norm conservation tests
threshold_norm = 1e-8

# Maximal size of J(t)
Jsizemax = 1000

# Maximal number of iterations in adaptivity steps
maxiter = 5

# Cache size, choose >> C * max |J(t)|^2
cachesize_integrals = 2**18
cachesize_wavepackets = 2**12
cachesize_wavefunctions = 2**12

# Grid for packet evaluation
limits = [(-6.283185307179586, 6.283185307179586)]
number_nodes = [4096]

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

Id = eye(2, dtype=integer)
directions = vstack([Id, -Id])
latdist = latdistratio * eps * sqrt(pi)

def neighbours(S):
    N = set([])
    for s in S:
        ars = array(s)
        n = vsplit(ars+directions, 4)
        N.update(map(lambda x: tuple(squeeze(x)), n))
    return N.difference(S)


def plot_grid(J, filename="phasespace"):
    fig = figure()
    ax = fig.gca()
    for k0, k1 in J:
        q = k0 * latdist
        p = k1 * latdist
        ax.plot(q, p, ".b")
        circle = Circle((q,p), eps, color="b", fill=False)
        ax.add_artist(circle)
    circle0 = Circle((q0,p0), eps, color="r", fill=False)
    ax.add_artist(circle0)
    grid(True)
    ax.axis('equal')
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$p$")
    ax.plot(q0, p0, 'ro')
    savefig(filename+".png")

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

QR = GaussHermiteQR(L+4)
Q = DirectHomogeneousQuadrature(QR)
IPwpho = HomogeneousInnerProduct(Q)

WP0 = lrucache(cachesize_wavepackets)

def new_wavepacket(k):
    kq, kp = split(array(k), 2)
    q = kq * latdist
    p = kp * latdist

    HAWP = HagedornWavepacket(1, 1, eps)
    HAWP.set_parameters((q, p), key=('q','p'))
    HAWP.set_coefficient(0, (0,), 1.0)
    HAWP.set_innerproduct(IPwpho)

    WP0[k] = HAWP

WP1 = lrucache(cachesize_wavepackets)

def propagate_wavepacket(k):
    HAWP = WP0[k].clone()

    # Give a larger basis to the packets we propagate
    B = HyperCubicShape([L])
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
Q = NSDInhomogeneous(QR)

IPwpih = InhomogeneousInnerProduct(Q)
IPlcih = InhomogeneousInnerProductLCWP(IPwpih)

Obs = ObservablesMixedHAWP(IPwpih)

IPcache_T = lrucache(cachesize_integrals)
IPcache_A = lrucache(cachesize_integrals)
IPcache_epot = lrucache(cachesize_integrals)
IPcache_ekin = lrucache(cachesize_integrals)

cache_A_fill = []
cache_T_fill = []

def a_cut(Jt):
    # <LC0 | LC1>
    Jt = sorted(Jt)
    wp0 = get_wavepackets_0(Jt)
    nrow = len(Jt)
    ncol = len(Jt)
    Acut = zeros((nrow, ncol), dtype=complexfloating)
    # Manually build matrix entry by entry
    for kr, jr in enumerate(Jt):
        for kc, jc in enumerate(Jt):
            # Look up value in cache
            if not (jr,jc) in IPcache_A:
                # Compute it if not there
                Ijrjc = IPwpih.quadrature(wp0[jr], wp0[jc], summed=True)
                IPcache_A[(jr,jc)] = Ijrjc.copy()
            # Fill in matrix entry
            Acut[kr,kc] = IPcache_A[(jr,jc)].copy()

    cache_A_fill.append(len(IPcache_A))
    return Acut


def theta_cut(J1, J0):
    # <LC0 | LC1>
    J0 = sorted(J0)
    J1 = sorted(J1)
    wp0 = get_wavepackets_0(J1)
    wp1 = get_wavepackets_1(J0)
    nrow = len(J1)
    ncol = len(J0)
    THETAcut = zeros((nrow, ncol), dtype=complexfloating)
    # Manually build matrix entry by entry
    for kr, jr in enumerate(J1):
        for kc, jc in enumerate(J0):
            # Look up value in cache
            if not (jr,jc) in IPcache_T:
                # Compute it if not there
                Ijrjc = IPwpih.quadrature(wp0[jr], wp1[jc], summed=True)
                IPcache_T[(jr,jc)] = Ijrjc.copy()
            # Fill in matrix entry
            THETAcut[kr,kc] = IPcache_T[(jr,jc)].copy()

    cache_T_fill.append(len(IPcache_T))
    return THETAcut


def epot_cut(Jt):
    # <LC0 | V | LC1>
    Jt = sorted(Jt)
    wp0 = get_wavepackets_0(Jt)
    nrow = len(Jt)
    ncol = len(Jt)
    EPOTcut = zeros((nrow, ncol), dtype=complexfloating)
    # Manually build matrix entry by entry
    for kr, jr in enumerate(Jt):
        for kc, jc in enumerate(Jt):
            # Look up value in cache
            if not (jr,jc) in IPcache_epot:
                # Compute it if not there
                Ijrjc = Obs.potential_overlap_energy(wp0[jr], wp0[jc], V.evaluate_at, summed=True)
                IPcache_epot[(jr,jc)] = Ijrjc.copy()
            # Fill in matrix entry
            EPOTcut[kr,kc] = IPcache_epot[(jr,jc)].copy()

    return EPOTcut


def ekin_cut(Jt):
    # <LC0 | T | LC1
    Jt = sorted(Jt)
    wp0 = get_wavepackets_0(Jt)
    nrow = len(Jt)
    ncol = len(Jt)
    EKINcut = zeros((nrow, ncol), dtype=complexfloating)
    # Manually build matrix entry by entry
    for kr, jr in enumerate(Jt):
        for kc, jc in enumerate(Jt):
            # Look up value in cache
            if not (jr,jc) in IPcache_ekin:
                # Compute it if not there
                Ijrjc = Obs.kinetic_overlap_energy(wp0[jr], wp0[jc], summed=True)
                IPcache_ekin[(jr,jc)] = Ijrjc.copy()
            # Fill in matrix entry
            EKINcut[kr,kc] = IPcache_ekin[(jr,jc)].copy()

    return EKINcut

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

    return squeeze(wf)

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
    q0i = list((q0 / latdist).round().astype(integer).reshape(1))
    p0i = list((p0 / latdist).round().astype(integer).reshape(1))
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
        LC0 = LinearCombinationOfWPs(1, 1)
        for j in J0k:
            LC0.add_wavepacket(wps[j])
        # Project initial value to J(0)^(k)
        b0k = IPlcih.build_matrix(LC0, LCI)
        b0k = squeeze(b0k)
        # Threshold and prune J(0)^(k)
        J0k, b0k = prune(J0k, b0k)
        print("  Size of pruned set J(0) is %d" % len(J0k))
        # Backproject to obtain Psi(0)^(k)
        Acut = a_cut(J0k)
        c0k = dot(pinv2(Acut, rcond=threshold_invert), b0k)
        # Compute norm of current candidate Psi(0)^(k)
        c0kbar = conjugate(transpose(c0k))
        norm0k = abs(dot(c0kbar, dot(Acut, c0k)))
        print("   Norm (try: %d): %f" % (k, norm0k))
        # Decide if the current solution Psi(0)^(k) is good enough
        normdiffk = abs(normiv - norm0k)
        print("   Norm difference: %f" % normdiffk)
        if normdiffk <= threshold_norm:
            print("   => success & break")
            break
        else:
            if abs(norm0km1 - norm0k) <= threshold_norm / 10.0:
                print("   WARNING: Norm did not improve enough")
                print("   => give up & break")
                break
            else:
                J0km1 = J0k
                norm0km1 = norm0k
                print("   => not sufficient & retry")
    else:
        print("*** WARNING: Maximal adaptivity exhausted ***")

    Jt = J0k
    ct = c0k
    no = norm0k
    return Jt, ct, no

# Build packet from initial value
IVWP = HagedornWavepacket(1, 1, eps)
IVWP.set_parameters((q0, p0), key=('q','p'))
IVWP.set_coefficient(0, (0,), 1.0)

LCI = LinearCombinationOfWPs(1, 1)
LCI.add_wavepacket(IVWP)

# Compute norm of initial value
normiv = IVWP.norm(summed=True)

# Perform the initial step
print("Initial Step 0:")
J0 = find_initial_J([q0, p0])
Jt, ct, no = initial_step(J0, LCI, normiv)

# Observables
ctbar = conjugate(transpose(ct))
EPOTcut = epot_cut(Jt)
EKINcut = ekin_cut(Jt)
ep = dot(ctbar, dot(EPOTcut, ct))
ek = dot(ctbar, dot(EKINcut, ct))

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

# Plot coefficients
figure()
plot(real(ct))
plot(imag(ct))
plot(abs(ct))
grid(True)
xlabel(r"$k$")
ylabel(r"$c_k$")
savefig("c.png")

# Evaluate wavepacket |Y>
psi = wavepackets_values(Jt, ct)

IOM.add_wavefunction({"ncomponents":1, "number_nodes":number_nodes})
IOM.save_wavefunction([psi], timestep=0)

# Plot frames
f = figure()
ax = f.gca()
plotcf(squeeze(G.get_nodes()), angle(psi), abs(psi)**2, axes=ax)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(0, 10)
f.savefig("frame_" + string.zfill(str(0),6)+".png")
close(f)

# Plot phase space grid
fig = figure(figsize=(10,10))
for j, co in zip(Jt, ct):
    q = j[0] * latdist
    p = j[1] * latdist
    plot(q, p, ".b")
    circle1 = Circle((q,p), eps, color="b", fill=True, alpha=clip(erf(abs(co)), 0.0, 1.0))
    fig.gca().add_artist(circle1)
grid(True)
ax = fig.gca()
ax.axis('equal')
ax.set_xlim(-2.5,2.5)
ax.set_ylim(-2.5,2.5)
xlabel(r"$q$")
ylabel(r"$p$")
savefig("phasespace_" + string.zfill(str(0),6)+".png")
close(fig)

# ------------------------------------------------------------------------
# Time propagation of the overall scheme

simparameters2 = {
    "T": Tend,
    "dt" : simparameters["T"],
    }

TM = TimeManager(simparameters2)
nsteps = TM.compute_number_timesteps()


def adaptive_step(Jt, ct, last_no, max_tries=5, threshold_norm=threshold_norm):
    Jt_try = clone(Jt)
    ct_try = ct.copy()
    no_last_try = last_no

    # Adaptively enlarge J(t)
    for new_try in xrange(max_tries):
        # Enlarge Einzugsgebiete
        Jtn_try = enlarge(Jt_try)
        print("  Size of new test set J(t) is %d" % len(Jtn_try))

        # Make a theta-step
        THETAcut = theta_cut(Jtn_try, Jt)
        btn_try = dot(THETAcut, ct_try)

        # Prune irrelevant indices
        # TODO: Consider early pruning
        #ibn = abs(btn) > threshold_prune
        #btnc = btn[ibn]
        #Jtnc = [ Jtn[i] for i, v in enumerate(ibn) if v == True ]

        # Backproject to grid
        Acut = a_cut(Jtn_try)
        ctn_try = dot(pinv2(Acut, rcond=threshold_invert), btn_try)

        # Observables
        ctnbar_try = conjugate(transpose(ctn_try))
        no_try = abs(dot(ctnbar_try, dot(Acut, ctn_try)))
        print("  Norm (try: %d): %f" % (new_try, no_try))

        if abs(last_no - no_try) <= threshold_norm:
            print("   Norm difference: %f" % abs(last_no - no_try))
            print("  => sucess")
            Jtn = Jtn_try
            break
        else:
            if abs(no_last_try - no_try) <= threshold_norm:
                print("+++ WARNING: Norm did not improve enough +++")
                Jtn = Jtn_try
                break
            else:
                Jt_try = Jtn_try
                no_last_try = no_try
                print("  => retry")

    else:
        print("*** WARNING: Maximal number of tries exhausted ***")
        Jtn = Jtn_try

    # Make a theta-step
    THETAcut = theta_cut(Jtn, Jt)
    btn = dot(THETAcut, ct)

    # Prune irrelevant indices
    ibn = abs(btn) > threshold_prune
    btnc = btn[ibn]
    Jtnc = [ Jtn[i] for i, v in enumerate(ibn) if v == True ]

    # Backproject to grid
    Acut = a_cut(Jtnc)
    ctn = dot(pinv2(Acut, rcond=threshold_invert), btnc)

    # Observables
    ctnbar = conjugate(transpose(ctn))
    no = abs(dot(ctnbar, dot(Acut, ctn)))

    return Jtnc, ctn, no


# Go
for n in xrange(1, nsteps+1):
    print("Step %d:" % n)

    # Loop
    Jt, ct, no = adaptive_step(Jt, ct, no)

    # Observables
    print(" Norm: %f" % no)

    EPOTcut = epot_cut(Jt)
    EKINcut = ekin_cut(Jt)
    ctbar = conjugate(transpose(ct))
    ep = dot(ctbar, dot(EPOTcut, ct))
    ek = dot(ctbar, dot(EKINcut, ct))

    IOM.save_norm(no, timestep=n)
    IOM.save_energy([ek, ep], timestep=n)

    # Trail
    trail = trail.union(Jt)

    # Statistics
    Jsize_hist.append(len(Jt))
    print(" Size of pruned J(t) is %d" % len(Jt))

    # Evaluate wavepacket |Y>
    psi = wavepackets_values(Jt, ct)

    IOM.save_wavefunction([psi], timestep=n)

    # Plot frames
    f = figure()
    ax = f.gca()
    plotcf(squeeze(G.get_nodes()), angle(psi), abs(psi)**2, axes=ax)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0, 10)
    f.savefig("frame_" + string.zfill(str(n),6)+".png")
    close(f)

    # Plot phase space grid
    fig = figure(figsize=(10,10))
    for j, co in zip(Jt, ct):
        q = j[0] * latdist
        p = j[1] * latdist
        plot(q, p, ".b")
        circle1 = Circle((q,p), eps, color="b", fill=True, alpha=clip(erf(abs(co)), 0.0, 1.0))
        fig.gca().add_artist(circle1)
    grid(True)
    ax = fig.gca()
    ax.axis('equal')
    ax.set_xlim(-2.5,2.5)
    ax.set_ylim(-2.5,2.5)
    xlabel(r"$q$")
    ylabel(r"$p$")
    savefig("phasespace_" + string.zfill(str(n),6)+".png")
    close(fig)

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


# Trail
fig = figure(figsize=(10,10))
for j in trail:
    q = j[0] * latdist
    p = j[1] * latdist
    plot(q, p, "ob")
grid(True)
ax = fig.gca()
ax.axis('equal')
ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)
xlabel(r"$q$")
ylabel(r"$p$")
savefig("phasespace_trail.png")
close(fig)


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
