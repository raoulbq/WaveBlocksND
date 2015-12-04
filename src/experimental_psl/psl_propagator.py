from numpy import *
from numpy.linalg import eig
from scipy.linalg import svd, pinv

import matplotlib.cm as cm
from matplotlib.pyplot import *
from matplotlib.patches import Circle, Ellipse

from WaveBlocksND import *
from WaveBlocksND.Plot import plotcf

# ========================================================================
# Model Parameters

eps = 0.1

# Maximal energy distance for pruning
delta = 4.0 * eps

# Potential
potential = {}
potential["variables"] = ["x"]
potential["potential"] = "x**4"

# Lattice spacing
latdistratio = 0.9 # 0.75

# Threshold for the pseudoinverse
threshold = 1e-6

# Define the size of one family
L = 7

# Maximal lattice extension
qm = 2.5
pm = 2.5

# Basis propagation
T = 0.05
dt = 0.005

# Simulation endtime
Tend = 2.0 * 4.4

# Initial value
q0 = 0.0
p0 = 1.0
Q0 = 1.0
P0 = 1.0j

# ========================================================================
# Potential setup
# ------------------------------------------------------------------------

pot = {}
pot['potential'] = potential

BF = BlockFactory()
V = BF.create_potential(pot)

# ========================================================================
# IO
IOM = IOManager()
IOM.create_file()
IOM.create_group()
IOM.create_block()

# ------------------------------------------------------------------------
# Initial Value

IVWP = HagedornWavepacket(1, 1, eps)
IVWP.set_parameters((q0, p0, Q0, P0), key=('q','p','Q','P'))
IVWP.set_coefficient(0, (0,), 1.0)

# ------------------------------------------------------------------------
# Phase space grid

QR = GaussHermiteQR(L + 4)
Q = DirectHomogeneousQuadrature(QR)
IPwpho = HomogeneousInnerProduct(Q)

Obs = ObservablesHAWP(IPwpho)
Ekin = Obs.kinetic_energy(IVWP, summed=True)
Epot = Obs.potential_energy(IVWP, V.evaluate_at, summed=True)
E0 = Ekin + Epot

latdist = latdistratio * eps * sqrt(pi)

Nq = int(qm/latdist)
qu = linspace(-qm, qm, 2*Nq+1)

Np = int(pm/latdist)
pu = linspace(-pm, pm, 2*Np+1)

x, y = ogrid[:2*Nq+1,:2*Np+1]

Z = V.evaluate_at(qu[x], entry=(0,0)).T + 0.5*pu[y]**2
qq, pp = where(abs(Z - E0) < delta)
qs = qu[qq].astype(floating)
ps = pu[pp].astype(floating)

fig = figure()
ax = fig.gca()
ax.plot(qs, ps, ".")
for q, p in zip(qs, ps):
    #circle = Circle((q,p), abs(eps*sqrt(1.0/1.0j)), color="b", fill=False)
    ellipse = Ellipse((q,p), 2*eps*abs(sqrt(1.0)), 2*eps*abs(sqrt(1.0j)), color="b", fill=False)
    ax.add_artist(ellipse)
#circle0 = Circle((q0,p0), abs(eps*sqrt(Q0/P0)), color="r", fill=False)
ellipse0 = Ellipse((q0,p0), 2*eps*abs(sqrt(Q0)), 2*eps*abs(sqrt(P0)), color="r", fill=False)
ax.add_artist(ellipse0)
grid(True)
ax.axis('equal')
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$p$")
ax.plot(qs, ps, 'bo')
ax.plot(q0, p0, 'ro')
savefig("phasespace.png")
close(fig)

# ------------------------------------------------------------------------
# The parameters governing the semiclassical splitting

simparameters = {
    "leading_component" : 0,
    "T" : T,
    "dt" : dt,
    "matrix_exponential" : "pade",
    "splitting_method" : "Y4"
    }

# ------------------------------------------------------------------------
# Construct the families of wavepackets to be propagated semiclassically

LC0 = LinearCombinationOfWPs(1, 1)

for q, p in zip(qs,ps):
    HAWP = HagedornWavepacket(1, 1, eps)
    HAWP.set_parameters((q, p), key=('q','p'))
    HAWP.set_coefficient(0, (0,), 1.0)
    HAWP.set_innerproduct(IPwpho)
    LC0.add_wavepacket(HAWP)

K = LC0.get_number_packets()
print("Number of wavepackets in the generating system: "+str(K))

LC1 = LC0.clone()

# Give a larger basis the the packets we propagate
B = HyperCubicShape([L])
for packet in LC1.get_wavepackets():
    packet.set_basis_shapes([B])

# Save the packets:
IOM.add_lincombwp(LC1.get_description())
IOM.save_lincombwp_description(LC1.get_description())
#IOM.save_lincombwp_wavepackets(LC1.get_wavepackets(), timestep=0)

# ------------------------------------------------------------------------
# Make the semiclassical propagation of each family
TM = TimeManager(simparameters)
nsteps = TM.compute_number_timesteps()

print("Propagate wavepackets")
prop = MagnusPropagator(simparameters, V)

# Add all wavepackets
for phi in LC1.get_wavepackets():
    prop.add_wavepacket((phi, 0))

# Propagate all wavepackets
for i in xrange(1, nsteps+1):
    prop.propagate()

#IOM.save_lincombwp_wavepackets(LC1.get_wavepackets(), timestep=1)

fig = figure()
ax = fig.gca()
for psi in LC1.get_wavepackets():
    qk, pk, Qk, Pk = map(squeeze, psi.get_parameters(key=("q","p","Q","P"), component=0))
    ax.plot(qk, pk, "ob")
    ellipse = Ellipse((real(qk),real(pk)), 2*eps*abs(sqrt(Qk)), 2*eps*abs(sqrt(Pk)), color="b", fill=False)
    ax.add_artist(ellipse)
grid(True)
ax.axis('equal')
ax.set_xlabel(r"$q$")
ax.set_ylabel(r"$p$")
savefig("phasespace_propagated.png")
close(fig)

# ------------------------------------------------------------------------
# Compute the overlaping matrix of the generating system
IOM.add_overlaplcwp({})

QR = GaussHermiteOriginalQR(5)
Q = NSDInhomogeneous(QR)
IPnsd = InhomogeneousInnerProduct(Q)

IPlincomb = HomogeneousInnerProductLCWP(IPnsd)

print("Compute overlap matrix A")

A = IPlincomb.build_matrix(LC0)

IOM.save_overlaplcwp([A], timestep=0, key=["ov"])

fig = figure()
matshow(real(A), cmap=cm.viridis)
colorbar()
xlabel(r'$\Re A$')
savefig("A_r.png")
close(fig)

fig = figure()
matshow(imag(A), cmap=cm.viridis)
colorbar()
xlabel(r'$\Im A$')
savefig("A_i.png")
close(fig)

fig = figure()
matshow(abs(A), cmap=cm.viridis)
colorbar()
xlabel(r'$|A|$')
savefig("A.png")
close(fig)

fig = figure()
matshow(log(abs(real(A))), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |\Re A|$')
savefig("A_r_log.png")
close(fig)

fig = figure()
matshow(log(abs(imag(A))), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |\Im A|$')
savefig("A_i_log.png")
close(fig)

fig = figure()
matshow(log(abs(A)), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |A|$')
savefig("A_log.png")
close(fig)

# ------------------------------------------------------------------------
# Prepare the pseudoinverse of the overlap matrix

# u, s, v = svd(A)
# r = 1.0 + where(s/s[0] > threshold)[0].max()
# print("Number of singular values above threshold: "+str(r))
# ur = u[:,:r]
# vr = v[:,:r]
# sr = s[:r]
# Apinv = dot(ur, dot(diag(1.0/sr), conjugate(transpose(ur))))

# fig = figure()
# plot(abs(s), ".")
# grid(True)
# xlim(0, K)
# xlabel(r"$i$")
# ylabel(r"$\sigma_i$")
# savefig("singvals.png")
# close(fig)

# fig = figure()
# semilogy(abs(s), ".")
# grid(True)
# xlim(0, K)
# xlabel(r"$i$")
# ylabel(r"$\sigma_i$")
# savefig("singvals_log.png")
# close(fig)

# or just
Apinv = pinv(A, rcond=threshold)

fig = figure()
matshow(real(Apinv), cmap=cm.viridis)
colorbar()
xlabel(r'$\Re A^{+}$')
savefig("Apinv_r.png")
close(fig)

fig = figure()
matshow(imag(Apinv), cmap=cm.viridis)
colorbar()
xlabel(r'$\Im A^{+}$')
savefig("Apinv_i.png")
close(fig)

fig = figure()
matshow(abs(Apinv), cmap=cm.viridis)
colorbar()
xlabel(r'$|A^{+}|$')
savefig("Apinv.png")
close(fig)

fig = figure()
matshow(log(abs(real(Apinv))), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |\Re A^{+}|$')
savefig("Apinv_r_log.png")
close(fig)

fig = figure()
matshow(log(abs(imag(Apinv))), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |\Im A^{+}|$')
savefig("Apinv_i_log.png")
close(fig)

fig = figure()
matshow(log(abs(Apinv)), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |A^{+}|$')
savefig("Apinv_log.png")
close(fig)

# ------------------------------------------------------------------------
# Project IV to generating system

LCI = LinearCombinationOfWPs(1, 1)
LCI.add_wavepacket(IVWP)

IPlincomb = InhomogeneousInnerProductLCWP(IPnsd)

print("Compute coefficients c")

b = IPlincomb.build_matrix(LC0, LCI)
b = squeeze(b)
c = dot(Apinv, b)

#IOM.save_lincombwp_coefficients(c, timestep=0)

fig = figure()
plot(real(b), label=r"$\Re b$")
plot(imag(b), label=r"$\Im b$")
plot(abs(b), label=r"$|b|$")
grid(True)
xlim(0, K-1)
legend(loc="upper right", framealpha=0.8)
xlabel(r"$k$")
ylabel(r"$b_k$")
savefig("b.png")
close(fig)

fig = figure()
plot(real(c), label=r"$\Re c$")
plot(imag(c), label=r"$\Im c$")
plot(abs(c), label=r"$|c|$")
grid(True)
xlim(0, K-1)
legend(loc="upper right", framealpha=0.8)
xlabel(r"$k$")
ylabel(r"$c_k$")
savefig("c.png")
close(fig)

# ------------------------------------------------------------------------
# Compute the overlap of the generating system with the propagated families

print("Compute correlation matrix theta")

THETA = IPlincomb.build_matrix(LC0, LC1)

# Maybe should save theta at a better point.
IOM.save_overlaplcwp([THETA], timestep=1, key=["ov"])

fig = figure()
matshow(real(THETA), cmap=cm.viridis)
colorbar()
xlabel(r'$\Re \Theta$')
savefig("theta_r.png")
close(fig)

fig = figure()
matshow(imag(THETA), cmap=cm.viridis)
colorbar()
xlabel(r'$\Im \Theta$')
savefig("theta_i.png")
close(fig)

fig = figure()
matshow(abs(THETA), cmap=cm.viridis)
colorbar()
xlabel(r'$|\Theta|$')
savefig("theta.png")
close(fig)

fig = figure()
matshow(log(abs(real(THETA))), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |\Re \Theta|$')
savefig("theta_r_log.png")
close(fig)

fig = figure()
matshow(log(abs(imag(THETA))), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |\Im \Theta|$')
savefig("theta_i_log.png")
close(fig)

fig = figure()
matshow(log(abs(THETA)), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |\Theta|$')
savefig("theta_log.png")
close(fig)

# ------------------------------------------------------------------------

ApinvTHETA = dot(Apinv, THETA)

fig = figure()
matshow(real(ApinvTHETA), cmap=cm.viridis)
colorbar()
xlabel(r'$\Re A^{+}\Theta$')
savefig("mpi_r.png")
close(fig)

fig = figure()
matshow(imag(ApinvTHETA), cmap=cm.viridis)
colorbar()
xlabel(r'$\Im A^{+}\Theta$')
savefig("mpi_i.png")
close(fig)

fig = figure()
matshow(abs(ApinvTHETA), cmap=cm.viridis)
colorbar()
xlabel(r'$|A^{+}\Theta|$')
savefig("mpi.png")
close(fig)

fig = figure()
matshow(log(abs(real(ApinvTHETA))), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |\Re A^{+}\Theta|$')
savefig("mpi_r_log.png")
close(fig)

fig = figure()
matshow(log(abs(imag(ApinvTHETA))), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |\Im A^{+}\Theta|$')
savefig("mpi_i_log.png")
close(fig)

fig = figure()
matshow(log(abs(ApinvTHETA)), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |A^{+}\Theta|$')
savefig("mpi_log.png")
close(fig)

ew, ev = eig(ApinvTHETA)
ewgood = ew[abs(abs(ew) - 1.0) < 1e-2]
ewbad = ew[abs(abs(ew) - 1.0) >= 1e-2]
ewabs = abs(ew)
ewabs.sort()

fig = figure(figsize=(6,6))
plot(real(ewbad), imag(ewbad), 'or')
plot(real(ewgood), imag(ewgood), 'ob')
grid(True)
xlim(-1.2, 1.2)
ylim(-1.2, 1.2)
xlabel(r"$\Re \lambda_k$")
ylabel(r"$\Im \lambda_k$")
savefig("ew_at.png")
close(fig)

fig = figure()
plot(ewabs[::-1], 'o')
xlim(0, K-1)
grid(True)
xlabel(r"$k$")
ylabel(r"$|\lambda_k|$")
savefig("ewabs_at.png")
close(fig)

# ------------------------------------------------------------------------
# Prepare for observables

print("Compute Epot matrix")

Obs = ObservablesLCWP(IPlincomb)

EPOT = Obs.potential_overlap_matrix(LC0, V.evaluate_at)

IOM.save_overlaplcwp([EPOT], timestep=0, key=["ovpot"])

fig = figure()
matshow(real(EPOT), cmap=cm.viridis)
colorbar()
xlabel(r'$\Re E_{{pot}}$')
savefig("epotm_r.png")
close(fig)

fig = figure()
matshow(imag(EPOT), cmap=cm.viridis)
colorbar()
xlabel(r'$\Im E_{{pot}}$')
savefig("epotm_i.png")
close(fig)

fig = figure()
matshow(abs(EPOT), cmap=cm.viridis)
colorbar()
xlabel(r'$|E_{{pot}}|$')
savefig("epotm.png")
close(fig)

fig = figure()
matshow(log(abs(real(EPOT))), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |\Re E_{{pot}}|$')
savefig("epotm_r_log.png")
close(fig)

fig = figure()
matshow(log(abs(imag(EPOT))), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |\Im E_{{pot}}|$')
savefig("epotm_i_log.png")
close(fig)

fig = figure()
matshow(log(abs(EPOT)), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |E_{{pot}}|$')
savefig("epotm_log.png")
close(fig)

print("Compute Ekin matrix")

EKIN = Obs.kinetic_overlap_matrix(LC0)

IOM.save_overlaplcwp([EKIN], timestep=0, key=["ovkin"])

figure()
matshow(real(EKIN), cmap=cm.viridis)
colorbar()
xlabel(r'$\Re E_{{kin}}$')
savefig("ekinm_r.png")
close(fig)

figure()
matshow(imag(EKIN), cmap=cm.viridis)
colorbar()
xlabel(r'$\Im E_{{kin}}$')
savefig("ekinm_i.png")
close(fig)

figure()
matshow(abs(EKIN), cmap=cm.viridis)
colorbar()
xlabel(r'$|E_{{kin}}|$')
savefig("ekinm.png")
close(fig)

figure()
matshow(log(abs(real(EKIN))), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |\Re E_{{kin}}|$')
savefig("ekinm_r_log.png")
close(fig)

figure()
matshow(log(abs(imag(EKIN))), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |\Im E_{{kin}}|$')
savefig("ekinm_i_log.png")
close(fig)

figure()
matshow(log(abs(EKIN)), cmap=cm.viridis)
colorbar()
xlabel(r'$\log |E_{{kin}}|$')
savefig("ekinm.png")
close(fig)

# ------------------------------------------------------------------------
# Time propagation of the overall scheme

simparameters2 = {
    "T": Tend,
    "dt" : simparameters["T"],
    }

TM = TimeManager(simparameters2)
nsteps = TM.compute_number_timesteps()

IOM.add_norm({"ncomponents":1})
IOM.add_energy({"ncomponents":1})

# Initial state values:
cbar = conjugate(transpose(c))
no = abs(dot(cbar, dot(A, c)))
ep = dot(cbar, dot(EPOT, c))
ek = 0.5 * dot(cbar, dot(EKIN, c))

IOM.save_norm(no, timestep=0)
IOM.save_energy([ek, ep], timestep=0)

# Go

for n in xrange(1, nsteps+1):
    print("Step: "+str(n))

    c = dot(ApinvTHETA, c)

    #IOM.save_lincombwp_coefficients(c, timestep=n)

    # Observables
    cbar = conjugate(transpose(c))
    no = abs( dot(cbar, dot(A, c)) )
    ep = dot(cbar, dot(EPOT, c))
    ek = 0.5 * dot(cbar, dot(EKIN, c))
    print(no)

    IOM.save_norm(no, timestep=n)
    IOM.save_energy([ek, ep], timestep=n)

timegrid = IOM.load_norm_timegrid()
time = array(map(TM.compute_time, timegrid))
norms = squeeze(IOM.load_norm())

fig = figure()
plot(time, norms)
grid(True)
xlim(time[0], time[-1])
xlabel(r"$t$")
ylabel(r"$|\Psi(t)|$")
savefig("norms.png")
close(fig)

fig = figure()
semilogy(time, abs(norms[0] - norms))
grid(True)
xlim(time[0], time[-1])
xlabel(r"$t$")
ylabel(r"$|\Psi(0)| - |\Psi(t)|$")
savefig("norms_drift_log.png")
close(fig)

timegridek, timegridep = IOM.load_energy_timegrid()
time = array(map(TM.compute_time, timegridek))
ekin, epot = map(squeeze, IOM.load_energy())

IOM.finalize()

fig = figure()
plot(time, epot)
grid(True)
xlim(time[0], time[-1])
xlabel(r"$t$")
ylabel(r"$E_{pot}(t)$")
savefig("epot.png")
close(fig)

fig = figure()
plot(time, ekin)
grid(True)
xlim(time[0], time[-1])
xlabel(r"$t$")
ylabel(r"$E_{kin}(t)$")
savefig("ekin.png")
close(fig)

fig = figure()
plot(time, ekin, label=r"$E_\mathrm{kin}$")
plot(time, epot, label=r"$E_\mathrm{pot}$")
plot(time, ekin+epot, label=r"$E_\mathrm{tot}$")
grid(True)
xlim(time[0], time[-1])
legend(loc="upper right", framealpha=0.8)
xlabel(r"$t$")
ylabel(r"$E(t)$")
savefig("etot.png")
close(fig)

fig = figure()
plot(time, (ekin+epot)[0] - (ekin+epot) )
grid(True)
xlim(time[0], time[-1])
xlabel(r"$t$")
ylabel(r"$E(0) - E(t)$")
savefig("etot_drift.png")
close(fig)

fig = figure()
semilogy(time, abs((ekin+epot)[0] - (ekin+epot)) )
grid(True)
xlim(time[0], time[-1])
xlabel(r"$t$")
ylabel(r"$|E(0) - E(t)|$")
savefig("etot_drift_log.png")
close(fig)
