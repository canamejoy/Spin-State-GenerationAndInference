# Materials and Methods

<!--
papersmith contract: sections/01-materials-and-methods.md
  mode            transposition — the formulation already exists; this carries it
  extent          560–3950 words of prose · 2 to 5 subsections
  carries         borrowed machinery, the proposal, one summary diagram
  does NOT carry  any result, measured number, or comparison
  dataset slot    NOT owned here. The dataset belongs to the experimental setup,
                  so this section opens directly on the machinery and stays
                  purely formal, carrying no figure but the proposal's diagram.
-->

This section carries the machinery the simulator borrows and then the simulator
itself. Two subsections of existing theory — the lattice model and the
checkerboard Metropolis chain — are followed by the proposal, which is the
single-kernel realisation of that chain.

## 1. The lattice model

Let $\Lambda \subset \mathbb{Z}^3$ be the set of lattice sites of a cylindrical
nanodot,

$$
\Lambda = \{(x,y,z) \in \mathbb{Z}^3 : x^2 + y^2 \le R_d^2,\ 0 \le z < L\},
$$

where $R_d$ is the dot radius in lattice steps and $L$ the number of layers.
Writing $L_2 = \lfloor R_d \rfloor + 1$ and $N = 2L_2 + 1$, the sites are
addressed inside a box of $L \times N \times N$, of which $|\Lambda|$ lie inside
the disc. Because $N$ is odd by construction, the box extent carries no factor
of two — a fact the proposal uses in Section 3.

To each site $i \in \Lambda$ is attached a classical spin
$\mathbf{s}_i \in \mathbb{R}^3$ with $\lVert \mathbf{s}_i \rVert = S_0$; the
configuration is $\mathbf{s} = \{\mathbf{s}_i\}_{i \in \Lambda}$. Let
$\mathcal{N}(i)$ denote the first-neighbour shell of $i$ restricted to $\Lambda$,
and let $\hat{\mathbf{x}}, \hat{\mathbf{y}}, \hat{\mathbf{z}}$ be the lattice
directions. The in-plane directions wrap; $z$ does not, so the top and bottom
layers have no neighbour beyond them.

The energy has five terms:

$$
\mathcal{H}(\mathbf{s}) =
  -J \sum_{\langle i j \rangle} \mathbf{s}_i \cdot \mathbf{s}_j
  \;+\; \sum_{i} \mathcal{A}_i(\mathbf{s}_i)
  \;-\; D \sum_{i} \Bigl[ \mathbf{s}_i \times \mathbf{s}_{i+\hat{\mathbf{x}}}
         \Bigr]_{y}
       - \Bigl[ \mathbf{s}_i \times \mathbf{s}_{i+\hat{\mathbf{y}}} \Bigr]_{x}
  \;-\; \mathbf{H} \cdot \sum_i \mathbf{s}_i .
$$

The first term is the first-shell exchange, summed once per bond. The
Dzyaloshinskii–Moriya term is **interfacial**: it runs over in-plane bonds only,
with the sign shown. That restriction is a modelling commitment, not a
simplification — the all-axes form is a different physical system, and on a
common configuration the two give total energies that differ by tens of percent.

The anisotropy $\mathcal{A}_i$ is the cubic series written on direction cosines,
with a site-dependent first constant:

$$
\mathcal{A}_i(\mathbf{s}) = \frac{K^{\text{eff}}_i}{S_0^4}
  \bigl[(s_x s_y)^2 + (s_x s_z)^2 + (s_y s_z)^2\bigr]
  + \frac{K_2}{S_0^6} (s_x s_y s_z)^2 ,
\qquad
K^{\text{eff}}_i = \begin{cases}
  K_{\!S} & i \in \partial\Lambda \\
  K_1     & \text{otherwise},
\end{cases}
$$

where $\partial\Lambda$ is the set of sites with fewer than six in-disc first
neighbours. Surface anisotropy therefore enters as a **field over sites**, not
as a scalar, and that is why it is carried as an array of the same shape as one
spin component throughout.

## 2. The checkerboard Metropolis chain

Let $\beta = 1/k_B T$. A single-site Metropolis step at site $i$ draws a trial
spin $\mathbf{t}_i$ uniformly on the sphere of radius $S_0$, forms
$\Delta \mathbf{s}_i = \mathbf{t}_i - \mathbf{s}_i$, and accepts with
probability $\min\{1, e^{-\beta \Delta E_i}\}$, where $\Delta E_i$ is the energy
difference the single-site change produces. Because every term of $\mathcal{H}$
couples $i$ only to $\mathcal{N}(i)$ and to itself, $\Delta E_i$ reads

$$
\Delta E_i = -J\, \Delta \mathbf{s}_i \cdot
   \!\!\sum_{j \in \mathcal{N}(i)}\!\! \mathbf{s}_j
 + \mathcal{A}_i(\mathbf{t}_i) - \mathcal{A}_i(\mathbf{s}_i)
 - D\, \delta^{\text{DM}}_i - \mathbf{H} \cdot \Delta \mathbf{s}_i ,
$$

with $\delta^{\text{DM}}_i$ the in-plane cross-product difference of the DMI
term. The quantity that matters for what follows is the **neighbour sum**
$\mathbf{v}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{s}_j$: it is the only part of
$\Delta E_i$ that reaches outside site $i$.

Partition $\Lambda$ by the parity of the coordinate sum,

$$
\Lambda^{(c)} = \{ i \in \Lambda : (x_i + y_i + z_i) \bmod 2 = c \},
\qquad c \in \{0, 1\},
$$

so that $\Lambda = \Lambda^{(0)} \sqcup \Lambda^{(1)}$. With first neighbours
only, every $j \in \mathcal{N}(i)$ has parity opposite to $i$. Consequently
$\Delta E_i$ for $i \in \Lambda^{(c)}$ depends on no spin of $\Lambda^{(c)}$,
and the sites of one colour may be updated **simultaneously** without changing
the chain: the simultaneous update is an exact reordering of the sequential
sweep, not an approximation of it. One sweep is two half-steps, $c = 0$ then
$c = 1$.

This exactness holds only while the interaction graph is bipartite with respect
to that colouring. Second- and fourth-shell exchange connect sites of the same
parity and would break it, which is why those couplings are held at zero
throughout.

## 3. The proposal: a single fused kernel for the sweep

The chain of Section 2 is the object being realised; the proposal changes only
how one half-step is computed. In an array-programming realisation, the
neighbour sum $\mathbf{v}_i$ of equation (4) is obtained by materialising six
shifted copies of the configuration and adding them. The proposal replaces that
with one kernel that never materialises a shifted copy, and it is defined by
four components, each named for the property it delivers.

### 3.1 Contiguous neighbour access: the component-planar layout

Let $B$ be the number of independent replicas advanced together. The
configuration tensor may be laid out component-minor, with shape
$(B, L, N, N, 3)$, or **component-planar**, with shape $(B, 3, L, N, N)$. The
proposal takes the second. Under it, a run along $x$ at fixed
$(b, a, z, y)$ — replica, component, layer, row — is contiguous in memory with
unit stride, so a row of $N$ spin components of one Cartesian component is a
single contiguous segment. The transposition between layouts is performed once,
outside the sweep loop, and therefore does not enter the per-sweep cost.

This component is a precondition for the next one: a shifted read is contiguous
only if the unshifted read was.

### 3.2 A synchronisation-free halo: the shifted-load stencil

Let a thread block own a tile of $B_y$ rows by $B_x$ lanes of one layer. The
stencil requires, for each of the three components, the tile itself and its
four in-plane translates together with the two out-of-plane ones. The proposal
obtains each translate as a **vector-shifted contiguous read**: the index
expression is $x + 1$ applied to a contiguous run, not an address computed
independently per lane. The distinction is the whole of the component. A
per-lane address prevents the compiler from proving contiguity and yields
scattered accesses; a shifted run keeps the access pattern a shifted contiguous
segment — the same pattern the materialised copy would have produced — while the
copy itself never exists. The arithmetic of equation (4) is then carried out in
registers and only the accepted configuration is written.

The halo requires **no barrier**. By the parity property established after
equation (6), every neighbour read by a block belongs to the opposite colour and
is therefore not written by any block during that half-step. A block may read
its halo from global memory with no coordination whatever. The bipartite
colouring, introduced in Section 2 as a correctness device, is thus also what
makes the locality safe.

### 3.3 Counter-based proposals: in-kernel generation

The trial spin and the acceptance variate are generated **inside** the kernel
from a counter-based generator rather than supplied by the host. For site $i$ of
replica $b$ at step $n$, let
$\omega(b, i) = ((b L + z_i) N + y_i) N + x_i$ be the global site offset. The
proposal draws

$$
g_k = \mathcal{N}\!\bigl(n,\; 4\omega + k\bigr),\ k \in \{0,1,2\},
\qquad
u = \mathcal{U}\!\bigl(n, \; 4\omega + 3\bigr),
\qquad
\mathbf{t}_i = S_0\, \frac{(g_0, g_1, g_2)}{\lVert (g_0,g_1,g_2) \rVert},
$$

with $\mathcal{N}$ and $\mathcal{U}$ the normal and uniform streams of the
generator. The four draws share one key, the step index $n$, and are separated
by **distinct counters**. The separation is the component: drawing them from
adjacent *keys* at a shared counter gives $u$ an input in common with the
direction $\mathbf{t}_i$, so $u$ correlates with $\Delta E_i$ and the acceptance
test of Section 2 is biased. One key and distinct counters is the textbook use
of a counter-based generator and removes the shared input.

### 3.4 Deterministic observables: block-partial reduction

The observables of equation (9) below are accumulated as **per-block partial
sums** written to a small array and reduced afterwards, never by atomic
accumulation into a shared scalar. An atomic float reduction makes the result
depend on the order in which blocks retire, so two evaluations of the same
configuration need not agree in the last bits; a partial-sum reduction is
order-independent and therefore admits exact comparison against the reference
implementation. The partial array is of size $O(B L \lceil N/B_y \rceil)$ and
its final reduction is negligible beside the sweep.

The topological charge is the one observable whose kernel is partial by
construction: the Berg–Lüscher plaquette form requires a two-argument arctangent
whose branch matters, since its denominator may change sign on a frustrated
plaquette. The kernel emits the four arguments of the two arctangents and the
final transcendental is applied outside it.

### 3.5 The observables

Let $n_a = |\Lambda|$. For a configuration $\mathbf{s}$, the simulator reports
the energy per site, the magnetisation and its out-of-plane component, and the
topological charge of the mid-layer:

$$
E = \frac{1}{n_a}\,\mathcal{H}(\mathbf{s}), \qquad
\mathbf{M} = \frac{1}{n_a} \sum_{i \in \Lambda} \mathbf{s}_i, \qquad
Q = \frac{1}{4\pi} \sum_{p} \Omega_p ,
$$

where the energy is evaluated with each bond counted once, using only the
$+\hat{\mathbf{x}}$, $+\hat{\mathbf{y}}$ and $+\hat{\mathbf{z}}$ translates, and
$\Omega_p$ is the signed solid angle of plaquette $p$,

$$
\Omega(\mathbf{a}, \mathbf{b}, \mathbf{c}) = 2 \arctan
  \frac{\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})}
       {1 + \mathbf{a}\!\cdot\!\mathbf{b} + \mathbf{b}\!\cdot\!\mathbf{c}
          + \mathbf{c}\!\cdot\!\mathbf{a}} ,
$$

each plaquette contributing $\Omega(\mathbf{s}_{00},\mathbf{s}_{10},
\mathbf{s}_{11}) + \Omega(\mathbf{s}_{00},\mathbf{s}_{11},\mathbf{s}_{01})$ over
the plaquettes whose four corners lie in the disc. This form is quantised,
unlike a finite-difference curl.

The response functions are variances,

$$
C_v = \frac{\langle (\Delta E)^2 \rangle}{k_B T^2 n_a},
\qquad
\chi = \frac{\langle (\Delta \lVert\mathbf{M}\rVert)^2 \rangle}{k_B T},
$$

and they are accumulated about a reference taken once per temperature rather
than from raw moments. In single precision the raw form is not merely imprecise
but qualitatively wrong: at low temperature one unit in the last place of
$\langle E^2 \rangle$ exceeds the variance itself, and $\langle E^2 \rangle -
\langle E \rangle^2$ comes out negative. A constant shift leaves a variance
unchanged, so accumulating the fluctuation about a fixed reference is exact and
keeps every accumulated quantity at the scale of the fluctuation instead of the
total.

### 3.6 The combination

One sweep applies the kernel of Sections 3.1–3.3 twice, once per colour; one
temperature applies the sweep $n_{\text{th}}$ times to thermalise and
$n_{\text{me}}$ times while accumulating the observables of Section 3.4; and one
simulation scans a decreasing temperature ladder $\{T_m\}_{m=1}^{M}$, carrying
the final configuration of each temperature into the next:

$$
\mathbf{s}^{(m)} = \Bigl(\mathcal{K}^{(1)}_{\beta_m} \circ
  \mathcal{K}^{(0)}_{\beta_m}\Bigr)^{\,n_{\text{th}} + n_{\text{me}}}
  \bigl(\mathbf{s}^{(m-1)}\bigr),
\qquad \beta_m = 1/k_B T_m ,
$$

with $\mathcal{K}^{(c)}_\beta$ the half-step kernel for colour $c$ at inverse
temperature $\beta$, and the whole composition evaluated for the $B$ replicas at
once. Every symbol in equation (12) traces to a declaration: $\mathcal{K}^{(c)}$
to Section 3.2, the colouring $c$ to equation (6), $\beta$ to Section 2, and $B$
to Section 3.1.

Figure 1 summarises the four components in the order this section defines them,
showing for one half-step how the layout, the halo, the generator and the
reduction compose into the kernel.

<!-- FIGURE 1 — mandatory summary diagram, the visual form of the contribution
     list. Same four components, same order, same names:
       (1) Contiguous neighbour access — the component-planar layout
       (2) A synchronisation-free halo — the shifted-load stencil
       (3) Counter-based proposals — in-kernel generation
       (4) Deterministic observables — block-partial reduction
     Caption enumerates them in that order and decodes the colour coding.
     Carries NO dataset box and NO baseline box: those belong to the
     experimental setup's own diagram.
     STATUS: to be drawn. -->

**Figure 1.** The fused half-step kernel. (1) *Contiguous neighbour access*: the
configuration is held component-planar, so a row of one Cartesian component is a
contiguous segment (blue). (2) *A synchronisation-free halo*: each block reads
its tile and the one-site border through shifted contiguous reads (green
arrows), with no barrier, because the halo is of the opposite colour. (3)
*Counter-based proposals*: the trial direction and the acceptance variate are
drawn in-kernel from one key and four distinct counters (yellow). (4)
*Deterministic observables*: per-block partial sums are written to a small array
and reduced outside the kernel (grey). Colour marks the component a box belongs
to; the dashed boundary is the kernel.
