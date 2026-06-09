# This is a script detailing how offline tutorials should work.

They should all be in jupyter notebook format. These tutorials, in contrast to the other tutorials in the directory currently, will be fully offline and not be interactive polyscope, but they will reuse a lot of the same structure and experiments from the current demos.

Instead it generates in windows inside the jupyter notebooks and pngs (using matplotlib module of simkit perhaps when encessary, adding extra vis things in utils.py when necessary). 



We will have multiple tutorials:


## 0. Finite Element Intuition

A foundational tutorial that motivates *why* we discretize at all, before any
deformation gradient / energy / solver machinery appears.

Start with a smooth, recognizable shape rendered with **no triangle edges**, and
ask what it means for it to deform: we are seeking a displacement field `u(x)` —
one arrow per material point — collected into a deformation map
`phi(x) = x + u(x)`. Draw a simple recognizable deformation (a gentle bend) and
the same deformation as a vector field on the rest shape. The shape is built
**analytically** (the parametric heart curve) and triangulated **without any
mesh file or `igl`** — scatter interior points, run a generic Delaunay
triangulation, and drop triangles whose centre falls outside the outline.

Then the catch: a computer can't store `u` at infinitely many points. The finite
element method triangulates the shape, stores `u` only at the vertices, and
reconstructs it everywhere else by barycentric interpolation. Show the three
linear (P1) basis functions of a triangle **flat, drawn as color** (bright `= 1`
at their vertex, dark `= 0` at the others), then animate interpolation twice as
a matched pair. First a *scalar*: each vertex holds a number and the moving
point's **color** is the blend `s(x) = sum_i phi_i(x) s_i`. Then a *vector*: each
vertex holds a displacement and the moving **arrow** is
`u(x) = sum_i phi_i(x) u_i` (the same blend, run once per component). In both, an
evaluation point sweeps the triangle and its value matches a vertex's value
exactly as it approaches that vertex.

Finally, deform the mesh (no solver — just store-at-vertices + interpolate) with
the triangle edges turned back **on**, revealing that the smooth teaser shape was
a triangle mesh the whole time, and that refining the mesh converges back to the
smooth limit. Close with a note that FEM is only one way to discretize geometry
(Gaussian splats, RKPM/meshless, point clouds, signed distance fields, …) and
that choosing the right discretization is an open research problem.


## 1. Deformation Gradient Intuition
A triangle living in a matplotlib grid.

We do several deformations of the triangle
1. Translations
2. Rotations
3. Scale
4. Shear

And show the resulting deformation gradient as text pasted into the picture. 

In particular I want them to note that deformation gradient ignores translations.


## 2. Elastic Energy Intuition

Similar to the previous, but the triangle is deforming in various ways, the same as the first, translations, rotations, scale and shear. 

I want to be generic to the elastic energy at first.

I want to show that elastic energy should be invariant to rotation and translations, and becomes larger with additional shearing and scaling. 
Make a plot for translation. 

Specifically I want to show that linear elastic energy increases with rotation, but that ARAP, neo hookean don't. 
Make a plot for rotation angle and energy for all. Also output an mp4 animation of the triangle.

Next I want to show how each of these energies vary with scales/shears.
Finally I want to show scaling. 
I want to scale the triangle, show how each of the energies change with scale.
Same with scale. 
Make a plot for both (scaling parameter and energy), as well as an animation. 

Then I want to show what happens when a vertex moves towards its opposite edge and creates a degenerate 0 volume tetrahedron. I want to show the elastic energies for all my materials, with hopefull appreciating that neo hookean increases to infinite, and arap just increases.
Make a plot for both (collapsing parameter and energy)


For all the matplotlib plots here, also make an animation of the plot being drawn as the scene is occurring. 

## 3. Elastostatics as a Minimization Problem

Now that we have an objective that measures how deformed a material is,
let's try out an elastostatic minimization problem. 

Specifically we are going to minimize the elastic energy with an extra soft pinning energy, that keeps left vertex pinned.

We will also have a handle pinning energy, but this one varies the position of the handle(the right vertex).

We will show as we move the right handle vertex, the middle vertex deforms differently with different elastic energies. 

We will do this first for a triangle, and then for a cantilever beam.

We minimize silently with Newton, but don't explain that yet. 

## 4. Numerical Solver for Solving Minimization Problem

Now we expose the numerical solver, comparing Gradient Descent and Newton's method. 

We pin all the left part of the beam. Use the middle-rightmost vertex as a handle.

Run Newton's Method/Gradient Descent until convergence for a target position of the handle moving down and to the right. 

Make a video/animation of the beam converging with both Newton's method and Gradient Descent.

Make a plot of the Energy/Gradient Norm and Newton Decrement as a function of iterations. 


## 5. Importance of Line Search. 

Show an example with fixed step size (no line search) with Newton's method. (Maybe handle moves to the right, with high poisson ratio)

Show that if we're not careful and pick a step size too large, then the solver can explode. backtracking line search ensures that the Newton step we did decreases the energy, instead of increasing it. 

To make the video/animation perceivable, don't automatically shift the handle, make it start from rest and move slowly right then come back. 

Show an animation without line search, then one with line search.


# Don't do the rest after but at a high level here they are

## 6. Time Integration Comparison

Forward euler/Backward Euler/BDF2

Time order convergence. Make plot showing the order convergence of forward euler/backward euler/bdf2.

Show video of how timestep affects perceived stiffness.

## 7. Contact Ball

An animation of a ball moving upwards, colliding with a block of static elastic material that is pinned at its top. 

Show the signed distance function to the ball. 

Show energy increasing with distance


## 8. Contact Plane

An elastodynamic block being dropped on a 2D plane. 

Plot all energies (elastic, kinetic, contact increasing )

## 9.  Time Complexity. 

Refine mesh and show time cost increasing.

Show how accuracy converges with increasing mesh resolution by comparing to an extremely fine mesh. 

Show that a super refined mesh isa ctually terrible to work with and iterate on.


## 24. Subspace Mixed FEM (MFEM)

Introduce the auxiliary-stretch trick from the [subspace Mixed FEM](https://www.dgp.toronto.edu/projects/subspace-mfem/) method. Build a reduced skinning-eigenmode subspace `B` (positions `x = B u + q`) plus a spectral cubature, then add a per-cubature symmetric-stretch auxiliary `a` with the constraint `S(F(u)) = a` so the elastic energy is evaluated element-locally instead of on the dense subspace deformation. Solve the constrained system with the flat `sqp_mfem` solver (which eliminates the Lagrange multiplier) and compare a subspace MFEM drop against a full Newton FEM solve in the same subspace — they should track each other while MFEM stays cheap per iteration. The interactive counterpart is demo 012.


## 25. Fast Complementary Dynamics

Explain Complementary Dynamics: an animator rig `J @ p` carries the low-frequency, art-directed motion and the simulation adds *complementary* secondary motion `B @ z` on top. The key idea (Benchekroun et al., "Fast Complementary Dynamics via Skinning Eigenmodes", SIGGRAPH 2023) is to build the simulation subspace `B` orthogonal to the rig via a momentum-leaking constraint, so the secondary motion never duplicates or fights the rig. Build the beam, rig, orthogonality constraint, rig-orthogonal subspace and cubature, then run a clustered local/global `block_coord` step while sweeping the rig left/right, contrasting rig-only `J @ p` against the full `J @ p + B @ z`. The interactive counterpart is demo 013.


## 26. Modal Muscles

Treat a small set of displacement modes `D` as **muscles**: activating one applies a clustered plastic stretch that passive ARAP elasticity and inertia respond to, producing whole-body motion. Build a 2D creature, its skinning-eigenmode subspace `B`, modal actuators `D` (linear modal analysis), and cubature, then step an actuated simulation inline with `block_coord` (passive + active clustered polar-SVD local steps, Cholesky global step) while sweeping a sinusoidal activation to produce a gait. A small optional CMA-ES cell shows how the activation parameters can be optimized for locomotion. The interactive counterpart is demo 014. 