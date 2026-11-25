# The mathematics of interception between different planes

For burn time `t_b` and arrival time `t_i`, propagate both initial Cartesian states in three dimensions:

```text
(r_B, v_B) at t_b
(r_A, v_A) at t_i
```

Solve Lambert directly between the spatial endpoints:

```text
r_B(t_b) --Lambert(dt = t_i - t_b)--> r_A(t_i)
```

The required inertial impulse is:

```text
Delta-v_I = v_Lambert(t_b) - v_B(t_b)
```

No coplanar reduction is made. The endpoint vectors define the transfer plane except at collinear geometry. For exact or near 0/180-degree geometry, OrbitWin uses a Cartesian shooting fallback with an explicit reference orbit normal.

## GMAT local frames

For VNB:

```text
V = v / |v|
N = (r x v) / |r x v|
B = V x N
C_VNB = [V N B]
components = C_VNB^T Delta-v_I
```

The second VNB component is normal to the current orbit and naturally carries an out-of-plane maneuver. LVLH and inertial `EarthMJ2000Eq` are also implemented.

## Multi-revolution branches

The universal-variable Lambert time-of-flight function is scanned between Stumpff singularities. All numeric roots are reconstructed and independently propagated; a branch is retained only when its arrival-position error is below the configured threshold.

## Why no arrival velocity match

The preliminary rules require position entry into a 5 km sphere and explicitly impose no relative-velocity constraint. Therefore a circularization or rendezvous burn after first entry is normally wasteful unless the formal scoring/validator says otherwise.
