---
title: "Interval Arithmetic"
---

Interval arithmetic provides guaranteed bounds on computations. Instead of a
single value, you track an interval $ [a, b] $ that is guaranteed to contain
the true value.

## The Interval Type

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
use numra::Interval;

// A value known to lie between 2.0 and 3.0
let x = Interval::<f64>::new(2.0, 3.0);

// An exact value
let exact = Interval::point(5.0);

// Create from center and half-width
let measured = Interval::from_center(10.0, 0.1);  // [9.9, 10.1]
```

## Basic Properties

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let x = Interval::<f64>::new(2.0, 5.0);

println!("Width:  {}", x.width());    // 3.0
println!("Center: {}", x.center());   // 3.5
println!("Contains 3.0: {}", x.contains(3.0));  // true
println!("Contains 6.0: {}", x.contains(6.0));  // false
```

## Arithmetic

Interval arithmetic computes the tightest interval containing all possible
results:

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let a = Interval::<f64>::new(1.0, 2.0);
let b = Interval::<f64>::new(3.0, 4.0);

// [1,2] + [3,4] = [4, 6]
let sum = a.add(&b);

// [1,2] - [3,4] = [1-4, 2-3] = [-3, -1]
let diff = a.sub(&b);

// Multiplication considers all endpoint combinations
let c = Interval::new(-1.0, 2.0);
let d = Interval::new(1.0, 3.0);
// Products: -1*1=-1, -1*3=-3, 2*1=2, 2*3=6 → [-3, 6]
let prod = c.mul(&d);

// Scaling
let scaled = a.scale(3.0);   // [3, 6]
let neg = a.scale(-2.0);     // [-4, -2]  (bounds swap!)
```

## Set Operations

<!-- book-ignore: illustrative excerpt; not a standalone crate entry point. -->
```rust
let a = Interval::<f64>::new(1.0, 4.0);
let b = Interval::<f64>::new(3.0, 6.0);

// Union: [1, 6]
let u = a.union(&b);

// Intersection: [3, 4]
let i = a.intersection(&b);  // Some([3, 4])

// Disjoint intervals
let c = Interval::new(5.0, 6.0);
let none = a.intersection(&c);  // None
```

## Use Cases

| Application | How intervals help |
|-------------|-------------------|
| **Validated numerics** | Prove a solution exists within bounds |
| **Root bracketing** | Guarantee an interval contains a zero |
| **Range analysis** | Find guaranteed min/max of a function |
| **Error bounds** | Bound the effect of measurement uncertainty |
| **Robust control** | Ensure stability for all parameter values in range |

## Interval vs Uncertain

| Feature | `Interval` | `Uncertain` |
|---------|-----------|-------------|
| **Represents** | Hard bounds $[a, b]$ | Mean + variance |
| **Guarantee** | True value is inside | Statistical confidence |
| **Operations** | Always widen (conservative) | Approximate (linearized) |
| **Best for** | Worst-case analysis | Typical-case analysis |

Use `Interval` when you need guaranteed bounds (safety-critical, verification).
Use `Uncertain` when you want statistical uncertainty quantification.

## Limitations

Interval arithmetic suffers from the **dependency problem**: repeated use of
the same variable in an expression is treated as independent, leading to
over-estimation. For example:

$$
x - x = [a, b] - [a, b] = [a - b, b - a] \neq [0, 0]
$$

The true result is always 0, but interval arithmetic returns a wider interval.
This over-conservatism grows with the number of operations, which is why
interval methods are best for short computation chains.
