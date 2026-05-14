# Getting started with Numra on Windows

Numra is a Rust library, so the toolchain (`rustup` + `cargo`) is the only prerequisite. No separate "Numra installer" — Cargo fetches and compiles it for you.

## 1. Install Rust

1. Go to <https://rustup.rs> and download **`rustup-init.exe`** (64-bit).
2. Run it. When it asks, accept the **default installation** (option 1).
3. On Windows, rustup will check for the **Microsoft C++ Build Tools** (needed because Rust uses MSVC's linker by default). If they're missing, it opens the Visual Studio Installer — choose **"Desktop development with C++"** and let it install. ~3–5 GB, one-time.
4. Close and reopen your terminal (PowerShell or Windows Terminal) so `PATH` picks up.

Verify:

```powershell
rustc --version
cargo --version
```

You should see something like `rustc 1.83.0` or newer. Numra requires **MSRV 1.83**.

## 2. Create a project

```powershell
cargo new numra-hello
cd numra-hello
```

This makes `Cargo.toml` (your project's manifest) and `src/main.rs`.

## 3. Add Numra as a dependency

Edit `Cargo.toml` and add under `[dependencies]`:

```toml
[dependencies]
numra = "0.1"
```

Or, equivalently, from the project folder:

```powershell
cargo add numra
```

If you want a specific feature set (e.g. only ODE solvers), see the per-crate options in the [Numra book](https://book.numra-rs.org/). For a first try, the default works.

## 4. Write your first program

Replace `src/main.rs` with the Quick Start from the README:

```rust
use numra::ode::{DoPri5, OdeProblem, Solver, SolverOptions};

fn main() {
    let problem = OdeProblem::new(
        |_t, y: &[f64], dydt: &mut [f64]| {
            dydt[0] = -y[0]; // dy/dt = -y
        },
        0.0,         // t0
        2.0,         // t_end
        vec![1.0],   // y0
    );

    let opts = SolverOptions::default().rtol(1e-8);
    let result = DoPri5::solve(&problem, 0.0, 2.0, &[1.0], &opts)
        .expect("solve failed");

    let y_end = result.y_final().expect("trajectory")[0];
    println!("y(2) ≈ {y_end:.6}  (exact: {:.6})", (-2.0_f64).exp());
}
```

## 5. Build and run

```powershell
cargo run --release
```

The first build downloads + compiles Numra and its dependencies — expect **5–15 minutes** on the first run. After that, incremental rebuilds are seconds. Use `--release` for real numerical work; debug builds are much slower for floating-point code.

You should see:

```
y(2) ≈ 0.135335  (exact: 0.135335)
```

## 6. Where to go next

- **Book / tutorials:** <https://book.numra-rs.org/>
- **API reference:** <https://docs.rs/numra>
- **Examples:** clone the repo and run any of them — e.g.

  ```powershell
  git clone https://github.com/moussaoutlook/numra-rs
  cd numra-rs
  cargo run --release -p numra --example lorenz
  ```

  Other examples: `van_der_pol`, `solver_zoo`, `heat_equation`, `gbm_monte_carlo` (full list in `numra/Cargo.toml`).

## Common Windows gotchas

- **"linker `link.exe` not found"** → MSVC Build Tools weren't installed. Re-run rustup or open the Visual Studio Installer and add "Desktop development with C++".
- **Long compile times** → normal on the first build; subsequent builds use the cache in `target/`. Don't delete it.
- **Antivirus slowdowns** → adding the project folder and `%USERPROFILE%\.cargo` to Defender's exclusion list helps noticeably.
- **OneDrive-synced project folder** → builds get slow and sometimes lock files. Put projects outside OneDrive (e.g. `C:\dev\`).
