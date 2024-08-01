//! Quick-and-dirty plotting in Rust using Python and [Matplotlib][matplotlib],
//! strongly inspired by the Haskell package [`matplotlib`][matplotlib-hs].
//!
//! ## Purpose
//! Both this crate and `matplotlib` internally use an existing Matplotlib
//! installation by generating a temporary Python source file, and simply
//! calling the system's Python interpreter. This approach affords a number of
//! advantages. The most significant is to use more familiar/convenient
//! construct to separate the logic and data surrounding plotting commands from
//! the canvases on which the data is eventually draw, leading to more modular
//! code overall. `matplotlib` provides an elegant model to monoidally compose
//! plotting commands, and this crate attempts to emulate it.
//!
//! However, neither this crate nor `matplotlib` are *safe* libraries. In
//! particular, both allow for the injection of arbitrary Python code from bare
//! string data. This allows for much flexibility, but of course makes a large
//! class of operations opaque to the compiler. Users are therefore warned
//! *against* using this crate in complex programs. Instead, this library
//! targets small programs that only need to quickly generate a plot.
//!
//! You **should** use this library if you:
//! - want an easy way to put some data in a nice-looking plot
//! - like and/or are familiar with Matplotlib, but don't want to use Python
//!   directly
//!
//! You **should not** use this library if you:
//! - want assurances against invalid Python code output
//! - want robust handling of errors generated by Python
//!
//! You may also be interested in:
//! - [Plotpy][plotpy], a Rust library with a similar strategy and safer
//!   constructs, but more verbose building patterns.
//! - [Plotters][plotters], a pure-Rust plotting library with full control
//!   over everything that goes on a figure.
//!
//! ## How it works
//! The main two components of the library are the [`Mpl`] type, representing a
//! plotting script, and the [`Matplotlib`] trait, representing an element of
//! the script. A given `Mpl` object can be combined with any number of objects
//! whose types implement `Matplotlib`, which allows for significant flexibility
//! when it comes to library users defining their own plotting elements. When
//! ready to be executed, the `Mpl` object's `run` method can be called to save
//! the output of the script to a file, launch Matplotlib's interactive Qt
//! interface, or both. The operations described above have also been overloaded
//! onto Rust's `&` and `|` operators to mimic `matplotlib`'s
//! API just for the fun of it.
//!
//! When `Mpl::run` is executed, any larger data structures associated with
//! plotting commands (mostly numerical arrays) are serialized to JSON. This
//! data, along with the plotting script itself, are written to the OS's default
//! temp directory (e.g. `/tmp` on Linux), and then the system's default
//! `python3` interpreter is called on the script using
//! [`std::process::Command`], which blocks the calling thread. Obviously, an
//! existing installation of Python 3 and Matplotlib are required. After the
//! script exits, both the script and the JSON file are deleted.
//!
//! Although many common plotting commands are defined in [`commands`], users
//! can define their own by simply implementing `Matplotlib`. This requires
//! declaring whether the command should be counted as part of the script's
//! prelude (in which case it is automatically sorted to the top of the script),
//! what data should be included in the JSON file, and what Python code should
//! eventually be included in the script. *This library does not validate any
//! Python code whatsoever*. Users may also wish to implement [`MatplotlibOpts`]
//! to add optional keyword arguments.
//!
//! ```
//! use mpl::{
//!     Matplotlib,
//!     MatplotlibOpts,
//!     Opt,
//!     PyValue,
//!     AsPy,
//!     serde_json::Value,
//! };
//!
//! // example impl for a basic call to `plot`
//!
//! #[derive(Clone, Debug)]
//! struct Plot {
//!     x: Vec<f64>,
//!     y: Vec<f64>,
//!     opts: Vec<Opt>, // optional keyword arguments
//! }
//!
//! impl Plot {
//!     /// Create a new `Plot` with no options.
//!     fn new<X, Y>(x: X, y: Y) -> Self
//!     where
//!         X: IntoIterator<Item = f64>,
//!         Y: IntoIterator<Item = f64>,
//!     {
//!         Self {
//!             x: x.into_iter().collect(),
//!             y: y.into_iter().collect(),
//!             opts: Vec::new(),
//!         }
//!     }
//! }
//!
//! impl Matplotlib for Plot {
//!     // Commands with `is_prelude == true` are run first
//!     fn is_prelude(&self) -> bool { false }
//!
//!     fn data(&self) -> Option<Value> {
//!         let x: Vec<Value> = self.x.iter().copied().map(Value::from).collect();
//!         let y: Vec<Value> = self.y.iter().copied().map(Value::from).collect();
//!         Some(Value::Array(vec![x.into(), y.into()]))
//!     }
//!
//!     fn py_cmd(&self) -> String {
//!         // JSON data is guaranteed to be loaded in a variable called `data`
//!         format!("ax.plot(data[0], data[1], {})", self.opts.as_py())
//!     }
//! }
//!
//! // allow for keyword arguments to be added
//! impl MatplotlibOpts for Plot {
//!     fn kwarg<T>(&mut self, key: &str, val: T) -> &mut Self
//!     where T: Into<PyValue>
//!     {
//!         self.opts.push((key, val).into());
//!         self
//!     }
//! }
//! ```
//!
//! ## Example
//! ```ignore
//! use std::f64::consts::TAU;
//! use mpl::{ Mpl, Run, MatplotlibOpts, commands as c };
//!
//! let dx: f64 = TAU / 50.0;
//! let x: Vec<f64> = (0..50_u32).map(|k| f64::from(k) * dx).collect();
//! let y1: Vec<f64> = x.iter().copied().map(f64::sin).collect();
//! let y2: Vec<f64> = x.iter().copied().map(f64::cos).collect();
//!
//! Mpl::new()
//!     & c::DefPrelude
//!     & c::rcparam("axes.grid", true) // global rc parameters
//!     & c::rcparam("axes.linewidth", 0.65)
//!     & c::rcparam("lines.linewidth", 0.8)
//!     & c::DefInit
//!     & c::plot(x.clone(), y1) // the basic plotting command
//!         .o("marker", "o") // pass optional keyword arguments
//!         .o("color", "b")  // via `MatplotlibOpts`
//!         .o("label", r"$\\sin(x)$")
//!     & c::plot(x,         y2) // `&` is overloaded to allow for Haskell-like
//!         .o("marker", "D")    // patterns, can also use `Mpl::then`
//!         .o("color", "r")
//!         .o("label", r"$\\cos(x)$")
//!     & c::legend()
//!     & c::xlabel("$x$")
//!     | Run::Show // `|` consumes the final `Mpl` value; this calls
//!                 // `pyplot.show` to launch an interactive interface
//! ```
//!
//! [matplotlib]: https://matplotlib.org/
//! [matplotlib-hs]: https://hackage.haskell.org/package/matplotlib
//! [plotpy]: https://crates.io/crates/plotpy
//! [plotters]: https://crates.io/crates/plotters

mod core;
pub use core::{
    Matplotlib,
    MatplotlibOpts,
    Mpl,
    GSPos,
    Run,
    Opt,
    opt,
    AsPy,
    PyValue,
    MplError,
    MplResult,
    PRELUDE,
    INIT,
};
pub mod commands;

/// Re-exported for compatibility.
pub use serde_json;

