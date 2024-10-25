//! Commonly used plotting commands.
//!
//! This module contains types representing many common plotting commands,
//! implementing [`Matplotlib`] and sometimes [`MatplotlibOpts`]. Each can be
//! instantiated using their constructor methods or using a corresponding
//! function from this module for convenience, e.g.
//!
//! ```
//! # use matplotlib::commands::*;
//! let p1 = Plot::new([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]);
//! let p2 =      plot([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]);
//!
//! assert_eq!(p1, p2);
//! ```
//!
//! **Note**: Several constructors take iterators of flat 3-, 4-, or 6-element
//! tuples. This is inconvenient with respect to [`Iterator::zip`], so this
//! module also provides [`Associator`] and [`assoc`] to help with
//! rearrangement.

use serde_json::Value;
use crate::core::{
    Matplotlib,
    MatplotlibOpts,
    Opt,
    GSPos,
    PyValue,
    AsPy,
    PRELUDE,
    INIT,
};

/// Direct injection of arbitrary Python.
///
/// See [`Prelude`] for prelude code.
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Raw(pub String);

impl Raw {
    /// Create a new [`Raw`].
    pub fn new(s: &str) -> Self { Self(s.into()) }
}

/// Create a new [`Raw`].
pub fn raw(s: &str) -> Raw { Raw::new(s) }

impl Matplotlib for Raw {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String { self.0.clone() }
}

/// Direct injection of arbitrary Python into the prelude.
///
/// See [`Raw`] for main body code.
///
/// Prelude: **Yes**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prelude(pub String);

impl Prelude {
    /// Create a new `Prelude`.
    pub fn new(s: &str) -> Self { Self(s.into()) }
}

/// Create a new [`Prelude`].
pub fn prelude(s: &str) -> Prelude { Prelude::new(s) }

impl Matplotlib for Prelude {
    fn is_prelude(&self) -> bool { true }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String { self.0.clone() }
}

/// Default list of imports and library setup, not including `rcParams`.
///
/// This is automatically added to a [`Mpl`][crate::core::Mpl] when it's run if
/// no other commands are present for which [`Matplotlib::is_prelude`] equals
/// `true`.
///
/// See [`PRELUDE`].
///
/// Prelude: **Yes**
///
/// JSON data: **None**
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DefPrelude;

impl Matplotlib for DefPrelude {
    fn is_prelude(&self) -> bool { true }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String { PRELUDE.into() }
}

/// Default initialization of `fig` and `ax` plotting objects.
///
/// See [`INIT`].
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DefInit;

impl Matplotlib for DefInit {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String { INIT.into() }
}

/// Initialize to a figure with a single set of 3D axes.
///
/// The type of the axes object is `mpl_toolkits.mplot3d.axes3d.Axes3D`.
///
/// Requires [`DefPrelude`].
///
/// ```python
/// fig = plt.figure()
/// ax = axes3d.Axes3D(fig, auto_add_to_figure=False, **{opts})
/// fig.add_axes(ax)
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Init3D {
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Init3D {
    /// Create a new `Init3D` with no options.
    pub fn new() -> Self { Self { opts: Vec::new() } }
}

impl Matplotlib for Init3D {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("\
            fig = plt.figure()\n\
            ax = axes3d.Axes3D(fig, auto_add_to_figure=False{}{})\n\
            fig.add_axes(ax)",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Init3D {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Initialize to a figure with a regular grid of plots.
///
/// All `Axes` objects will be stored in a 2D Numpy array under the local
/// variable `AX`, and the script will be initially focused on the upper-left
/// corner of the array, i.e. `ax = AX[0, 0]`.
///
/// Requires [`DefPrelude`].
///
/// ```python
/// fig, AX = plt.subplots(nrows={nrows}, ncols={ncols}, **{opts})
/// AX = AX.reshape(({nrows}, {ncols}))
/// ax = AX[0, 0]
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct InitGrid {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl InitGrid {
    /// Create a new `InitGrid` with no options.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self { nrows, ncols, opts: Vec::new() }
    }
}

/// Create a new [`InitGrid`] with no options.
pub fn init_grid(nrows: usize, ncols: usize) -> InitGrid {
    InitGrid::new(nrows, ncols)
}

impl Matplotlib for InitGrid {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("\
            fig, AX = plt.subplots(nrows={}, ncols={}{}{})\n\
            AX = AX.reshape(({}, {}))\n\
            ax = AX[0, 0]",
            self.nrows,
            self.ncols,
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
            self.nrows,
            self.ncols,
        )
    }
}

impl MatplotlibOpts for InitGrid {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Initialize a figure with Matplotlib's `gridspec`.
///
/// Keyword arguments are passed to `plt.Figure.add_gridspec`, and each
/// subplot's position in the gridspec is specified using a [`GSPos`]. All
/// `Axes` objects will be stored in a 1D Numpy array under the local variable
/// `AX`, and the script will be initially focused to the subplot corresponding
/// to the first `GSPos` encountered, i.e. `ax = AX[0]`.
///
/// Requires [`DefPrelude`].
///
/// ```python
/// fig = plt.figure()
/// gs = fig.add_gridspec(**{opts})
/// AX = np.array([
///     # sub-plots generated from {positions}...
/// ])
/// # share axes between sub-plots...
/// ax = AX[0]
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct InitGridSpec {
    /// Keyword arguments.
    pub gridspec_kw: Vec<Opt>,
    /// Sub-plot positions and axis sharing.
    pub positions: Vec<GSPos>,
}

impl InitGridSpec {
    /// Create a new `InitGridSpec`.
    pub fn new<I, P>(gridspec_kw: I, positions: P) -> Self
    where
        I: IntoIterator<Item = Opt>,
        P: IntoIterator<Item = GSPos>,
    {
        Self {
            gridspec_kw: gridspec_kw.into_iter().collect(),
            positions: positions.into_iter().collect(),
        }
    }
}

/// Create a new [`InitGridSpec`].
pub fn init_gridspec<I, P>(gridspec_kw: I, positions: P) -> InitGridSpec
where
    I: IntoIterator<Item = Opt>,
    P: IntoIterator<Item = GSPos>,
{
    InitGridSpec::new(gridspec_kw, positions)
}

impl Matplotlib for InitGridSpec {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        let mut code =
            format!("\
                fig = plt.figure()\n\
                gs = fig.add_gridspec({})\n\
                AX = np.array([\n",
                self.gridspec_kw.as_py(),
            );
        for GSPos { i, j, sharex: _, sharey: _ } in self.positions.iter() {
            code.push_str(
                &format!("    fig.add_subplot(gs[{}:{}, {}:{}]),\n",
                    i.start, i.end, j.start, j.end,
                )
            );
        }
        code.push_str("])\n");
        let iter = self.positions.iter().enumerate();
        for (k, GSPos { i: _, j: _, sharex, sharey }) in iter {
            if let Some(x) = sharex {
                code.push_str(&format!("AX[{}].sharex(AX[{}])\n", k, x));
            }
            if let Some(y) = sharey {
                code.push_str(&format!("AX[{}].sharey(AX[{}])\n", k, y));
            }
        }
        code.push_str("ax = AX[0]\n");
        code
    }
}

impl MatplotlibOpts for InitGridSpec {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.gridspec_kw.push((key, val).into());
        self
    }
}

/// Set the value of an RC parameter.
///
/// **Note**: This type limits values to basic Python types; this is fine for
/// all but a few of the RC parameters; e.g. `axes.prop_cycle`. For the
/// remainder, use [`Raw`] or [`Prelude`].
///
/// ```python
/// plt.rcParams["{key}"] = {val}
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct RcParam {
    /// Key in `matplotlib.pyplot.rcParams`.
    pub key: String,
    /// Value setting.
    pub val: PyValue,
}

impl RcParam {
    /// Create a new `RcParam`.
    pub fn new<T: Into<PyValue>>(key: &str, val: T) -> Self {
        Self { key: key.into(), val: val.into() }
    }
}

/// Create a new [`RcParam`].
pub fn rcparam<T: Into<PyValue>>(key: &str, val: T) -> RcParam {
    RcParam::new(key, val)
}

impl Matplotlib for RcParam {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("plt.rcParams[\"{}\"] = {}", self.key, self.val.as_py())
    }
}

/// Activate or deactivate TeX text.
///
/// ```python
/// plt.rcParams["text.usetex"] = {0}
/// ```
///
/// Prelude: **Yes**
///
/// JSON data: **None**
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TeX(pub bool);

impl TeX {
    /// Turn TeX text on.
    pub fn on() -> Self { Self(true) }

    /// Turn TeX text off.
    pub fn off() -> Self { Self(false) }
}

/// Turn TeX text on.
pub fn tex_on() -> TeX { TeX(true) }

/// Turn TeX text off.
pub fn tex_off() -> TeX { TeX(false) }

impl Matplotlib for TeX {
    fn is_prelude(&self) -> bool { true }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("plt.rcParams[\"text.usetex\"] = {}", self.0.as_py())
    }
}

/// Set the local variable `ax` to a different set of axes.
///
/// ```python
/// ax = {0}
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FocusAx(pub String);

impl FocusAx {
    /// Create a new `FocusAx`.
    pub fn new(expr: &str) -> Self { Self(expr.into()) }
}

/// Create a new [`FocusAx`].
pub fn focus_ax(expr: &str) -> FocusAx { FocusAx::new(expr) }

impl Matplotlib for FocusAx {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String { format!("ax = {}", self.0) }
}

/// Set the local variable `fig` to a different figure.
///
/// ```python
/// fig = {0}
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FocusFig(pub String);

impl FocusFig {
    /// Create a new `FocusFig`.
    pub fn new(expr: &str) -> Self { Self(expr.into()) }
}

/// Create a new [`FocusFig`].
pub fn focus_fig(expr: &str) -> FocusFig { FocusFig::new(expr) }

impl Matplotlib for FocusFig {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String { format!("fig = {}", self.0) }
}

/// Set the local variable `cbar` to a different colorbar.
///
/// ```python
/// cbar = {0}
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FocusCBar(pub String);

impl FocusCBar {
    /// Create a new `FocusCBar`.
    pub fn new(expr: &str) -> Self { Self(expr.into()) }
}

/// Create a new [`FocusCBar`].
pub fn focus_cbar(expr: &str) -> FocusCBar { FocusCBar::new(expr) }

impl Matplotlib for FocusCBar {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String { format!("cbar = {}", self.0) }
}

/// Set the local variable `im` to a different colorbar.
///
/// ```python
/// im = {0}
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FocusIm(pub String);

impl FocusIm {
    /// Create a new `FocusIm`.
    pub fn new(expr: &str) -> Self { Self(expr.into()) }
}

/// Create a new [`FocusIm`].
pub fn focus_im(expr: &str) -> FocusIm { FocusIm::new(expr) }

impl Matplotlib for FocusIm {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String { format!("im = {}", self.0) }
}

/// A (*x*, *y*) plot.
///
/// ```python
/// ax.plot({x}, {y}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Plot {
    /// X-coordinates.
    pub x: Vec<f64>,
    /// Y-coordinates.
    pub y: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Plot {
    /// Create a new `Plot` with no options.
    pub fn new<X, Y>(x: X, y: Y) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
    {
        Self {
            x: x.into_iter().collect(),
            y: y.into_iter().collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `Plot` with no options from a single iterator.
    pub fn new_pairs<I>(data: I) -> Self
    where I: IntoIterator<Item = (f64, f64)>
    {
        let (x, y): (Vec<f64>, Vec<f64>) = data.into_iter().unzip();
        Self { x, y, opts: Vec::new() }
    }
}

/// Create a new [`Plot`] with no options.
pub fn plot<X, Y>(x: X, y: Y) -> Plot
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
{
    Plot::new(x, y)
}

/// Create a new [`Plot`] with no options from a single iterator.
pub fn plot_pairs<I>(data: I) -> Plot
where I: IntoIterator<Item = (f64, f64)>
{
    Plot::new_pairs(data)
}

impl Matplotlib for Plot {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> = self.x.iter().copied().map(Value::from).collect();
        let y: Vec<Value> = self.y.iter().copied().map(Value::from).collect();
        Some(Value::Array(vec![x.into(), y.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.plot(data[0], data[1]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Plot {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A histogram of a data set.
///
/// ```python
/// ax.hist({data}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `list[float]`
#[derive(Clone, Debug, PartialEq)]
pub struct Hist {
    /// Data set.
    pub data: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Hist {
    /// Create a new `Hist` with no options.
    pub fn new<I>(data: I) -> Self
    where I: IntoIterator<Item = f64>
    {
        let data: Vec<f64> = data.into_iter().collect();
        Self { data, opts: Vec::new() }
    }
}

/// Create a new [`Hist`] with no options.
pub fn hist<I>(data: I) -> Hist
where I: IntoIterator<Item = f64>
{
    Hist::new(data)
}

impl Matplotlib for Hist {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let data: Vec<Value> =
            self.data.iter().copied().map(Value::from).collect();
        Some(Value::Array(data))
    }

    fn py_cmd(&self) -> String {
        format!("ax.hist(data{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Hist {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A histogram of two variables.
///
/// ```python
/// ax.hist2d({data}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Hist2d {
    /// X data set.
    pub x: Vec<f64>,
    /// Y data set.
    pub y: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Hist2d {
    /// Create a new `Hist2d` with no options.
    pub fn new<X, Y>(x: X, y: Y) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
    {
        let x: Vec<f64> = x.into_iter().collect();
        let y: Vec<f64> = y.into_iter().collect();
        Self { x, y, opts: Vec::new() }
    }

    /// Create a new `Hist2d` with no options from a single iterator.
    pub fn new_pairs<I>(data: I) -> Self
    where I: IntoIterator<Item = (f64, f64)>
    {
        let (x, y): (Vec<f64>, Vec<f64>) = data.into_iter().unzip();
        Self { x, y, opts: Vec::new() }
    }
}

/// Create a new [`Hist2d`] with no options.
pub fn hist2d<X, Y>(x: X, y: Y) -> Hist2d
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
{
    Hist2d::new(x, y)
}

/// Create a new [`Hist2d`] with no options from a single iterator.
pub fn hist2d_pairs<I>(data: I) -> Hist2d
where I: IntoIterator<Item = (f64, f64)>
{
    Hist2d::new_pairs(data)
}

impl Matplotlib for Hist2d {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> = self.x.iter().copied().map(Value::from).collect();
        let y: Vec<Value> = self.y.iter().copied().map(Value::from).collect();
        Some(Value::Array(vec![x.into(), y.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.hist2d(data[0], data[1]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Hist2d {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A (*x*, *y*) scatter plot.
///
/// ```python
/// ax.scatter({x} {y}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Scatter {
    /// X-coordinates.
    pub x: Vec<f64>,
    /// Y-coordinates.
    pub y: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Scatter {
    /// Create a new `Scatter` with no options.
    pub fn new<X, Y>(x: X, y: Y) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
    {
        Self {
            x: x.into_iter().collect(),
            y: y.into_iter().collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `Plot` with no options from a single iterator.
    pub fn new_pairs<I>(data: I) -> Self
    where I: IntoIterator<Item = (f64, f64)>
    {
        let (x, y): (Vec<f64>, Vec<f64>) = data.into_iter().unzip();
        Self { x, y, opts: Vec::new() }
    }
}

/// Create a new [`Scatter`] with no options.
pub fn scatter<X, Y>(x: X, y: Y) -> Scatter
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
{
    Scatter::new(x, y)
}

/// Create a new [`Scatter`] with no options from a single iterator.
pub fn scatter_pairs<I>(data: I) -> Scatter
where I: IntoIterator<Item = (f64, f64)>
{
    Scatter::new_pairs(data)
}

impl Matplotlib for Scatter {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> = self.x.iter().copied().map(Value::from).collect();
        let y: Vec<Value> = self.y.iter().copied().map(Value::from).collect();
        Some(Value::Array(vec![x.into(), y.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.scatter(data[0], data[1]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Scatter {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A vector field plot.
///
/// ```python
/// ax.quiver({x}, {y}, {vx}, {vy}, **{ops})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float], list[float], list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Quiver {
    /// X-coordinates.
    pub x: Vec<f64>,
    /// Y-coordinates.
    pub y: Vec<f64>,
    /// Vector X-components.
    pub vx: Vec<f64>,
    /// Vector Y-components.
    pub vy: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Quiver {
    /// Create a new `Quiver` with no options.
    pub fn new<X, Y, VX, VY>(x: X, y: Y, vx: VX, vy: VY) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
        VX: IntoIterator<Item = f64>,
        VY: IntoIterator<Item = f64>,
    {
        Self {
            x: x.into_iter().collect(),
            y: y.into_iter().collect(),
            vx: vx.into_iter().collect(),
            vy: vy.into_iter().collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `Quiver` with no options from iterators over coordinate
    /// pairs.
    pub fn new_pairs<I, VI>(xy: I, vxy: VI) -> Self
    where
        I: IntoIterator<Item = (f64, f64)>,
        VI: IntoIterator<Item = (f64, f64)>,
    {
        let (x, y) = xy.into_iter().unzip();
        let (vx, vy) = vxy.into_iter().unzip();
        Self { x, y, vx, vy, opts: Vec::new() }
    }

    /// Create a new `Quiver` with no options from a single iterator. The first
    /// two elements of each iterator item should be spatial coordinates and the
    /// last two should be vector components.
    pub fn new_data<I>(data: I) -> Self
    where I: IntoIterator<Item = (f64, f64, f64, f64)>
    {
        let (((x, y), vx), vy) = data.into_iter().map(assoc).unzip();
        Self { x, y, vx, vy, opts: Vec::new() }
    }
}

/// Create a new [`Quiver`] with no options.
pub fn quiver<X, Y, VX, VY>(x: X, y: Y, vx: VX, vy: VY) -> Quiver
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
    VX: IntoIterator<Item = f64>,
    VY: IntoIterator<Item = f64>,
{
    Quiver::new(x, y, vx, vy)
}

/// Create a new [`Quiver`] with no options from iterators over coordinate
/// pairs.
pub fn quiver_pairs<I, VI>(xy: I, vxy: VI) -> Quiver
where
    I: IntoIterator<Item = (f64, f64)>,
    VI: IntoIterator<Item = (f64, f64)>,
{
    Quiver::new_pairs(xy, vxy)
}

/// Create a new [`Quiver`] with no options from a single iterator. The first
/// two elements of each iterator item should be spatial coordinates and
/// the last two should be vector components.
pub fn quiver_data<I>(data: I) -> Quiver
where I: IntoIterator<Item = (f64, f64, f64, f64)>
{
    Quiver::new_data(data)
}

impl Matplotlib for Quiver {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> = self.x.iter().copied().map(Value::from).collect();
        let y: Vec<Value> = self.y.iter().copied().map(Value::from).collect();
        let vx: Vec<Value> = self.vx.iter().copied().map(Value::from).collect();
        let vy: Vec<Value> = self.vy.iter().copied().map(Value::from).collect();
        Some(Value::Array(vec![x.into(), y.into(), vx.into(), vy.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.quiver(data[0], data[1], data[2], data[3]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Quiver {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A bar plot.
///
/// ```python
/// ax.bar({x}, {y}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Bar {
    /// X-coordinates.
    pub x: Vec<f64>,
    /// Y-coordinates.
    pub y: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Bar {
    /// Create a new `Bar` with no options.
    pub fn new<X, Y>(x: X, y: Y) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
    {
        Self {
            x: x.into_iter().collect(),
            y: y.into_iter().collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `Bar` with options from a single iterator.
    pub fn new_pairs<I>(data: I) -> Self
    where I: IntoIterator<Item = (f64, f64)>
    {
        let (x, y): (Vec<f64>, Vec<f64>) = data.into_iter().unzip();
        Self { x, y, opts: Vec::new() }
    }
}

/// Create a new [`Bar`] with no options.
pub fn bar<X, Y>(x: X, y: Y) -> Bar
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
{
    Bar::new(x, y)
}

/// Create a new [`Bar`] with options from a single iterator.
pub fn bar_pairs<I>(data: I) -> Bar
where I: IntoIterator<Item = (f64, f64)>
{
    Bar::new_pairs(data)
}

impl Matplotlib for Bar {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> =
            self.x.iter().copied().map(Value::from).collect();
        let y: Vec<Value> =
            self.y.iter().copied().map(Value::from).collect();
        Some(Value::Array(vec![x.into(), y.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.bar(data[0], data[1]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Bar {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A horizontal bar plot.
///
/// ```python
/// ax.barh({y}, {w}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct BarH {
    /// Y-coordinates.
    pub y: Vec<f64>,
    /// Bar widths.
    pub w: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl BarH {
    /// Create a new `BarH` with no options.
    pub fn new<Y, W>(y: Y, w: W) -> Self
    where
        Y: IntoIterator<Item = f64>,
        W: IntoIterator<Item = f64>,
    {
        Self {
            y: y.into_iter().collect(),
            w: w.into_iter().collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `BarH` with options from a single iterator.
    pub fn new_pairs<I>(data: I) -> Self
    where I: IntoIterator<Item = (f64, f64)>
    {
        let (y, w): (Vec<f64>, Vec<f64>) = data.into_iter().unzip();
        Self { y, w, opts: Vec::new() }
    }
}

/// Create a new [`BarH`] with no options.
pub fn barh<Y, W>(y: Y, w: W) -> BarH
where
    Y: IntoIterator<Item = f64>,
    W: IntoIterator<Item = f64>,
{
    BarH::new(y, w)
}

/// Create a new [`BarH`] with options from a single iterator.
pub fn barh_pairs<I>(data: I) -> BarH
where I: IntoIterator<Item = (f64, f64)>
{
    BarH::new_pairs(data)
}

impl Matplotlib for BarH {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let y: Vec<Value> =
            self.y.iter().copied().map(Value::from).collect();
        let w: Vec<Value> =
            self.w.iter().copied().map(Value::from).collect();
        Some(Value::Array(vec![y.into(), w.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.barh(data[0], data[1]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for BarH {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Plot with error bars.
///
/// ```python
/// ax.errorbar({x}, {y}, {e}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float], list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Errorbar {
    /// X-coordinates.
    pub x: Vec<f64>,
    /// Y-coordinates.
    pub y: Vec<f64>,
    /// Symmetric error bar sizes on Y-coordinates.
    pub e: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Errorbar {
    /// Create a new `Errorbar` with no options.
    pub fn new<X, Y, E>(x: X, y: Y, e: E) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
        E: IntoIterator<Item = f64>,
    {
        Self {
            x: x.into_iter().collect(),
            y: y.into_iter().collect(),
            e: e.into_iter().collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `Errorbar` with no options from a single iterator.
    pub fn new_data<I>(data: I) -> Self
    where I: IntoIterator<Item = (f64, f64, f64)>
    {
        let ((x, y), e) = data.into_iter().map(assoc).unzip();
        Self { x, y, e, opts: Vec::new() }
    }
}

/// Create a new [`Errorbar`] with no options.
pub fn errorbar<X, Y, E>(x: X, y: Y, e: E) -> Errorbar
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
    E: IntoIterator<Item = f64>,
{
    Errorbar::new(x, y, e)
}

/// Create a new [`Errorbar`] with no options from a single iterator.
pub fn errorbar_data<I>(data: I) -> Errorbar
where I: IntoIterator<Item = (f64, f64, f64)>
{
    Errorbar::new_data(data)
}

impl Matplotlib for Errorbar {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> = self.x.iter().copied().map(Value::from).collect();
        let y: Vec<Value> = self.y.iter().copied().map(Value::from).collect();
        let e: Vec<Value> = self.e.iter().copied().map(Value::from).collect();
        Some(Value::Array(vec![x.into(), y.into(), e.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.errorbar(data[0], data[1], data[2]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Errorbar {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Convert a `FillBetween` to an `Errorbar`, maintaining all options.
impl From<FillBetween> for Errorbar {
    fn from(fill_between: FillBetween) -> Self {
        let FillBetween { x, mut y1, mut y2, opts } = fill_between;
        y1.iter_mut()
            .zip(y2.iter_mut())
            .for_each(|(y1k, y2k)| {
                let y1 = *y1k;
                let y2 = *y2k;
                *y1k = 0.5 * (y1 + y2);
                *y2k = 0.5 * (y1 - y2).abs();
            });
        Self { x, y: y1, e: y2, opts }
    }
}

/// Plot with asymmetric error bars.
///
/// ```python
/// ax.errorbar({x}, {y}, [{e_neg}, {e_pos}], **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float], list[float], list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Errorbar2 {
    /// X-coordinates.
    pub x: Vec<f64>,
    /// Y-coordinates.
    pub y: Vec<f64>,
    /// Negative-sided error bar sizes on Y-coordinates.
    pub e_neg: Vec<f64>,
    /// Positive-sided error bar sizes on Y-coordinates.
    pub e_pos: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Errorbar2 {
    /// Create a new `Errorbar2` with no options.
    pub fn new<X, Y, E1, E2>(x: X, y: Y, e_neg: E1, e_pos: E2) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
        E1: IntoIterator<Item = f64>,
        E2: IntoIterator<Item = f64>,
    {
        Self {
            x: x.into_iter().collect(),
            y: y.into_iter().collect(),
            e_neg: e_neg.into_iter().collect(),
            e_pos: e_pos.into_iter().collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `Errorbar2` with no options from a single iterator.
    pub fn new_data<I>(data: I) -> Self
    where I: IntoIterator<Item = (f64, f64, f64, f64)>
    {
        let (((x, y), e_neg), e_pos) =
            data.into_iter().map(assoc).unzip();
        Self { x, y, e_neg, e_pos, opts: Vec::new() }
    }
}

/// Create a new [`Errorbar2`] with no options.
pub fn errorbar2<X, Y, E1, E2>(x: X, y: Y, e_neg: E1, e_pos: E2) -> Errorbar2
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
    E1: IntoIterator<Item = f64>,
    E2: IntoIterator<Item = f64>,
{
    Errorbar2::new(x, y, e_neg, e_pos)
}

/// Create a new [`Errorbar2`] with no options from a single iterator.
pub fn errorbar2_data<I>(data: I) -> Errorbar2
where I: IntoIterator<Item = (f64, f64, f64, f64)>
{
    Errorbar2::new_data(data)
}

impl Matplotlib for Errorbar2 {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> = self.x.iter().copied().map(Value::from).collect();
        let y: Vec<Value> = self.y.iter().copied().map(Value::from).collect();
        let e_neg: Vec<Value> =
            self.e_neg.iter().copied().map(Value::from).collect();
        let e_pos: Vec<Value> =
            self.e_pos.iter().copied().map(Value::from).collect();
        Some(Value::Array(
            vec![x.into(), y.into(), e_neg.into(), e_pos.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.errorbar(data[0], data[1], [data[2], data[3]]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Errorbar2 {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

struct Chunks<I, T>
where I: Iterator<Item = T>
{
    chunksize: usize,
    buflen: usize,
    buf: Vec<T>,
    iter: I,
}

impl<I, T> Chunks<I, T>
where I: Iterator<Item = T>
{
    fn new(iter: I, chunksize: usize) -> Self {
        if chunksize == 0 { panic!("chunk size cannot be zero"); }
        Self {
            chunksize,
            buflen: 0,
            buf: Vec::with_capacity(chunksize),
            iter,
        }
    }
}

impl<I, T> Iterator for Chunks<I, T>
where I: Iterator<Item = T>
{
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(item) = self.iter.next() {
                self.buf.push(item);
                self.buflen += 1;
                if self.buflen == self.chunksize {
                    let mut bufswap = Vec::with_capacity(self.chunksize);
                    std::mem::swap(&mut bufswap, &mut self.buf);
                    self.buflen = 0;
                    return Some(bufswap);
                } else {
                    continue;
                }
            } else if self.buflen > 0 {
                let mut bufswap = Vec::with_capacity(0);
                std::mem::swap(&mut bufswap, &mut self.buf);
                self.buflen = 0;
                return Some(bufswap);
            } else {
                return None;
            }
        }
    }
}


/// Box(-and-whisker) plots for a number of data sets.
///
/// ```python
/// ax.boxplot({data}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `list[list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Boxplot {
    /// List of data sets.
    pub data: Vec<Vec<f64>>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Boxplot {
    /// Create a new `Boxplot` with no options.
    pub fn new<I, J>(data: I) -> Self
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = f64>,
    {
        let data: Vec<Vec<f64>> =
            data.into_iter()
            .map(|row| row.into_iter().collect())
            .collect();
        Self { data, opts: Vec::new() }
    }

    /// Create a new `Boxplot` from a flattened iterator over a number of data
    /// set of size `size`.
    ///
    /// The last data set is truncated if `size` does not evenly divide the
    /// length of the iterator.
    ///
    /// *Panics if `size == 0`*.
    pub fn new_flat<I>(data: I, size: usize) -> Self
    where I: IntoIterator<Item = f64>
    {
        if size == 0 { panic!("data set size cannot be zero"); }
        let data: Vec<Vec<f64>> =
            Chunks::new(data.into_iter(), size)
            .collect();
        Self { data, opts: Vec::new() }
    }
}

/// Create a new [`Boxplot`] with no options.
pub fn boxplot<I, J>(data: I) -> Boxplot
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = f64>,
{
    Boxplot::new(data)
}

/// Create a new [`Boxplot`] from a flattened iterator over a number of data
/// set of size `size`.
///
/// The last data set is truncated if `size` does not evenly divide the
/// length of the iterator.
///
/// *Panics if `size == 0`*.
pub fn boxplot_flat<I>(data: I, size: usize) -> Boxplot
where I: IntoIterator<Item = f64>
{
    Boxplot::new_flat(data, size)
}

impl Matplotlib for Boxplot {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let data: Vec<Value> =
            self.data.iter()
            .map(|row| {
                let row: Vec<Value> =
                    row.iter().copied().map(Value::from).collect();
                Value::Array(row)
            })
            .collect();
        Some(Value::Array(data))
    }

    fn py_cmd(&self) -> String {
        format!("ax.boxplot(data{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Boxplot {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Violin plots for a number of data sets.
///
/// ```python
/// ax.violinplot({data}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `list[list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Violinplot {
    /// List of data sets.
    pub data: Vec<Vec<f64>>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Violinplot {
    /// Create a new `Violinplot` with no options.
    pub fn new<I, J>(data: I) -> Self
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = f64>,
    {
        let data: Vec<Vec<f64>> =
            data.into_iter()
            .map(|row| row.into_iter().collect())
            .collect();
        Self { data, opts: Vec::new() }
    }

    /// Create a new `Violinplot` from a flattened iterator over a number of
    /// data set of size `size`.
    ///
    /// The last data set is truncated if `size` does not evenly divide the
    /// length of the iterator.
    ///
    /// *Panics if `size == 0`*.
    pub fn new_flat<I>(data: I, size: usize) -> Self
    where I: IntoIterator<Item = f64>
    {
        if size == 0 { panic!("data set size cannot be zero"); }
        let data: Vec<Vec<f64>> =
            Chunks::new(data.into_iter(), size)
            .collect();
        Self { data, opts: Vec::new() }
    }
}

/// Create a new [`Violinplot`] with no options.
pub fn violinplot<I, J>(data: I) -> Violinplot
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = f64>,
{
    Violinplot::new(data)
}

/// Create a new [`Violinplot`] from a flattened iterator over a number of
/// data set of size `size`.
///
/// The last data set is truncated if `size` does not evenly divide the
/// length of the iterator.
///
/// *Panics if `size == 0`*.
pub fn violinplot_flat<I>(data: I, size: usize) -> Violinplot
where I: IntoIterator<Item = f64>
{
    Violinplot::new_flat(data, size)
}

impl Matplotlib for Violinplot {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let data: Vec<Value> =
            self.data.iter()
            .map(|row| {
                let row: Vec<Value> =
                    row.iter().copied().map(Value::from).collect();
                Value::Array(row)
            })
            .collect();
        Some(Value::Array(data))
    }

    fn py_cmd(&self) -> String {
        format!("ax.violinplot(data{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Violinplot {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A contour plot for a (*x*, *y*, *z*) surface.
///
/// This command sets a local variable `im` to the output of the call to
/// `contour` for use with [`Colorbar`]
///
/// ```python
/// im = ax.contour({x}, {y}, {z}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float], list[list[float]]]`
///
/// **Note**: No checking is performed for the shapes/sizes of the data arrays.
#[derive(Clone, Debug, PartialEq)]
pub struct Contour {
    /// X-coordinates.
    pub x: Vec<f64>,
    /// Y-coordinates.
    pub y: Vec<f64>,
    /// Z-coordinates.
    ///
    /// Columns correspond to x-coordinates.
    pub z: Vec<Vec<f64>>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Contour {
    /// Create a new `Contour` with no options.
    pub fn new<X, Y, ZI, ZJ>(x: X, y: Y, z: ZI) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
        ZI: IntoIterator<Item = ZJ>,
        ZJ: IntoIterator<Item = f64>,
    {
        let x: Vec<f64> = x.into_iter().collect();
        let y: Vec<f64> = y.into_iter().collect();
        let z: Vec<Vec<f64>> =
            z.into_iter()
            .map(|row| row.into_iter().collect())
            .collect();
        Self { x, y, z, opts: Vec::new() }
    }

    /// Create a new `Contour` with no options using a flattened iterator over
    /// z-coordinates.
    ///
    /// *Panics if the number of x-coordinates is zero*.
    pub fn new_flat<X, Y, Z>(x: X, y: Y, z: Z) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
        Z: IntoIterator<Item = f64>,
    {
        let x: Vec<f64> = x.into_iter().collect();
        if x.is_empty() { panic!("x-coordinate array cannot be empty"); }
        let y: Vec<f64> = y.into_iter().collect();
        let z: Vec<Vec<f64>> =
            Chunks::new(z.into_iter(), x.len())
            .collect();
        Self { x, y, z, opts: Vec::new() }
    }
}

/// Create a new [`Contour`] with no options.
pub fn contour<X, Y, ZI, ZJ>(x: X, y: Y, z: ZI) -> Contour
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
    ZI: IntoIterator<Item = ZJ>,
    ZJ: IntoIterator<Item = f64>,
{
    Contour::new(x, y, z)
}

/// Create a new [`Contour`] with no options using a flattened iterator over
/// z-coordinates.
///
/// *Panics if the number of x-coordinates is zero*.
pub fn contour_flat<X, Y, Z>(x: X, y: Y, z: Z) -> Contour
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
    Z: IntoIterator<Item = f64>,
{
    Contour::new_flat(x, y, z)
}

impl Matplotlib for Contour {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> = self.x.iter().copied().map(Value::from).collect();
        let y: Vec<Value> = self.y.iter().copied().map(Value::from).collect();
        let z: Vec<Value> =
            self.z.iter()
            .map(|row| {
                let row: Vec<Value> =
                    row.iter().copied().map(Value::from).collect();
                Value::Array(row)
            })
            .collect();
        Some(Value::Array(vec![x.into(), y.into(), z.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("im = ax.contour(data[0], data[1], data[2]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Contour {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A filled contour plot for a (*x*, *y*, *z*) surface.
///
/// This command sets a local variable `im` to the output of the call to
/// `contourf` for use with [`Colorbar`].
///
/// ```python
/// im = ax.contourf({x}, {y}, {z}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float], list[list[float]]]`
///
/// **Note**: No checking is performed for the shapes/sizes of the data arrays.
#[derive(Clone, Debug, PartialEq)]
pub struct Contourf {
    /// X-coordinates.
    pub x: Vec<f64>,
    /// Y-coordinates.
    pub y: Vec<f64>,
    /// Z-coordinates.
    ///
    /// Columns correspond to x-coordinates.
    pub z: Vec<Vec<f64>>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Contourf {
    /// Create a new `Contourf` with no options.
    pub fn new<X, Y, ZI, ZJ>(x: X, y: Y, z: ZI) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
        ZI: IntoIterator<Item = ZJ>,
        ZJ: IntoIterator<Item = f64>,
    {
        let x: Vec<f64> = x.into_iter().collect();
        let y: Vec<f64> = y.into_iter().collect();
        let z: Vec<Vec<f64>> =
            z.into_iter()
            .map(|row| row.into_iter().collect())
            .collect();
        Self { x, y, z, opts: Vec::new() }
    }

    /// Create a new `Contourf` with no options using a flattened iterator over
    /// z-coordinates.
    ///
    /// *Panics if the number of x-coordinates is zero*.
    pub fn new_flat<X, Y, Z>(x: X, y: Y, z: Z) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
        Z: IntoIterator<Item = f64>,
    {
        let x: Vec<f64> = x.into_iter().collect();
        if x.is_empty() { panic!("x-coordinate array cannot be empty"); }
        let y: Vec<f64> = y.into_iter().collect();
        let z: Vec<Vec<f64>> =
            Chunks::new(z.into_iter(), x.len())
            .collect();
        Self { x, y, z, opts: Vec::new() }
    }
}

/// Create a new [`Contourf`] with no options.
pub fn contourf<X, Y, ZI, ZJ>(x: X, y: Y, z: ZI) -> Contourf
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
    ZI: IntoIterator<Item = ZJ>,
    ZJ: IntoIterator<Item = f64>,
{
    Contourf::new(x, y, z)
}

/// Create a new [`Contourf`] with no options using a flattened iterator over
/// z-coordinates.
///
/// *Panics if the number of x-coordinates is zero*.
pub fn contourf_flat<X, Y, Z>(x: X, y: Y, z: Z) -> Contourf
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
    Z: IntoIterator<Item = f64>,
{
    Contourf::new_flat(x, y, z)
}

impl Matplotlib for Contourf {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> = self.x.iter().copied().map(Value::from).collect();
        let y: Vec<Value> = self.y.iter().copied().map(Value::from).collect();
        let z: Vec<Value> =
            self.z.iter()
            .map(|row| {
                let row: Vec<Value> =
                    row.iter().copied().map(Value::from).collect();
                Value::Array(row)
            })
            .collect();
        Some(Value::Array(vec![x.into(), y.into(), z.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("im = ax.contourf(data[0], data[1], data[2]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Contourf {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A 2D data set as an image.
///
/// This command sets a local variable `im` to the output of the call to
/// `imshow` for use with [`Colorbar`]
///
/// ```python
/// im = ax.imshow({data}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `list[list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Imshow {
    /// Image data.
    pub data: Vec<Vec<f64>>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Imshow {
    /// Create a new `Imshow` with no options.
    pub fn new<I, J>(data: I) -> Self
    where
        I: IntoIterator<Item = J>,
        J: IntoIterator<Item = f64>,
    {
        let data: Vec<Vec<f64>> =
            data.into_iter()
            .map(|row| row.into_iter().collect())
            .collect();
        Self { data, opts: Vec::new() }
    }

    /// Create a new `Imshow` from a flattened, column-major iterator over image
    /// data with row length `rowlen`.
    ///
    /// *Panics if `rowlen == 0`*.
    pub fn new_flat<I>(data: I, rowlen: usize) -> Self
    where I: IntoIterator<Item = f64>
    {
        if rowlen == 0 { panic!("row length cannot be zero"); }
        let data: Vec<Vec<f64>> =
            Chunks::new(data.into_iter(), rowlen)
            .collect();
        Self { data, opts: Vec::new() }
    }
}

/// Create a new [`Imshow`] with no options.
pub fn imshow<I, J>(data: I) -> Imshow
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = f64>,
{
    Imshow::new(data)
}

/// Create a new [`Imshow`] from a flattened, column-major iterator over image
/// data with row length `rowlen`.
///
/// *Panics if `rowlen == 0`*.
pub fn imshow_flat<I>(data: I, rowlen: usize) -> Imshow
where I: IntoIterator<Item = f64>
{
    Imshow::new_flat(data, rowlen)
}

impl Matplotlib for Imshow {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let data: Vec<Value> =
            self.data.iter()
            .map(|row| {
                let row: Vec<Value> =
                    row.iter().copied().map(Value::from).collect();
                Value::Array(row)
            })
            .collect();
        Some(Value::Array(data))
    }

    fn py_cmd(&self) -> String {
        format!("im = ax.imshow(data{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Imshow {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A filled area between two horizontal curves.
///
/// ```python
/// ax.fill_between({x}, {y1}, {y2}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float], list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct FillBetween {
    /// X-coordinates.
    pub x: Vec<f64>,
    /// Y-coordinates of the first curve.
    pub y1: Vec<f64>,
    /// Y-coordinates of the second curve.
    pub y2: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl FillBetween {
    /// Create a new `FillBetween` with no options.
    pub fn new<X, Y1, Y2>(x: X, y1: Y1, y2: Y2) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y1: IntoIterator<Item = f64>,
        Y2: IntoIterator<Item = f64>,
    {
        Self {
            x: x.into_iter().collect(),
            y1: y1.into_iter().collect(),
            y2: y2.into_iter().collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `FillBetween` with no options from a single iterator.
    pub fn new_data<I>(data: I) -> Self
    where I: IntoIterator<Item = (f64, f64, f64)>
    {
        let ((x, y1), y2) = data.into_iter().map(assoc).unzip();
        Self { x, y1, y2, opts: Vec::new() }
    }
}

/// Create a new [`FillBetween`] with no options.
pub fn fill_between<X, Y1, Y2>(x: X, y1: Y1, y2: Y2) -> FillBetween
where
    X: IntoIterator<Item = f64>,
    Y1: IntoIterator<Item = f64>,
    Y2: IntoIterator<Item = f64>,
{
    FillBetween::new(x, y1, y2)
}

/// Create a new [`FillBetween`] with no options from a single iterator.
pub fn fill_between_data<I>(data: I) -> FillBetween
where I: IntoIterator<Item = (f64, f64, f64)>
{
    FillBetween::new_data(data)
}

impl Matplotlib for FillBetween {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> =
            self.x.iter().copied().map(Value::from).collect();
        let y1: Vec<Value> =
            self.y1.iter().copied().map(Value::from).collect();
        let y2: Vec<Value> =
            self.y2.iter().copied().map(Value::from).collect();
        Some(Value::Array(vec![x.into(), y1.into(), y2.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.fill_between(data[0], data[1], data[2]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for FillBetween {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Convert an `Errorbar` to a `FillBetween`, maintaining all options.
impl From<Errorbar> for FillBetween {
    fn from(errorbar: Errorbar) -> Self {
        let Errorbar { x, mut y, mut e, opts } = errorbar;
        y.iter_mut()
            .zip(e.iter_mut())
            .for_each(|(yk, ek)| {
                let y = *yk;
                let e = *ek;
                *yk -= e;
                *ek += y;
            });
        Self { x, y1: y, y2: e, opts }
    }
}

/// Convert an `Errorbar2` to a `FillBetween`, maintaining all options.
impl From<Errorbar2> for FillBetween {
    fn from(errorbar2: Errorbar2) -> Self {
        let Errorbar2 { x, mut y, mut e_neg, e_pos, opts } = errorbar2;
        y.iter_mut()
            .zip(e_neg.iter_mut().zip(e_pos.iter()))
            .for_each(|(yk, (emk, epk))| {
                let y = *yk;
                let em = *emk;
                let ep = *epk;
                *yk -= em;
                *emk = y + ep;
            });
        Self { x, y1: y, y2: e_neg, opts }
    }
}

/// A filled area between two vertical curves.
///
/// ```python
/// ax.fill_betweenx({y}, {x1}, {x2}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float], list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct FillBetweenX {
    /// Y-coordinates.
    pub y: Vec<f64>,
    /// X-coordinates of the first curve.
    pub x1: Vec<f64>,
    /// X-coordinates of the second curve.
    pub x2: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl FillBetweenX {
    /// Create a new `FillBetweenX` with no options.
    pub fn new<Y, X1, X2>(y: Y, x1: X1, x2: X2) -> Self
    where
        Y: IntoIterator<Item = f64>,
        X1: IntoIterator<Item = f64>,
        X2: IntoIterator<Item = f64>,
    {
        Self {
            y: y.into_iter().collect(),
            x1: x1.into_iter().collect(),
            x2: x2.into_iter().collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `FillBetweenX` with no options from a single iterator.
    pub fn new_data<I>(data: I) -> Self
    where I: IntoIterator<Item = (f64, f64, f64)>
    {
        let ((y, x1), x2) = data.into_iter().map(assoc).unzip();
        Self { y, x1, x2, opts: Vec::new() }
    }
}

/// Create a new [`FillBetweenX`] with no options.
pub fn fill_betweenx<Y, X1, X2>(y: Y, x1: X1, x2: X2) -> FillBetweenX
where
    Y: IntoIterator<Item = f64>,
    X1: IntoIterator<Item = f64>,
    X2: IntoIterator<Item = f64>,
{
    FillBetweenX::new(y, x1, x2)
}

/// Create a new [`FillBetweenX`] with no options from a single iterator.
pub fn fill_betweenx_data<I>(data: I) -> FillBetweenX
where I: IntoIterator<Item = (f64, f64, f64)>
{
    FillBetweenX::new_data(data)
}

impl Matplotlib for FillBetweenX {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let y: Vec<Value> =
            self.y.iter().copied().map(Value::from).collect();
        let x1: Vec<Value> =
            self.x1.iter().copied().map(Value::from).collect();
        let x2: Vec<Value> =
            self.x2.iter().copied().map(Value::from).collect();
        Some(Value::Array(vec![y.into(), x1.into(), x2.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.fill_betweenx(data[0], data[1], data[2]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for FillBetweenX {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A horizontal line.
///
/// ```python
/// ax.axhline({y}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct AxHLine {
    /// Y-coordinate of the line.
    pub y: f64,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl AxHLine {
    /// Create a new `AxHLine` with no options.
    pub fn new(y: f64) -> Self {
        Self { y, opts: Vec::new() }
    }
}

/// Create a new [`AxHLine`] with no options.
pub fn axhline(y: f64) -> AxHLine { AxHLine::new(y) }

impl Matplotlib for AxHLine {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("ax.axhline({}{}{})",
            self.y,
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for AxHLine {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A vertical line.
///
/// ```python
/// ax.axvline({x}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct AxVLine {
    /// X-coordinate of the line.
    pub x: f64,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl AxVLine {
    /// Create a new `AxVLine` with no options.
    pub fn new(x: f64) -> Self {
        Self { x, opts: Vec::new() }
    }
}

/// Create a new [`AxVLine`] with no options.
pub fn axvline(x: f64) -> AxVLine { AxVLine::new(x) }

impl Matplotlib for AxVLine {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("ax.axvline({}{}{})",
            self.x,
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for AxVLine {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A line passing through two points.
///
/// ```python
/// ax.axline({xy1}, {xy2}, **{opts})
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct AxLine {
    /// First (*x*, *y*) point.
    pub xy1: (f64, f64),
    /// Second (*x*, *y*) point.
    pub xy2: (f64, f64),
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl AxLine {
    /// Create a new `AxLine` with no options.
    pub fn new(xy1: (f64, f64), xy2: (f64, f64)) -> Self {
        Self { xy1, xy2, opts: Vec::new() }
    }
}

/// Create a new [`AxLine`] with no options.
pub fn axline(xy1: (f64, f64), xy2: (f64, f64)) -> AxLine {
    AxLine::new(xy1, xy2)
}

impl Matplotlib for AxLine {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("ax.axline({:?}, {:?}{}{})",
            self.xy1,
            self.xy2,
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for AxLine {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A line passing through one point with a slope.
///
/// ```python
/// ax.axline({xy}, xy2=None, slope={m}, **{opts})
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct AxLineM {
    /// (*x*, *y*) point.
    pub xy: (f64, f64),
    /// Slope.
    pub m: f64,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl AxLineM {
    /// Create a new `AxLineM` with no options.
    pub fn new(xy: (f64, f64), m: f64) -> Self {
        Self { xy, m, opts: Vec::new() }
    }
}

/// Create a new [`AxLineM`] with no options.
pub fn axlinem(xy: (f64, f64), m: f64) -> AxLineM { AxLineM::new(xy, m) }

impl Matplotlib for AxLineM {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("ax.axline({:?}, xy2=None, slope={}{}{})",
            self.xy,
            self.m,
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for AxLineM {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A pie chart for a single data set.
///
/// ```python
/// ax.pie({data}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `list[float]`
#[derive(Clone, Debug, PartialEq)]
pub struct Pie {
    /// Data values.
    pub data: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Pie {
    /// Create a new `Pie` with no options.
    pub fn new<I>(data: I) -> Self
    where I: IntoIterator<Item = f64>
    {
        Self { data: data.into_iter().collect(), opts: Vec::new() }
    }
}

/// Create a new [`Pie`] with no options.
pub fn pie<I>(data: I) -> Pie
where I: IntoIterator<Item = f64>
{
    Pie::new(data)
}

impl Matplotlib for Pie {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        Some(Value::Array(
            self.data.iter().copied().map(Value::from).collect()))
    }

    fn py_cmd(&self) -> String {
        format!("ax.pie(data{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Pie {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Some text placed in a plot via data coordinates.
///
/// ```python
/// ax.text({x}, {y}, {s}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[float, float, str]`
///
/// See also [`AxText`].
#[derive(Clone, Debug, PartialEq)]
pub struct Text {
    /// X-coordinate.
    pub x: f64,
    /// Y-coordinate.
    pub y: f64,
    /// Text to place.
    pub s: String,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Text {
    /// Create a new `Text` with no options.
    pub fn new(x: f64, y: f64, s: &str) -> Self {
        Self { x, y, s: s.into(), opts: Vec::new() }
    }
}

/// Create a new [`Text`] with no options.
pub fn text(x: f64, y: f64, s: &str) -> Text { Text::new(x, y, s) }

impl Matplotlib for Text {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        Some(Value::Array(
                vec![self.x.into(), self.y.into(), (&*self.s).into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.text(data[0], data[1], data[2]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Text {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Some text placed in a plot via axes [0, 1] coordinates.
///
/// ```python
/// ax.text({x}, {y}, {s}, transform=ax.transAxes, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[float, float, str]`
#[derive(Clone, Debug, PartialEq)]
pub struct AxText {
    /// X-coordinate.
    pub x: f64,
    /// Y-coordinate.
    pub y: f64,
    /// Text to place.
    pub s: String,
    /// Option keyword arguments.
    pub opts: Vec<Opt>,
}

impl AxText {
    /// Create a new `AxText` with no options.
    pub fn new(x: f64, y: f64, s: &str) -> Self {
        Self { x, y, s: s.into(), opts: Vec::new() }
    }
}

/// Create a new [`AxText`] with no options.
pub fn axtext(x: f64, y: f64, s: &str) -> AxText { AxText::new(x, y, s) }

impl Matplotlib for AxText {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        Some(Value::Array(
                vec![self.x.into(), self.y.into(), (&*self.s).into()]))
    }

    fn py_cmd(&self) -> String {
        format!(
            "ax.text(data[0], data[1], data[2], transform=ax.transAxes{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for AxText {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Some text placed in a figure via figure [0, 1] coordinates.
///
/// ```python
/// fig.text({x}, {y}, {s}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[float, float, str]`
///
/// **Note** that this python command calls a method of the `fig` variable,
/// rather than `ax`.
#[derive(Clone, Debug, PartialEq)]
pub struct FigText {
    /// X-coordinate.
    pub x: f64,
    /// Y-coordinate.
    pub y: f64,
    /// Text to place.
    pub s: String,
    /// Option keyword arguments.
    pub opts: Vec<Opt>,
}

impl FigText {
    /// Create a new `FigText` with no options.
    pub fn new(x: f64, y: f64, s: &str) -> Self {
        Self { x, y, s: s.into(), opts: Vec::new() }
    }
}

/// Create a new [`FigText`] with no options.
pub fn figtext(x: f64, y: f64, s: &str) -> FigText { FigText::new(x, y, s) }

impl Matplotlib for FigText {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        Some(Value::Array(
                vec![self.x.into(), self.y.into(), (&*self.s).into()]))
    }

    fn py_cmd(&self) -> String {
        format!(
            "fig.text(data[0], data[1], data[2]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for FigText {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Add a colorbar to the figure.
///
/// This command relies on the local variable `im` being defined and set equal
/// to the output of a plotting command to which a color map can be applied
/// (e.g. [`Imshow`]). The output of this command is stored in a local variable
/// `cbar`.
///
/// ```python
/// cbar = fig.colorbar(im, ax=ax, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct Colorbar {
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Default for Colorbar {
    fn default() -> Self { Self::new() }
}

impl Colorbar {
    /// Create a new `Colorbar` with no options.
    pub fn new() -> Self { Self { opts: Vec::new() } }
}

/// Create a new [`Colorbar`] with no options.
pub fn colorbar() -> Colorbar { Colorbar::new() }

impl Matplotlib for Colorbar {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("cbar = fig.colorbar(im, ax=ax{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Colorbar {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Set the scaling of an axis.
///
/// ```python
/// ax.set_{axis}scale("{scale}")
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Scale {
    /// Which axis to scale.
    pub axis: Axis,
    /// What scaling to use.
    pub scale: AxisScale,
}

impl Scale {
    /// Create a new `Scale`.
    pub fn new(axis: Axis, scale: AxisScale) -> Self { Self { axis, scale } }
}

/// Create a new [`Scale`].
pub fn scale(axis: Axis, scale: AxisScale) -> Scale { Scale::new(axis, scale) }

/// Create a new [`Scale`] for the X-axis.
pub fn xscale(scale: AxisScale) -> Scale { Scale::new(Axis::X, scale) }

/// Create a new [`Scale`] for the Y-axis.
pub fn yscale(scale: AxisScale) -> Scale { Scale::new(Axis::Y, scale) }

/// Create a new [`Scale`] for the Z-axis.
pub fn zscale(scale: AxisScale) -> Scale { Scale::new(Axis::Z, scale) }

impl Matplotlib for Scale {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        let ax = format!("{:?}", self.axis).to_lowercase();
        let sc = format!("{:?}", self.scale).to_lowercase();
        format!("ax.set_{}scale(\"{}\")", ax, sc)
    }
}

/// An axis of a Matplotlib `Axes` or `Axes3D` object.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Axis {
    /// The X-axis.
    X,
    /// The Y-axis.
    Y,
    /// The Z-axis.
    Z,
}

/// An axis scaling.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AxisScale {
    /// Linear scaling.
    Linear,
    /// Logarithmic scaling.
    Log,
    /// Symmetric logarithmic scaling.
    ///
    /// Allows for negative values by scaling their absolute values.
    SymLog,
    /// Scaling through the logit function.
    ///
    /// ```python
    /// logit(x) = log(x / (1 - x))
    /// ```
    ///
    /// Specifically designed for values in the (0, 1) range.
    Logit,
}

/// Set the plotting limits of an axis.
///
/// ```python
/// ax.set_{axis}lim({min}, {max})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Lim {
    /// Which axis.
    pub axis: Axis,
    /// Minimum value.
    ///
    /// Pass `None` to auto-set.
    pub min: Option<f64>,
    /// Maximum value.
    ///
    /// Pass `None` to auto-set.
    pub max: Option<f64>,
}

impl Lim {
    /// Create a new `Lim`.
    pub fn new(axis: Axis, min: Option<f64>, max: Option<f64>) -> Self {
        Self { axis, min, max }
    }
}

/// Create a new [`Lim`].
pub fn lim(axis: Axis, min: Option<f64>, max: Option<f64>) -> Lim {
    Lim::new(axis, min, max)
}

/// Create a new [`Lim`] for the X-axis.
pub fn xlim(min: Option<f64>, max: Option<f64>) -> Lim {
    Lim::new(Axis::X, min, max)
}

/// Create a new [`Lim`] for the Y-axis.
pub fn ylim(min: Option<f64>, max: Option<f64>) -> Lim {
    Lim::new(Axis::Y, min, max)
}

/// Create a new [`Lim`] for the Z-axis.
pub fn zlim(min: Option<f64>, max: Option<f64>) -> Lim {
    Lim::new(Axis::Z, min, max)
}

impl Matplotlib for Lim {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        let ax = format!("{:?}", self.axis).to_lowercase();
        let min =
            self.min.as_ref()
            .map(|x| format!("{}", x))
            .unwrap_or("None".into());
        let max =
            self.max.as_ref()
            .map(|x| format!("{}", x))
            .unwrap_or("None".into());
        format!("ax.set_{}lim({}, {})", ax, min, max)
    }
}

/// Set the plotting limits of the colorbar.
///
/// This relies on an existing local variable `im` produced by e.g. [`Imshow`].
///
/// ```python
/// im.set_clim({min}, {max})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CLim {
    /// Minimum value.
    ///
    /// Pass `None` to auto-set.
    pub min: Option<f64>,
    /// Maximum value.
    ///
    /// Pass `None` to auto-set.
    pub max: Option<f64>,
}

impl CLim {
    /// Create a new `CLim`.
    pub fn new(min: Option<f64>, max: Option<f64>) -> Self {
        Self { min, max }
    }
}

/// Create a new [`CLim`].
pub fn clim(min: Option<f64>, max: Option<f64>) -> CLim { CLim::new(min, max) }

impl Matplotlib for CLim {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        let min =
            self.min.as_ref()
            .map(|x| format!("{}", x))
            .unwrap_or("None".into());
        let max =
            self.max.as_ref()
            .map(|x| format!("{}", x))
            .unwrap_or("None".into());
        format!("im.set_clim({}, {})", min, max)
    }
}

/// Set the title of a set of axes.
///
/// ```python
/// ax.set_title("{s}", **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct Title {
    /// Axes title.
    pub s: String,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Title {
    /// Create a new `Title` with no options.
    pub fn new(s: &str) -> Self {
        Self { s: s.into(), opts: Vec::new() }
    }
}

/// Create a new [`Title`] with no options.
pub fn title(s: &str) -> Title { Title::new(s) }

impl Matplotlib for Title {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("ax.set_title(\"{}\"{}{})",
            self.s,
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Title {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Set a label on a set of axes.
///
/// ```python
/// ax.set_xlabel("{s}", **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    /// Which axis to label.
    pub axis: Axis,
    /// Axis label.
    pub s: String,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Label {
    /// Create a new `Label` with no options.
    pub fn new(axis: Axis, s: &str) -> Self {
        Self { axis, s: s.into(), opts: Vec::new() }
    }
}

/// Create a new [`Label`] with no options.
pub fn label(axis: Axis, s: &str) -> Label { Label::new(axis, s) }

/// Create a new [`Label`] for the X-axis with no options.
pub fn xlabel(s: &str) -> Label { Label::new(Axis::X, s) }

/// Create a new [`Label`] for the Y-axis with no options.
pub fn ylabel(s: &str) -> Label { Label::new(Axis::Y, s) }

/// Create a new [`Label`] for the Z-axis with no options.
pub fn zlabel(s: &str) -> Label { Label::new(Axis::Z, s) }

impl Matplotlib for Label {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        let ax = format!("{:?}", self.axis).to_lowercase();
        format!("ax.set_{}label(\"{}\"{}{})",
            ax,
            self.s,
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Label {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Set a label on a colorbar.
///
/// This relies on an existing local variable `cbar` produced by e.g.
/// [`Colorbar`].
///
/// ```python
/// cbar.set_label("{s}", **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct CLabel {
    /// Colorbar label.
    pub s: String,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl CLabel {
    /// Create a new `CLabel` with no options.
    pub fn new(s: &str) -> Self {
        Self { s: s.into(), opts: Vec::new() }
    }
}

/// Create a new [`CLabel`] with no options.
pub fn clabel(s: &str) -> CLabel { CLabel::new(s) }

impl Matplotlib for CLabel {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("cbar.set_label(\"{}\"{}{})",
            self.s,
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for CLabel {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Set the values for which ticks are placed on an axis.
///
/// ```python
/// ax.set_{axis}ticks({v}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `list[float]`
#[derive(Clone, Debug, PartialEq)]
pub struct Ticks {
    /// Which axis.
    pub axis: Axis,
    /// Tick values.
    pub v: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Ticks {
    /// Create a new `Ticks` with no options.
    pub fn new<I>(axis: Axis, v: I) -> Self
    where I: IntoIterator<Item = f64>
    {
        Self { axis, v: v.into_iter().collect(), opts: Vec::new() }
    }
}

/// Create a new [`Ticks`] with no options.
pub fn ticks<I>(axis: Axis, v: I) -> Ticks
where I: IntoIterator<Item = f64>
{
    Ticks::new(axis, v)
}

/// Create a new [`Ticks`] for the X-axis with no options.
pub fn xticks<I>(v: I) -> Ticks
where I: IntoIterator<Item = f64>
{
    Ticks::new(Axis::X, v)
}

/// Create a new [`Ticks`] for the Y-axis with no options.
pub fn yticks<I>(v: I) -> Ticks
where I: IntoIterator<Item = f64>
{
    Ticks::new(Axis::Y, v)
}

/// Create a new [`Ticks`] for the Z-axis with no options.
pub fn zticks<I>(v: I) -> Ticks
where I: IntoIterator<Item = f64>
{
    Ticks::new(Axis::Z, v)
}

impl Matplotlib for Ticks {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let v: Vec<Value> = self.v.iter().copied().map(Value::from).collect();
        Some(Value::Array(v))
    }

    fn py_cmd(&self) -> String {
        format!("ax.set_{}ticks(data{}{})",
            format!("{:?}", self.axis).to_lowercase(),
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Ticks {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Set the values for which ticks are placed on a colorbar.
///
/// This relies on an existing local variable `cbar` produced by e.g.
/// [`Colorbar`].
///
/// ```python
/// cbar.set_ticks({v}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `list[float]`
#[derive(Clone, Debug, PartialEq)]
pub struct CTicks {
    /// Tick values.
    pub v: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl CTicks {
    /// Create a new `CTicks` with no options.
    pub fn new<I>(v: I) -> Self
    where I: IntoIterator<Item = f64>
    {
        Self { v: v.into_iter().collect(), opts: Vec::new() }
    }
}

/// Create a new [`CTicks`] with no options.
pub fn cticks<I>(v: I) -> CTicks
where I: IntoIterator<Item = f64>
{
    CTicks::new(v)
}

impl Matplotlib for CTicks {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let v: Vec<Value> =
            self.v.iter().copied().map(Value::from).collect();
        Some(Value::Array(v))
    }

    fn py_cmd(&self) -> String {
        format!("cbar.set_ticks(data{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for CTicks {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Set the values and labels for which ticks are placed on an axis.
///
/// ```python
/// ax.set_{axis}ticks({v}, labels={s}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[str]]`.
#[derive(Clone, Debug, PartialEq)]
pub struct TickLabels {
    /// Which axis.
    pub axis: Axis,
    /// Tick values.
    pub v: Vec<f64>,
    /// Tick labels.
    pub s: Vec<String>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl TickLabels {
    /// Create a new `TickLabels` with no options.
    pub fn new<I, J, S>(axis: Axis, v: I, s: J) -> Self
    where
        I: IntoIterator<Item = f64>,
        J: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            axis,
            v: v.into_iter().collect(),
            s: s.into_iter().map(|sk| sk.into()).collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `TickLabels` with no options from a single iterator.
    pub fn new_data<I, S>(axis: Axis, ticklabels: I) -> Self
    where
        I: IntoIterator<Item = (f64, S)>,
        S: Into<String>,
    {
        let (v, s): (Vec<f64>, Vec<String>) =
            ticklabels.into_iter()
            .map(|(vk, sk)| (vk, sk.into()))
            .unzip();
        Self { axis, v, s, opts: Vec::new() }
    }
}

/// Create a new [`TickLabels`] with no options.
pub fn ticklabels<I, J, S>(axis: Axis, v: I, s: J) -> TickLabels
where
    I: IntoIterator<Item = f64>,
    J: IntoIterator<Item = S>,
    S: Into<String>,
{
    TickLabels::new(axis, v, s)
}

/// Create a new [`TickLabels`] with no options from a single iterator.
pub fn ticklabels_data<I, S>(axis: Axis, ticklabels: I) -> TickLabels
where
    I: IntoIterator<Item = (f64, S)>,
    S: Into<String>,
{
    TickLabels::new_data(axis, ticklabels)
}

/// Create a new [`TickLabels`] for the X-axis with no options.
pub fn xticklabels<I, J, S>(v: I, s: J) -> TickLabels
where
    I: IntoIterator<Item = f64>,
    J: IntoIterator<Item = S>,
    S: Into<String>,
{
    TickLabels::new(Axis::X, v, s)
}

/// Create a new [`TickLabels`] for the X-axis with no options from a single
/// iterator.
pub fn xticklabels_data<I, S>(ticklabels: I) -> TickLabels
where
    I: IntoIterator<Item = (f64, S)>,
    S: Into<String>,
{
    TickLabels::new_data(Axis::X, ticklabels)
}

/// Create a new [`TickLabels`] for the Y-axis with no options.
pub fn yticklabels<I, J, S>(v: I, s: J) -> TickLabels
where
    I: IntoIterator<Item = f64>,
    J: IntoIterator<Item = S>,
    S: Into<String>,
{
    TickLabels::new(Axis::Y, v, s)
}

/// Create a new [`TickLabels`] for the Y-axis with no options from a single
/// iterator.
pub fn yticklabels_data<I, S>(ticklabels: I) -> TickLabels
where
    I: IntoIterator<Item = (f64, S)>,
    S: Into<String>,
{
    TickLabels::new_data(Axis::Y, ticklabels)
}

/// Create a new [`TickLabels`] for the Z-axis with no options.
pub fn zticklabels<I, J, S>(v: I, s: J) -> TickLabels
where
    I: IntoIterator<Item = f64>,
    J: IntoIterator<Item = S>,
    S: Into<String>,
{
    TickLabels::new(Axis::Z, v, s)
}

/// Create a new [`TickLabels`] for the Z-axis with no options from a single
/// iterator.
pub fn zticklabels_data<I, S>(ticklabels: I) -> TickLabels
where
    I: IntoIterator<Item = (f64, S)>,
    S: Into<String>,
{
    TickLabels::new_data(Axis::Z, ticklabels)
}

impl Matplotlib for TickLabels {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let v: Vec<Value> = self.v.iter().copied().map(Value::from).collect();
        let s: Vec<Value> = self.s.iter().cloned().map(Value::from).collect();
        Some(Value::Array(vec![v.into(), s.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.set_{}ticks(data[0], labels=data[1]{}{})",
            format!("{:?}", self.axis).to_lowercase(),
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for TickLabels {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Set the values and labels for which ticks are placed on a colorbar.
///
/// This relies on an existing local variable `cbar` produced by e.g.
/// [`Colorbar`].
///
/// ```python
/// cbar.set_ticks({v}, labels={s}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[str]]`
#[derive(Clone, Debug, PartialEq)]
pub struct CTickLabels {
    /// Tick values.
    pub v: Vec<f64>,
    /// Tick labels.
    pub s: Vec<String>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl CTickLabels {
    /// Create a new `CTickLabels` with no options.
    pub fn new<I, J, S>(v: I, s: J) -> Self
    where
        I: IntoIterator<Item = f64>,
        J: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            v: v.into_iter().collect(),
            s: s.into_iter().map(|sk| sk.into()).collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `CTickLabels` with no options from a single iterator.
    pub fn new_data<I, S>(ticklabels: I) -> Self
    where
        I: IntoIterator<Item = (f64, S)>,
        S: Into<String>,
    {
        let (v, s): (Vec<f64>, Vec<String>) =
            ticklabels.into_iter()
            .map(|(vk, sk)| (vk, sk.into()))
            .unzip();
        Self { v, s, opts: Vec::new() }
    }
}

/// Create a new [`CTickLabels`] with no options.
pub fn cticklabels<I, J, S>(v: I, s: J) -> CTickLabels
where
    I: IntoIterator<Item = f64>,
    J: IntoIterator<Item = S>,
    S: Into<String>,
{
    CTickLabels::new(v, s)
}

/// Create a new [`CTickLabels`] with no options from a single iterator.
pub fn cticklabels_data<I, S>(ticklabels: I) -> CTickLabels
where
    I: IntoIterator<Item = (f64, S)>,
    S: Into<String>,
{
    CTickLabels::new_data(ticklabels)
}

impl Matplotlib for CTickLabels {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let v: Vec<Value> =
            self.v.iter().copied().map(Value::from).collect();
        let s: Vec<Value> =
            self.s.iter().cloned().map(Value::from).collect();
        Some(Value::Array(vec![v.into(), s.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("cbar.set_ticks(data[0], labels=data[1]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for CTickLabels {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Set the appearance of ticks, tick labels, and gridlines.
///
/// ```python
/// ax.tick_params({axis}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct TickParams {
    /// Which axis.
    pub axis: Axis2,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl TickParams {
    /// Create a new `TickParams` with no options.
    pub fn new(axis: Axis2) -> Self {
        Self { axis, opts: Vec::new() }
    }
}

/// Create a new [`TickParams`] with no options.
pub fn tick_params(axis: Axis2) -> TickParams { TickParams::new(axis) }

/// Create a new [`TickParams`] for the X-axis with no options.
pub fn xtick_params() -> TickParams { TickParams::new(Axis2::X) }

/// Create a new [`TickParams`] for the Y-axis with no options.
pub fn ytick_params() -> TickParams { TickParams::new(Axis2::Y) }

impl Matplotlib for TickParams {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("ax.tick_params(\"{}\"{}{})",
            format!("{:?}", self.axis).to_lowercase(),
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for TickParams {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Like [`Axis`], but limited to X or Y and with the option of both.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Axis2 {
    /// The X-axis.
    X,
    /// The Y-axis.
    Y,
    /// Both the X- and Y-axes.
    Both,
}

/// Set the title of the figure.
///
/// ```python
/// fig.suptitle({s}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct SupTitle {
    /// Figure title.
    pub s: String,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl SupTitle {
    /// Create a new `SupTitle` with no options.
    pub fn new(s: &str) -> Self {
        Self { s: s.into(), opts: Vec::new() }
    }
}

/// Create a new [`SupTitle`] with no options.
pub fn suptitle(s: &str) -> SupTitle { SupTitle::new(s) }

impl Matplotlib for SupTitle {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("fig.suptitle(\"{}\"{}{})",
            self.s,
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for SupTitle {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Set the X label of the figure.
///
/// ```python
/// fig.supxlabel({s}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct SupXLabel {
    /// Figure X label.
    pub s: String,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl SupXLabel {
    /// Create a new `SupXLabel` with no options.
    pub fn new(s: &str) -> Self {
        Self { s: s.into(), opts: Vec::new() }
    }
}

/// Create a new [`SupXLabel`] with no options.
pub fn supxlabel(s: &str) -> SupXLabel { SupXLabel::new(s) }

impl Matplotlib for SupXLabel {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("fig.supxlabel(\"{}\"{}{})",
            self.s,
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for SupXLabel {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Set the title of the figure.
///
/// ```python
/// fig.supylabel({s}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct SupYLabel {
    /// Figure title.
    pub s: String,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl SupYLabel {
    /// Create a new `SupYLabel` with no options.
    pub fn new(s: &str) -> Self {
        Self { s: s.into(), opts: Vec::new() }
    }
}

/// Create a new [`SupYLabel`] with no options.
pub fn supylabel(s: &str) -> SupYLabel { SupYLabel::new(s) }

impl Matplotlib for SupYLabel {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("fig.supylabel(\"{}\"{}{})",
            self.s,
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for SupYLabel {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Place a legend on a set of axes.
///
/// ```python
/// ax.legend(**{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct Legend {
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Default for Legend {
    fn default() -> Self { Self::new() }
}

impl Legend {
    /// Create a new `Legend` with no options.
    pub fn new() -> Self {
        Self { opts: Vec::new() }
    }
}

/// Create a new [`Legend`] with no options.
pub fn legend() -> Legend { Legend::new() }

impl Matplotlib for Legend {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("ax.legend({})", self.opts.as_py())
    }
}

impl MatplotlibOpts for Legend {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Activate or modify the coordinate grid.
///
/// ```python
/// ax.grid({onoff}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    /// On/off setting.
    pub onoff: bool,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Grid {
    /// Create a new `Grid` with no options.
    pub fn new(onoff: bool) -> Self { Self { onoff, opts: Vec::new() } }
}

/// Create a new [`Grid`] with no options.
pub fn grid(onoff: bool) -> Grid { Grid::new(onoff) }

impl Matplotlib for Grid {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("ax.grid({}, {})", self.onoff.as_py(), self.opts.as_py())
    }
}

impl MatplotlibOpts for Grid {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Adjust the padding between and around subplots.
///
/// ```python
/// fig.tight_layout(**{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct TightLayout {
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Default for TightLayout {
    fn default() -> Self { Self::new() }
}

impl TightLayout {
    /// Create a new `TightLayout` with no options.
    pub fn new() -> Self { Self { opts: Vec::new() } }
}

/// Create a new [`TightLayout`] with no options.
pub fn tight_layout() -> TightLayout { TightLayout::new() }

impl Matplotlib for TightLayout {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("fig.tight_layout({})", self.opts.as_py())
    }
}

impl MatplotlibOpts for TightLayout {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Create and refocus to a set of axes inset to `ax`.
///
/// Coordinates and sizes are in axis [0, 1] units.
///
/// ```python
/// ax = ax.inset_axes([{x}, {y}, {w}, {h}], **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct InsetAxes {
    /// X-coordinate of the lower-left corner of the inset.
    pub x: f64,
    /// Y-coordinate of the lower-left cordiner of the inset.
    pub y: f64,
    /// Width of the inset.
    pub w: f64,
    /// Height of the inset.
    pub h: f64,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl InsetAxes {
    /// Create a new `InsetAxes` with no options.
    pub fn new(x: f64, y: f64, w: f64, h: f64) -> Self {
        Self { x, y, w, h, opts: Vec::new() }
    }

    /// Create a new `InsetAxes` with no options from (*x*, *y*) and (*width*,
    /// *height*) pairs.
    pub fn new_pairs(xy: (f64, f64), wh: (f64, f64)) -> Self {
        Self { x: xy.0, y: xy.1, w: wh.0, h: wh.1, opts: Vec::new() }
    }
}

/// Create a new [`InsetAxes`] with no options.
pub fn inset_axes(x: f64, y: f64, w: f64, h: f64) -> InsetAxes {
    InsetAxes::new(x, y, w, h)
}

/// Create a new [`InsetAxes`] with no options from (*x*, *y*) and (*width*,
/// *height*) pairs.
pub fn inset_axes_pairs(xy: (f64, f64), wh: (f64, f64)) -> InsetAxes {
    InsetAxes::new_pairs(xy, wh)
}

impl Matplotlib for InsetAxes {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("ax = ax.inset_axes([{}, {}, {}, {}]{}{})",
            self.x,
            self.y,
            self.w,
            self.h,
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for InsetAxes {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A (*x*, *y*, *z*) plot.
///
/// ```python
/// ax.plot({x}, {y}, {z}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float], list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Plot3 {
    /// X-coordinates.
    pub x: Vec<f64>,
    /// Y-coordinates.
    pub y: Vec<f64>,
    /// Z-coordinates.
    pub z: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Plot3 {
    /// Create a new `Plot3` with no options.
    pub fn new<X, Y, Z>(x: X, y: Y, z: Z) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
        Z: IntoIterator<Item = f64>,
    {
        Self {
            x: x.into_iter().collect(),
            y: y.into_iter().collect(),
            z: z.into_iter().collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `Plot3` with no options from a single iterator.
    pub fn new_data<I>(data: I) -> Self
    where I: IntoIterator<Item = (f64, f64, f64)>
    {
        let ((x, y), z) = data.into_iter().map(assoc).unzip();
        Self { x, y, z, opts: Vec::new() }
    }
}

/// Create a new [`Plot3`] with no options.
pub fn plot3<X, Y, Z>(x: X, y: Y, z: Z) -> Plot3
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
    Z: IntoIterator<Item = f64>,
{
    Plot3::new(x, y, z)
}

/// Create a new [`Plot3`] with no options from a single iterator.
pub fn plot3_data<I>(data: I) -> Plot3
where I: IntoIterator<Item = (f64, f64, f64)>
{
    Plot3::new_data(data)
}

impl Matplotlib for Plot3 {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> =
            self.x.iter().copied().map(Value::from).collect();
        let y: Vec<Value> =
            self.y.iter().copied().map(Value::from).collect();
        let z: Vec<Value> =
            self.z.iter().copied().map(Value::from).collect();
        Some(Value::Array(vec![x.into(), y.into(), z.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.plot(data[0], data[1], data[2]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Plot3 {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A (*x*, *y*, *z*) scatter plot.
///
/// ```python
/// ax.scatter({x}, {y}, {z}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float], list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Scatter3 {
    /// X-coordinates.
    pub x: Vec<f64>,
    /// Y-coordinates.
    pub y: Vec<f64>,
    /// Z-coordinates.
    pub z: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Scatter3 {
    /// Create a new `Scatter3` with no options.
    pub fn new<X, Y, Z>(x: X, y: Y, z: Z) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
        Z: IntoIterator<Item = f64>,
    {
        Self {
            x: x.into_iter().collect(),
            y: y.into_iter().collect(),
            z: z.into_iter().collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `Scatter3` with no options from a single iterator.
    pub fn new_data<I>(data: I) -> Self
    where I: IntoIterator<Item = (f64, f64, f64)>
    {
        let ((x, y), z) = data.into_iter().map(assoc).unzip();
        Self { x, y, z, opts: Vec::new() }
    }
}

/// Create a new [`Scatter3`] with no options.
pub fn scatter3<X, Y, Z>(x: X, y: Y, z: Z) -> Scatter3
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
    Z: IntoIterator<Item = f64>,
{
    Scatter3::new(x, y, z)
}

/// Create a new [`Scatter3`] with no options from a single iterator.
pub fn scatter3_data<I>(data: I) -> Scatter3
where I: IntoIterator<Item = (f64, f64, f64)>
{
    Scatter3::new_data(data)
}

impl Matplotlib for Scatter3 {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> =
            self.x.iter().copied().map(Value::from).collect();
        let y: Vec<Value> =
            self.y.iter().copied().map(Value::from).collect();
        let z: Vec<Value> =
            self.z.iter().copied().map(Value::from).collect();
        Some(Value::Array(vec![x.into(), y.into(), z.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.scatter(data[0], data[1], data[2]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Scatter3 {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A 3D vector field plot.
///
/// ```python
/// ax.quiver({x}, {y}, {z}, {vx}, {vy}, {vz}, **{ops})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float], list[float], list[float], list[float], list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Quiver3 {
    /// X-coordinates.
    pub x: Vec<f64>,
    /// Y-coordinates.
    pub y: Vec<f64>,
    /// Z-coordinates.
    pub z: Vec<f64>,
    /// Vector X-components.
    pub vx: Vec<f64>,
    /// Vector Y-components.
    pub vy: Vec<f64>,
    /// Vector Z-components.
    pub vz: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Quiver3 {
    /// Create a new `Quiver3` with no options.
    pub fn new<X, Y, Z, VX, VY, VZ>(x: X, y: Y, z: Z, vx: VX, vy: VY, vz: VZ)
        -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
        Z: IntoIterator<Item = f64>,
        VX: IntoIterator<Item = f64>,
        VY: IntoIterator<Item = f64>,
        VZ: IntoIterator<Item = f64>,
    {
        Self {
            x: x.into_iter().collect(),
            y: y.into_iter().collect(),
            z: z.into_iter().collect(),
            vx: vx.into_iter().collect(),
            vy: vy.into_iter().collect(),
            vz: vz.into_iter().collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `Quiver3` with no options from iterators over coordinate
    /// triples.
    pub fn new_triples<I, VI>(xyz: I, vxyz: VI) -> Self
    where
        I: IntoIterator<Item = (f64, f64, f64)>,
        VI: IntoIterator<Item = (f64, f64, f64)>,
    {
        let ((x, y), z) = xyz.into_iter().map(assoc).unzip();
        let ((vx, vy), vz) = vxyz.into_iter().map(assoc).unzip();
        Self { x, y, z, vx, vy, vz, opts: Vec::new() }
    }

    /// Create a new `Quiver3` with no options from a single iterator. The first
    /// three elements of each iterator item should be spatial coordinates and
    /// the last three should be vector components.
    pub fn new_data<I>(data: I) -> Self
    where I: IntoIterator<Item = (f64, f64, f64, f64, f64, f64)>
    {
        let (((((x, y), z), vx), vy), vz) = data.into_iter().map(assoc).unzip();
        Self { x, y, z, vx, vy, vz, opts: Vec::new() }
    }
}

/// Create a new [`Quiver3`] with no options.
pub fn quiver3<X, Y, Z, VX, VY, VZ>(x: X, y: Y, z: Z, vx: VX, vy: VY, vz: VZ)
    -> Quiver3
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
    Z: IntoIterator<Item = f64>,
    VX: IntoIterator<Item = f64>,
    VY: IntoIterator<Item = f64>,
    VZ: IntoIterator<Item = f64>,
{
    Quiver3::new(x, y, z, vx, vy, vz)
}

/// Create a new [`Quiver3`] with no options from iterators over coordinate
/// triples.
pub fn quiver3_triples<I, VI>(xyz: I, vxyz: VI) -> Quiver3
where
    I: IntoIterator<Item = (f64, f64, f64)>,
    VI: IntoIterator<Item = (f64, f64, f64)>,
{
    Quiver3::new_triples(xyz, vxyz)
}

/// Create a new [`Quiver3`] with no options from a single iterator.
///
/// The first three elements of each iterator item should be spatial coordinates
/// and the last three should be vector components.
pub fn quiver3_data<I>(data: I) -> Quiver3
where I: IntoIterator<Item = (f64, f64, f64, f64, f64, f64)>
{
    Quiver3::new_data(data)
}

impl Matplotlib for Quiver3 {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> = self.x.iter().copied().map(Value::from).collect();
        let y: Vec<Value> = self.y.iter().copied().map(Value::from).collect();
        let z: Vec<Value> = self.z.iter().copied().map(Value::from).collect();
        let vx: Vec<Value> = self.vx.iter().copied().map(Value::from).collect();
        let vy: Vec<Value> = self.vy.iter().copied().map(Value::from).collect();
        let vz: Vec<Value> = self.vz.iter().copied().map(Value::from).collect();
        Some(Value::Array(vec![
                x.into(), y.into(), z.into(), vx.into(), vy.into(), vz.into()]))
    }

    fn py_cmd(&self) -> String {
        format!(
            "ax.quiver(\
            data[0], data[1], data[2], data[3], data[4], data[5]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Quiver3 {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A 3D surface plot.
///
/// ```python
/// ax.plot_surface({x}, {y}, {z}, **{opts})
/// ```
///
/// **Note**: `plot_surface` requires input to be in the form of a NumPy array.
/// Therefore, this command requires that NumPy be imported under the usual
/// name, `np`.
///
/// Prelude: **No**
///
/// JSON data: `[list[list[float]], list[list[float]], list[list[float]]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Surface {
    /// X-coordinates.
    pub x: Vec<Vec<f64>>,
    /// Y-coordinates.
    pub y: Vec<Vec<f64>>,
    /// Z-coordinates.
    pub z: Vec<Vec<f64>>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Surface {
    /// Create a new `Surface` with no options.
    pub fn new<XI, XJ, YI, YJ, ZI, ZJ>(x: XI, y: YI, z: ZI) -> Self
    where
        XI: IntoIterator<Item = XJ>,
        XJ: IntoIterator<Item = f64>,
        YI: IntoIterator<Item = YJ>,
        YJ: IntoIterator<Item = f64>,
        ZI: IntoIterator<Item = ZJ>,
        ZJ: IntoIterator<Item = f64>,
    {
        let x: Vec<Vec<f64>> =
            x.into_iter()
            .map(|row| row.into_iter().collect())
            .collect();
        let y: Vec<Vec<f64>> =
            y.into_iter()
            .map(|row| row.into_iter().collect())
            .collect();
        let z: Vec<Vec<f64>> =
            z.into_iter()
            .map(|row| row.into_iter().collect())
            .collect();
        Self { x, y, z, opts: Vec::new() }
    }

    /// Create a new `Surface` from flattened, column-major iterators over
    /// coordinate data with row length `rowlen`.
    ///
    /// *Panics if `rowlen == 0`*.
    pub fn new_flat<X, Y, Z>(x: X, y: Y, z: Z, rowlen: usize) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
        Z: IntoIterator<Item = f64>,
    {
        if rowlen == 0 { panic!("row length cannot be zero"); }
        let x: Vec<Vec<f64>> =
            Chunks::new(x.into_iter(), rowlen)
            .collect();
        let y: Vec<Vec<f64>> =
            Chunks::new(y.into_iter(), rowlen)
            .collect();
        let z: Vec<Vec<f64>> =
            Chunks::new(z.into_iter(), rowlen)
            .collect();
        Self { x, y, z, opts: Vec::new() }
    }

    /// Create a new `Surface` from a single flattened, row-major iterator over
    /// coordinate data with row length `rowlen`.
    ///
    /// *Panics if `rowlen == 0`*.
    pub fn new_data<I>(data: I, rowlen: usize) -> Self
    where I: IntoIterator<Item = (f64, f64, f64)>
    {
        if rowlen == 0 { panic!("row length cannot be zero"); }
        let mut x: Vec<Vec<f64>> = Vec::new();
        let mut y: Vec<Vec<f64>> = Vec::new();
        let mut z: Vec<Vec<f64>> = Vec::new();
        Chunks::new(data.into_iter(), rowlen)
            .for_each(|points| {
                let mut xi: Vec<f64> = Vec::with_capacity(rowlen);
                let mut yi: Vec<f64> = Vec::with_capacity(rowlen);
                let mut zi: Vec<f64> = Vec::with_capacity(rowlen);
                points.into_iter()
                    .for_each(|(xij, yij, zij)| {
                        xi.push(xij); yi.push(yij); zi.push(zij);
                    });
                x.push(xi); y.push(yi); z.push(zi);
            });
        Self { x, y, z, opts: Vec::new() }
    }
}

/// Create a new [`Surface`] with no options.
pub fn surface<XI, XJ, YI, YJ, ZI, ZJ>(x: XI, y: YI, z: ZI) -> Surface
where
    XI: IntoIterator<Item = XJ>,
    XJ: IntoIterator<Item = f64>,
    YI: IntoIterator<Item = YJ>,
    YJ: IntoIterator<Item = f64>,
    ZI: IntoIterator<Item = ZJ>,
    ZJ: IntoIterator<Item = f64>,
{
    Surface::new(x, y, z)
}

/// Create a new [`Surface`] from flattened, column-major iterators over
/// coordinate data with row length `rowlen`.
///
/// *Panics if `rowlen == 0`*.
pub fn surface_flat<X, Y, Z>(x: X, y: Y, z: Z, rowlen: usize) -> Surface
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
    Z: IntoIterator<Item = f64>,
{
    Surface::new_flat(x, y, z, rowlen)
}

/// Create a new [`Surface`] from a single flattened, row-major iterator over
/// coordinate data with row length `rowlen`.
///
/// *Panics if `rowlen == 0`*.
pub fn surface_data<I>(data: I, rowlen: usize) -> Surface
where I: IntoIterator<Item = (f64, f64, f64)>
{
    Surface::new_data(data, rowlen)
}

impl Matplotlib for Surface {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> =
            self.x.iter()
            .map(|row| {
                let row: Vec<Value> =
                    row.iter().copied().map(Value::from).collect();
                Value::Array(row)
            })
            .collect();
        let y: Vec<Value> =
            self.y.iter()
            .map(|row| {
                let row: Vec<Value> =
                    row.iter().copied().map(Value::from).collect();
                Value::Array(row)
            })
            .collect();
        let z: Vec<Value> =
            self.z.iter()
            .map(|row| {
                let row: Vec<Value> =
                    row.iter().copied().map(Value::from).collect();
                Value::Array(row)
            })
            .collect();
        Some(Value::Array(vec![x.into(), y.into(), z.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("\
            ax.plot_surface(\
            np.array(data[0]), \
            np.array(data[1]), \
            np.array(data[2])\
            {}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Surface {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// A 3D surface plot using triangulation.
///
/// ```python
/// ax.plot_trisurf({x}, {y}, {z}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: `[list[float], list[float], list[float]]`
#[derive(Clone, Debug, PartialEq)]
pub struct Trisurf {
    /// X-coordinates.
    pub x: Vec<f64>,
    /// Y-coordinates.
    pub y: Vec<f64>,
    /// Z-coordinates.
    pub z: Vec<f64>,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl Trisurf {
    /// Create a new `Trisurf` with no options.
    pub fn new<X, Y, Z>(x: X, y: Y, z: Z) -> Self
    where
        X: IntoIterator<Item = f64>,
        Y: IntoIterator<Item = f64>,
        Z: IntoIterator<Item = f64>,
    {
        Self {
            x: x.into_iter().collect(),
            y: y.into_iter().collect(),
            z: z.into_iter().collect(),
            opts: Vec::new(),
        }
    }

    /// Create a new `Trisurf` with no options from a single iterator.
    pub fn new_data<I>(data: I) -> Self
    where I: IntoIterator<Item = (f64, f64, f64)>
    {
        let ((x, y), z) = data.into_iter().map(assoc).unzip();
        Self { x, y, z, opts: Vec::new() }
    }
}

/// Create a new [`Trisurf`] with no options.
pub fn trisurf<X, Y, Z>(x: X, y: Y, z: Z) -> Trisurf
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
    Z: IntoIterator<Item = f64>,
{
    Trisurf::new(x, y, z)
}

/// Create a new [`Trisurf`] with no options from a single iterator.
pub fn trisurf_data<I>(data: I) -> Trisurf
where I: IntoIterator<Item = (f64, f64, f64)>
{
    Trisurf::new_data(data)
}

impl Matplotlib for Trisurf {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> {
        let x: Vec<Value> = self.x.iter().copied().map(Value::from).collect();
        let y: Vec<Value> = self.y.iter().copied().map(Value::from).collect();
        let z: Vec<Value> = self.z.iter().copied().map(Value::from).collect();
        Some(Value::Array(vec![x.into(), y.into(), z.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.plot_trisurf(data[0], data[1], data[2]{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for Trisurf {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Set the view on a set of 3D axes.
///
/// Angles are in degrees.
///
/// ```python
/// ax.view_init(azim={azim}, elev={elev}, **{opts})
/// ```
///
/// Prelude: **No**
///
/// JSON data: **None**
#[derive(Clone, Debug, PartialEq)]
pub struct ViewInit {
    /// Azimuthal angle.
    pub azim: f64,
    /// Elevational angle.
    pub elev: f64,
    /// Optional keyword arguments.
    pub opts: Vec<Opt>,
}

impl ViewInit {
    /// Create a new `ViewInit` with no options.
    pub fn new(azim: f64, elev: f64) -> Self {
        Self { azim, elev, opts: Vec::new() }
    }
}

/// Create a new [`ViewInit`] with no options.
pub fn view_init(azim: f64, elev: f64) -> ViewInit { ViewInit::new(azim, elev) }

impl Matplotlib for ViewInit {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<Value> { None }

    fn py_cmd(&self) -> String {
        format!("ax.view_init(azim={}, elev={}{}{})",
            self.azim,
            self.elev,
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

impl MatplotlibOpts for ViewInit {
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self {
        self.opts.push((key, val).into());
        self
    }
}

/// Rearrange the grouping of tuples.
///
/// Although this trait can in principle describe any effective isomorphism
/// between two types, the implementations in this crate focus on those
/// describing how tuples can be rearranged trivially. That is, this crate
/// implements `Associator` to perform "flattening" (or "unflattening")
/// operations on tuples of few to several elements.
///
/// This is helpful in interfacing chains of calls to [`Iterator::zip`] with
/// several constructors in this module that require iterators over "flat"
/// tuples.
/// ```
/// use matplotlib::commands::assoc;
///
/// let x = vec![1,    2,     3_usize];
/// let y = vec!['a',  'b',   'c'    ];
/// let z = vec![true, false, true   ];
///
/// let flat: Vec<(usize, char, bool)>
///     = x.iter().copied()
///     .zip(y.iter().copied())
///     .zip(z.iter().copied()) // element type is ((usize, char), bool)
///     .map(assoc) // ((A, B), C) -> (A, B, C)
///     .collect();
///
/// assert_eq!(flat, vec![(1, 'a', true), (2, 'b', false), (3, 'c', true)]);
///
/// // can also be used for unzipping
/// let ((x2, y2), z2): ((Vec<usize>, Vec<char>), Vec<bool>)
///     = flat.into_iter().map(assoc).unzip();
///
/// assert_eq!(x2, x);
/// assert_eq!(y2, y);
/// assert_eq!(z2, z);
/// ```
pub trait Associator<P> {
    /// Rearrange the elements of `self`.
    fn assoc(self) -> P;
}

// there may be a way to do all these with recursive macros, but I'm too dumb
// for it; instead, we'll bootstrap with four base impls:

impl<A, B, C> Associator<((A, B), C)> for (A, B, C) {
    fn assoc(self) -> ((A, B), C) { ((self.0, self.1), self.2) }
}

impl<A, B, C> Associator<(A, B, C)> for ((A, B), C) {
    fn assoc(self) -> (A, B, C) { (self.0.0, self.0.1, self.1) }
}

impl<A, B, C> Associator<(A, (B, C))> for (A, B, C) {
    fn assoc(self) -> (A, (B, C)) { (self.0, (self.1, self.2)) }
}

impl<A, B, C> Associator<(A, B, C)> for (A, (B, C)) {
    fn assoc(self) -> (A, B, C) { (self.0, self.1.0, self.1.1) }
}

// now use the base impls to cover cases with more elements

macro_rules! impl_biassoc {
    (
        <$( $gen:ident ),+>,
        $pair:ty,
        ($( $l:ident ),+),
        $r:ident $(,)?
    ) => {
        impl<$( $gen ),+> Associator<$pair> for ($( $gen ),+) {
            fn assoc(self) -> $pair {
                let ($( $l ),+, $r) = self;
                (($( $l ),+).assoc(), $r)
            }
        }

        impl<$( $gen ),+> Associator<($( $gen ),+)> for $pair {
            fn assoc(self) -> ($( $gen ),+) {
                let ($( $l ),+) = self.0.assoc();
                ($( $l ),+, self.1)
            }
        }
    };
    (
        <$( $gen:ident ),+>,
        $pair:ty,
        $l:ident,
        ($( $r:ident ),+) $(,)?
    ) => {
        impl<$( $gen ),+> Associator<$pair> for ($( $gen ),+) {
            fn assoc(self) -> $pair {
                let ($l, $( $r ),+) = self;
                ($l, ($( $r ),+).assoc())
            }
        }

        impl<$( $gen ),+> Associator<($( $gen ),+)> for $pair {
            fn assoc(self) -> ($( $gen ),+) {
                let ($( $r ),+) = self.1.assoc();
                (self.0, $( $r ),+)
            }
        }
    };
}

impl_biassoc!(<A, B, C, D>, (((A, B), C), D), (a, b, c), d);
impl_biassoc!(<A, B, C, D>, ((A, (B, C)), D), (a, b, c), d);
impl_biassoc!(<A, B, C, D>, (A, ((B, C), D)), a, (b, c, d));
impl_biassoc!(<A, B, C, D>, (A, (B, (C, D))), a, (b, c, d));
impl_biassoc!(<A, B, C, D, E>, ((((A, B), C), D), E), (a, b, c, d), e);
impl_biassoc!(<A, B, C, D, E>, (((A, (B, C)), D), E), (a, b, c, d), e);
impl_biassoc!(<A, B, C, D, E>, ((A, ((B, C), D)), E), (a, b, c, d), e);
impl_biassoc!(<A, B, C, D, E>, ((A, (B, (C, D))), E), (a, b, c, d), e);
impl_biassoc!(<A, B, C, D, E>, (A, (((B, C), D), E)), a, (b, c, d, e));
impl_biassoc!(<A, B, C, D, E>, (A, ((B, (C, D)), E)), a, (b, c, d, e));
impl_biassoc!(<A, B, C, D, E>, (A, (B, ((C, D), E))), a, (b, c, d, e));
impl_biassoc!(<A, B, C, D, E>, (A, (B, (C, (D, E)))), a, (b, c, d, e));
impl_biassoc!(<A, B, C, D, E, F>, (((((A, B), C), D), E), F), (a, b, c, d, e), f);
impl_biassoc!(<A, B, C, D, E, F>, ((((A, (B, C)), D), E), F), (a, b, c, d, e), f);
impl_biassoc!(<A, B, C, D, E, F>, (((A, ((B, C), D)), E), F), (a, b, c, d, e), f);
impl_biassoc!(<A, B, C, D, E, F>, (((A, (B, (C, D))), E), F), (a, b, c, d, e), f);
impl_biassoc!(<A, B, C, D, E, F>, ((A, (((B, C), D), E)), F), (a, b, c, d, e), f);
impl_biassoc!(<A, B, C, D, E, F>, ((A, ((B, (C, D)), E)), F), (a, b, c, d, e), f);
impl_biassoc!(<A, B, C, D, E, F>, ((A, (B, ((C, D), E))), F), (a, b, c, d, e), f);
impl_biassoc!(<A, B, C, D, E, F>, ((A, (B, (C, (D, E)))), F), (a, b, c, d, e), f);
impl_biassoc!(<A, B, C, D, E, F>, (A, ((((B, C), D), E), F)), a, (b, c, d, e, f));
impl_biassoc!(<A, B, C, D, E, F>, (A, (((B, (C, D)), E), F)), a, (b, c, d, e, f));
impl_biassoc!(<A, B, C, D, E, F>, (A, ((B, ((C, D), E)), F)), a, (b, c, d, e, f));
impl_biassoc!(<A, B, C, D, E, F>, (A, ((B, (C, (D, E))), F)), a, (b, c, d, e, f));
impl_biassoc!(<A, B, C, D, E, F>, (A, (B, (((C, D), E), F))), a, (b, c, d, e, f));
impl_biassoc!(<A, B, C, D, E, F>, (A, (B, ((C, (D, E)), F))), a, (b, c, d, e, f));
impl_biassoc!(<A, B, C, D, E, F>, (A, (B, (C, ((D, E), F)))), a, (b, c, d, e, f));
impl_biassoc!(<A, B, C, D, E, F>, (A, (B, (C, (D, (E, F))))), a, (b, c, d, e, f));

/// Quick shortcut to [`Associator::assoc`] that doesn't require importing the
/// trait.
pub fn assoc<A, B>(a: A) -> B
where A: Associator<B>
{
    a.assoc()
}

/// Quick shortcut to calling `.map` on an iterator with [`assoc`].
pub fn assoc_iter<I, J, A, B>(iter: I) -> std::iter::Map<J, fn(A) -> B>
where
    I: IntoIterator<IntoIter = J, Item = A>,
    J: Iterator<Item = A>,
    A: Associator<B>,
{
    iter.into_iter().map(assoc)
}

#[cfg(test)]
mod tests {
    use crate::{ Mpl, Run, MatplotlibOpts, opt, GSPos };
    use super::*;

    fn runner() -> Run { Run::Debug }

    #[test]
    fn test_prelude_init() {
        Mpl::default()
            & DefPrelude
            & DefInit
            | runner()
    }

    #[test]
    fn test_axhline() {
        Mpl::default()
            & axhline(10.0).o("linestyle", "-")
            | runner()
    }

    #[test]
    fn test_axline() {
        Mpl::default()
            & axline((0.0, 0.0), (10.0, 10.0)).o("linestyle", "-")
            | runner()
    }

    #[test]
    fn test_axlinem() {
        Mpl::default()
            & axlinem((0.0, 0.0), 1.0).o("linestyle", "-")
            | runner()
    }

    #[test]
    fn test_axtext() {
        Mpl::default()
            & axtext(0.5, 0.5, "hello world").o("ha", "left").o("va", "bottom")
            | runner()
    }

    #[test]
    fn test_axvline() {
        Mpl::default()
            & axvline(10.0).o("linestyle", "-")
            | runner()
    }

    #[test]
    fn test_bar() {
        Mpl::default()
            & bar([0.0, 1.0], [0.5, 0.5]).o("color", "C0")
            | runner()
    }

    #[test]
    fn test_bar_pairs() {
        Mpl::default()
            & bar_pairs([(0.0, 0.5), (1.0, 0.5)]).o("color", "C0")
            | runner()
    }

    #[test]
    fn test_bar_eq() {
        assert_eq!(
            bar([0.0, 1.0], [0.5, 0.5]).o("color", "C0"),
            bar_pairs([(0.0, 0.5), (1.0, 0.5)]).o("color", "C0"),
        )
    }

    #[test]
    fn test_barh() {
        Mpl::default()
            & barh([0.0, 1.0], [0.5, 0.5]).o("color", "C0")
            | runner()
    }

    #[test]
    fn test_barh_pairs() {
        Mpl::default()
            & barh_pairs([(0.0, 0.5), (1.0, 0.5)]).o("color", "C0")
            | runner()
    }

    #[test]
    fn test_barh_eq() {
        assert_eq!(
            barh([0.0, 1.0], [0.5, 0.5]).o("color", "C0"),
            barh_pairs([(0.0, 0.5), (1.0, 0.5)]).o("color", "C0"),
        )
    }

    #[test]
    fn test_boxplot() {
        Mpl::default()
            & boxplot([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]).o("notch", true)
            | runner()
    }

    #[test]
    fn test_boxplot_flat() {
        Mpl::default()
            & boxplot_flat([0.0, 1.0, 2.0, 2.0, 3.0, 4.0], 3).o("notch", true)
            | runner()
    }

    #[test]
    fn test_boxplot_eq() {
        assert_eq!(
            boxplot([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]).o("notch", true),
            boxplot_flat([0.0, 1.0, 2.0, 2.0, 3.0, 4.0], 3).o("notch", true),
        )
    }

    #[test]
    fn test_clabel() {
        Mpl::default()
            & imshow([[0.0, 1.0], [2.0, 3.0]])
            & colorbar()
            & clabel("hello world").o("fontsize", "medium")
            | runner()
    }

    #[test]
    fn test_clim() {
        Mpl::default()
            & imshow([[0.0, 1.0], [2.0, 3.0]])
            & colorbar()
            & clim(Some(0.0), Some(1.0))
            | runner()
    }

    #[test]
    fn test_colorbar() {
        Mpl::default()
            & imshow([[0.0, 1.0], [2.0, 3.0]])
            & colorbar().o("location", "top")
            | runner()
    }

    #[test]
    fn test_contour() {
        Mpl::default()
            & contour([0.0, 1.0], [0.0, 1.0], [[0.0, 1.0], [2.0, 3.0]])
                .o("cmap", "bone")
            & colorbar()
            | runner()
    }

    #[test]
    fn test_contour_flat() {
        Mpl::default()
            & contour_flat([0.0, 1.0], [0.0, 1.0], [0.0, 1.0, 2.0, 3.0])
                .o("cmap", "bone")
            & colorbar()
            | runner()
    }

    #[test]
    fn test_contour_eq() {
        assert_eq!(
            contour([0.0, 1.0], [0.0, 1.0], [[0.0, 1.0], [2.0, 3.0]])
                .o("cmap", "bone"),
            contour_flat([0.0, 1.0], [0.0, 1.0], [0.0, 1.0, 2.0, 3.0])
                .o("cmap", "bone"),
        )
    }

    #[test]
    fn test_contourf() {
        Mpl::default()
            & contour([0.0, 1.0], [0.0, 1.0], [[0.0, 1.0], [2.0, 3.0]])
                .o("cmap", "bone")
            & colorbar()
            | runner()
    }

    #[test]
    fn test_contourf_flat() {
        Mpl::default()
            & contour_flat([0.0, 1.0], [0.0, 1.0], [0.0, 1.0, 2.0, 3.0])
                .o("cmap", "bone")
            & colorbar()
            | runner()
    }

    #[test]
    fn test_contourf_eq() {
        assert_eq!(
            contourf([0.0, 1.0], [0.0, 1.0], [[0.0, 1.0], [2.0, 3.0]])
                .o("cmap", "bone"),
            contourf_flat([0.0, 1.0], [0.0, 1.0], [0.0, 1.0, 2.0, 3.0])
                .o("cmap", "bone"),
        )
    }

    #[test]
    fn test_cticklabels() {
        Mpl::default()
            & imshow([[0.0, 1.0], [2.0, 3.0]])
            & colorbar()
            & cticklabels([0.0, 1.0], ["zero", "one"]).o("minor", true)
            | runner()
    }

    #[test]
    fn test_cticklabels_data() {
        Mpl::default()
            & imshow([[0.0, 1.0], [2.0, 3.0]])
            & colorbar()
            & cticklabels_data([(0.0, "zero"), (1.0, "one")]).o("minor", true)
            | runner()
    }

    #[test]
    fn test_cticklabels_eq() {
        assert_eq!(
            cticklabels([0.0, 1.0], ["zero", "one"]).o("minor", true),
            cticklabels_data([(0.0, "zero"), (1.0, "one")]).o("minor", true),
        )
    }

    #[test]
    fn test_cticks() {
        Mpl::default()
            & imshow([[0.0, 1.0], [2.0, 3.0]])
            & colorbar()
            & cticks([0.0, 1.0])
                .o("labels", PyValue::list(["zero", "one"]))
            | runner()
    }

    #[test]
    fn test_errorbar() {
        Mpl::default()
            & errorbar([0.0, 1.0], [0.0, 1.0], [0.5, 1.0]).o("color", "C0")
            | runner()
    }

    #[test]
    fn test_errorbar_data() {
        Mpl::default()
            & errorbar_data([(0.0, 0.0, 0.5), (1.0, 1.0, 1.0)]).o("color", "C0")
            | runner()
    }

    #[test]
    fn test_errorbar_eq() {
        assert_eq!(
            errorbar([0.0, 1.0], [0.0, 1.0], [0.5, 1.0]).o("color", "C0"),
            errorbar_data([(0.0, 0.0, 0.5), (1.0, 1.0, 1.0)]).o("color", "C0"),
        )
    }

    #[test]
    fn test_errorbar2() {
        Mpl::default()
            & errorbar2([0.0, 1.0], [0.0, 1.0], [1.0, 0.5], [0.5, 1.0])
                .o("color", "C0")
            | runner()
    }

    #[test]
    fn test_errorbar2_data() {
        Mpl::default()
            & errorbar2_data([(0.0, 0.0, 1.0, 0.5), (1.0, 1.0, 0.5, 1.0)])
                .o("color", "C0")
            | runner()
    }

    #[test]
    fn test_errorbar2_eq() {
        assert_eq!(
            errorbar2([0.0, 1.0], [0.0, 1.0], [1.0, 0.5], [0.5, 1.0])
                .o("color", "C0"),
            errorbar2_data([(0.0, 0.0, 1.0, 0.5), (1.0, 1.0, 0.5, 1.0)])
                .o("color", "C0"),
        )
    }

    #[test]
    fn test_figtext() {
        Mpl::default()
            & figtext(0.5, 0.5, "hello world").o("ha", "left").o("va", "bottom")
            | runner()
    }

    #[test]
    fn test_fill_between() {
        Mpl::default()
            & fill_between([0.0, 1.0], [-0.5, 0.0], [0.5, 2.0]).o("color", "C0")
            | runner()
    }

    #[test]
    fn test_fill_between_data() {
        Mpl::default()
            & fill_between_data([(0.0, -0.5, 0.5), (1.0, 0.0, 2.0)])
                .o("color", "C0")
            | runner()
    }

    #[test]
    fn test_fill_between_eq() {
        assert_eq!(
            fill_between([0.0, 1.0], [-0.5, 0.0], [0.5, 2.0]).o("color", "C0"),
            fill_between_data([(0.0, -0.5, 0.5), (1.0, 0.0, 2.0)])
                .o("color", "C0"),
        )
    }

    #[test]
    fn test_fillbetween_from_errorbar() {
        let ebar =
            errorbar_data([(0.0, 0.0, 0.5), (1.0, 1.0, 1.0)]);
        let ebar2 =
            errorbar2_data([(0.0, 0.25, 0.75, 0.25), (1.0, 1.0, 1.0, 1.0)]);
        let fbetw =
            fill_between_data([(0.0, -0.5, 0.5), (1.0, 0.0, 2.0)]);
        assert_eq!(FillBetween::from(ebar),  fbetw);
        assert_eq!(FillBetween::from(ebar2), fbetw);
    }

    #[test]
    fn test_errorbar_from_fillbetween() {
        let fbetw = fill_between_data([(0.0, -0.5, 0.5), (1.0, 0.0, 2.0)]);
        let ebar = errorbar_data([(0.0, 0.0, 0.5), (1.0, 1.0, 1.0)]);
        assert_eq!(Errorbar::from(fbetw), ebar);
    }

    #[test]
    fn test_fill_betweenx() {
        Mpl::default()
            & fill_betweenx([0.0, 1.0], [-0.5, 0.0], [0.5, 2.0])
                .o("color", "C0")
            | runner()
    }

    #[test]
    fn test_fill_betweenx_data() {
        Mpl::default()
            & fill_betweenx_data([(0.0, -0.5, 0.5), (1.0, 0.0, 2.0)])
                .o("color", "C0")
            | runner()
    }

    #[test]
    fn test_fill_betweenx_eq() {
        assert_eq!(
            fill_betweenx([0.0, 1.0], [-0.5, 0.0], [0.5, 2.0]).o("color", "C0"),
            fill_betweenx_data([(0.0, -0.5, 0.5), (1.0, 0.0, 2.0)])
                .o("color", "C0"),
        )
    }

    #[test]
    fn test_grid() {
        Mpl::default()
            & grid(true).o("which", "both")
            | runner()
    }

    #[test]
    fn test_hist() {
        Mpl::default()
            & hist([0.0, 1.0, 2.0])
                .o("bins", PyValue::list([-0.5, 0.5, 1.5, 2.5]))
            | runner()
    }

    #[test]
    fn test_hist2d() {
        Mpl::default()
            & hist2d([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]).o("cmap", "bone")
            | runner()
    }

    #[test]
    fn test_hist2d_pairs() {
        Mpl::default()
            & hist2d_pairs([(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)])
                .o("cmap", "bone")
            | runner()
    }

    #[test]
    fn test_hist2d_eq() {
        assert_eq!(
            hist2d([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]).o("cmap", "bone"),
            hist2d_pairs([(0.0, 0.0), (1.0, 2.0), (2.0, 4.0)]).o("cmap", "bone"),
        )
    }

    #[test]
    fn test_violinplot() {
        Mpl::default()
            & violinplot([[0.0, 1.0], [2.0, 3.0]]).o("vert", false)
            | runner()
    }

    #[test]
    fn test_violinplot_flat() {
        Mpl::default()
            & violinplot_flat([0.0, 1.0, 2.0, 3.0], 2).o("vert", false)
            | runner()
    }

    #[test]
    fn test_violinplot_eq() {
        assert_eq!(
            violinplot([[0.0, 1.0], [2.0, 3.0]]).o("vert", false),
            violinplot_flat([0.0, 1.0, 2.0, 3.0], 2).o("vert", false),
        )
    }

    #[test]
    fn test_imshow() {
        Mpl::default()
            & imshow([[0.0, 1.0], [2.0, 3.0]]).o("cmap", "bone")
            | runner()
    }

    #[test]
    fn test_imshow_flat() {
        Mpl::default()
            & imshow_flat([0.0, 1.0, 2.0, 3.0], 2).o("cmap", "bone")
            | runner()
    }

    #[test]
    fn test_imshow_eq() {
        assert_eq!(
            imshow([[0.0, 1.0], [2.0, 3.0]]).o("cmap", "bone"),
            imshow_flat([0.0, 1.0, 2.0, 3.0], 2).o("cmap", "bone"),
        )
    }

    #[test]
    fn test_inset_axes() {
        Mpl::default()
            & inset_axes(0.5, 0.5, 0.25, 0.25).o("polar", true)
            | runner()
    }

    #[test]
    fn test_inset_axes_pairs() {
        Mpl::default()
            & inset_axes_pairs((0.5, 0.5), (0.25, 0.25)).o("polar", true)
            | runner()
    }

    #[test]
    fn test_label() {
        Mpl::default()
            & label(Axis::X, "xlabel").o("fontsize", "large")
            & label(Axis::Y, "ylabel").o("fontsize", "large")
            | runner()
    }

    #[test]
    fn test_xlabel() {
        Mpl::default()
            & xlabel("xlabel").o("fontsize", "large")
            | runner()
    }

    #[test]
    fn test_ylabel() {
        Mpl::default()
            & ylabel("ylabel").o("fontsize", "large")
            | runner()
    }

    #[test]
    fn test_label_eq() {
        assert_eq!(label(Axis::X, "xlabel"), xlabel("xlabel"));
        assert_eq!(label(Axis::Y, "ylabel"), ylabel("ylabel"));
    }

    #[test]
    fn test_legend() {
        Mpl::default()
            & plot([0.0], [0.0]).o("label", "hello world")
            & legend().o("loc", "lower left")
            | runner()
    }

    #[test]
    fn test_lim() {
        Mpl::default()
            & lim(Axis::X, Some(-10.0), Some(10.0))
            & lim(Axis::Y, Some(-10.0), Some(10.0))
            | runner()
    }

    #[test]
    fn test_xlim() {
        Mpl::default()
            & xlim(Some(-10.0), Some(10.0))
            | runner()
    }

    #[test]
    fn test_ylim() {
        Mpl::default()
            & ylim(Some(-10.0), Some(10.0))
            | runner()
    }

    #[test]
    fn test_lim_eq() {
        assert_eq!(
            lim(Axis::X, Some(-10.0), Some(15.0)),
            xlim(Some(-10.0), Some(15.0)),
        );
        assert_eq!(
            lim(Axis::Y, Some(-10.0), Some(15.0)),
            ylim(Some(-10.0), Some(15.0)),
        );
        assert_eq!(
            lim(Axis::Z, Some(-10.0), Some(15.0)),
            zlim(Some(-10.0), Some(15.0)),
        )
    }

    #[test]
    fn test_pie() {
        Mpl::default()
            & pie([1.0, 2.0]).o("radius", 2)
            | runner()
    }

    #[test]
    fn test_plot() {
        Mpl::default()
            & plot([0.0, 1.0], [0.0, 1.0]).o("color", "C0")
            | runner()
    }

    #[test]
    fn test_plot_pairs() {
        Mpl::default()
            & plot_pairs([(0.0, 0.0), (1.0, 1.0)]).o("color", "C0")
            | runner()
    }

    #[test]
    fn test_plot_eq() {
        assert_eq!(
            plot([0.0, 1.0], [0.0, 1.0]).o("color", "C0"),
            plot_pairs([(0.0, 0.0), (1.0, 1.0)]).o("color", "C0"),
        )
    }

    #[test]
    fn test_quiver() {
        Mpl::default()
            & quiver([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
                .o("pivot", "middle")
            | runner()
    }

    #[test]
    fn test_quiver_data() {
        Mpl::default()
            & quiver_data([(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)])
                .o("pivot", "middle")
            | runner()
    }

    #[test]
    fn test_quiver_pairs() {
        Mpl::default()
            & quiver_pairs([(0.0, 0.0), (1.0, 1.0)], [(0.0, 0.0), (1.0, 1.0)])
                .o("pivot", "middle")
            | runner()
    }

    #[test]
    fn test_quiver_eq() {
        let norm =
            quiver([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
            .o("pivot", "middle");
        let data =
            quiver_data([(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)])
                .o("pivot", "middle");
        let pairs =
            quiver_pairs([(0.0, 0.0), (1.0, 1.0)], [(0.0, 0.0), (1.0, 1.0)])
                .o("pivot", "middle");
        assert_eq!(norm, data);
        assert_eq!(norm, pairs);
    }

    #[test]
    fn test_rcparam() {
        Mpl::default()
            & rcparam("figure.figsize", PyValue::list([2.5, 3.5]))
            | runner()
    }

    #[test]
    fn test_scale() {
        Mpl::default()
            & scale(Axis::X, AxisScale::Log)
            & scale(Axis::Y, AxisScale::Logit)
            | runner()
    }

    #[test]
    fn test_xscale() {
        Mpl::default()
            & xscale(AxisScale::Log)
            | runner()
    }

    #[test]
    fn test_yscale() {
        Mpl::default()
            & yscale(AxisScale::Logit)
            | runner()
    }

    #[test]
    fn test_scale_eq() {
        assert_eq!(scale(Axis::X, AxisScale::Log), xscale(AxisScale::Log));
        assert_eq!(scale(Axis::Y, AxisScale::Logit), yscale(AxisScale::Logit));
        assert_eq!(scale(Axis::Z, AxisScale::SymLog), zscale(AxisScale::SymLog));
    }

    #[test]
    fn test_scatter() {
        Mpl::default()
            & scatter([0.0, 1.0], [0.0, 1.0]).o("marker", "D")
            | runner()
    }

    #[test]
    fn test_scatter_pairs() {
        Mpl::default()
            & scatter_pairs([(0.0, 0.0), (1.0, 1.0)]).o("marker", "D")
            | runner()
    }

    #[test]
    fn test_scatter_eq() {
        assert_eq!(
            scatter([0.0, 1.0], [0.0, 1.0]).o("marker", "D"),
            scatter_pairs([(0.0, 0.0), (1.0, 1.0)]).o("marker", "D"),
        )
    }

    #[test]
    fn test_suptitle() {
        Mpl::default()
            & suptitle("hello world").o("fontsize", "xx-small")
            | runner()
    }

    #[test]
    fn test_supxlabel() {
        Mpl::default()
            & supxlabel("hello world").o("fontsize", "xx-small")
            | runner()
    }

    #[test]
    fn test_supylabel() {
        Mpl::default()
            & supylabel("hello world").o("fontsize", "xx-small")
            | runner()
    }

    #[test]
    fn test_make_grid() {
        Mpl::new_grid(3, 3, [opt("sharex", true), opt("sharey", true)])
            & focus_ax("AX[1, 1]")
            & plot([0.0, 1.0], [0.0, 1.0]).o("color", "C1")
            | runner()
    }

    #[test]
    fn test_make_gridspec() {
        //        0   1   2
        //       |--||--||----|
        //
        //   -   +------++----+
        // 0 |   | 0    || 2  |
        //   -   |      ||    |
        //   -   |      ||    |
        // 1 |   |      ||    |
        //   -   +------+|    |
        //   -   +------+|    |
        // 2 |   | 1    ||    |
        //   -   +------++----+
        //       <sharex>
        Mpl::new_gridspec(
                [
                    opt("nrows", 3),
                    opt("ncols", 3),
                    opt("width_ratios", PyValue::list([1, 1, 2])),
                ],
                [
                    GSPos::new(0..2, 0..2),
                    GSPos::new(2..3, 0..2).sharex(Some(0)),
                    GSPos::new(0..3, 2..3),
                ],
            )
            & focus_ax("AX[1]")
            & plot([0.0, 1.0], [0.0, 1.0]).o("color", "C1")
            | runner()
    }

    #[test]
    fn test_tex_off() {
        Mpl::default()
            & DefPrelude
            & tex_off()
            | runner()
    }

    #[test]
    fn test_tex_on() {
        Mpl::default()
            & DefPrelude
            & tex_on()
            | runner()
    }

    #[test]
    fn test_text() {
        Mpl::default()
            & text(0.5, 0.5, "hello world").o("fontsize", "large")
            | runner()
    }

    #[test]
    fn test_tick_params() {
        Mpl::default()
            & tick_params(Axis2::Both).o("color", "r")
            | runner()
    }

    #[test]
    fn test_xtick_params() {
        Mpl::default()
            & xtick_params().o("color", "r")
            | runner()
    }

    #[test]
    fn test_ytick_params() {
        Mpl::default()
            & xtick_params().o("color", "r")
            | runner()
    }

    #[test]
    fn test_tick_params_eq() {
        assert_eq!(
            tick_params(Axis2::X).o("color", "r"),
            xtick_params().o("color", "r"),
        );
        assert_eq!(
            tick_params(Axis2::Y).o("color", "r"),
            ytick_params().o("color", "r"),
        );
    }

    #[test]
    fn test_ticklabels() {
        Mpl::default()
            & ticklabels(Axis::X, [0.0, 1.0], ["x:zero", "x:one"])
                .o("fontsize", "small")
            & ticklabels(Axis::Y, [0.0, 1.0], ["y:zero", "y:one"])
                .o("fontsize", "small")
            | runner()
    }

    #[test]
    fn test_ticklabels_data() {
        Mpl::default()
            & ticklabels_data(Axis::X, [(0.0, "x:zero"), (1.0, "x:one")])
                .o("fontsize", "small")
            & ticklabels_data(Axis::Y, [(0.0, "y:zero"), (1.0, "y:one")])
                .o("fontsize", "small")
            | runner()
    }

    #[test]
    fn test_xticklabels() {
        Mpl::default()
            & xticklabels([0.0, 1.0], ["x:zero", "x:one"])
                .o("fontsize", "small")
            | runner()
    }

    #[test]
    fn test_xticklabels_data() {
        Mpl::default()
            & xticklabels_data([(0.0, "x:zero"), (1.0, "x:one")])
                .o("fontsize", "small")
            | runner()
    }

    #[test]
    fn test_yticklabels() {
        Mpl::default()
            & yticklabels([0.0, 1.0], ["y:zero", "y:one"])
                .o("fontsize", "small")
            | runner()
    }

    #[test]
    fn test_yticklabels_data() {
        Mpl::default()
            & yticklabels_data([(0.0, "y:zero"), (1.0, "y:one")])
                .o("fontsize", "small")
            | runner()
    }

    #[test]
    fn test_ticklabels_eq() {
        let normx =
            ticklabels(Axis::X, [0.0, 1.0], ["x:zero", "x:one"]);
        let normx_data =
            ticklabels_data(Axis::X, [(0.0, "x:zero"), (1.0, "x:one")]);
        let aliasx =
            xticklabels([0.0, 1.0], ["x:zero", "x:one"]);
        let aliasx_data =
            xticklabels_data([(0.0, "x:zero"), (1.0, "x:one")]);
        let normy =
            ticklabels(Axis::Y, [0.0, 1.0], ["y:zero", "y:one"]);
        let normy_data =
            ticklabels_data(Axis::Y, [(0.0, "y:zero"), (1.0, "y:one")]);
        let aliasy =
            yticklabels([0.0, 1.0], ["y:zero", "y:one"]);
        let aliasy_data =
            yticklabels_data([(0.0, "y:zero"), (1.0, "y:one")]);
        assert_eq!(normx, normx_data);
        assert_eq!(aliasx, aliasx_data);
        assert_eq!(normx, aliasx);
        assert_eq!(normy, normy_data);
        assert_eq!(aliasy, aliasy_data);
        assert_eq!(normy, aliasy);
    }

    #[test]
    fn test_ticks() {
        Mpl::default()
            & ticks(Axis::X, [0.0, 1.0]).o("minor", true)
            & ticks(Axis::Y, [0.0, 2.0]).o("minor", true)
            | runner()
    }

    #[test]
    fn test_xticks() {
        Mpl::default()
            & xticks([0.0, 1.0]).o("minor", true)
            | runner()
    }

    #[test]
    fn test_yticks() {
        Mpl::default()
            & yticks([0.0, 2.0]).o("minor", true)
            | runner()
    }

    #[test]
    fn test_ticks_eq() {
        let normx = ticks(Axis::X, [0.0, 1.0]);
        let aliasx = xticks([0.0, 1.0]);
        let normy = ticks(Axis::Y, [0.0, 2.0]);
        let aliasy = yticks([0.0, 2.0]);
        assert_eq!(normx, aliasx);
        assert_eq!(normy, aliasy);
    }

    #[test]
    fn test_title() {
        Mpl::default()
            & title("hello world").o("fontsize", "large")
            | runner()
    }

    #[test]
    fn test_tight_layout() {
        Mpl::new_grid(3, 3, [])
            & tight_layout().o("h_pad", 1.0).o("w_pad", 0.5)
            | runner()
    }

    #[test]
    fn test_make_3d() {
        Mpl::new_3d([opt("elev", 50.0)])
            | runner()
    }

    #[test]
    fn test_plot3() {
        Mpl::new_3d([])
            & plot3([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]).o("marker", "D")
            | runner()
    }

    #[test]
    fn test_plot3_data() {
        Mpl::new_3d([])
            & plot3_data([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]).o("marker", "D")
            | runner()
    }

    #[test]
    fn test_plot3_eq() {
        assert_eq!(
            plot3([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]).o("marker", "D"),
            plot3_data([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]).o("marker", "D"),
        )
    }

    #[test]
    fn test_scatter3() {
        Mpl::new_3d([])
            & scatter3([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]).o("marker", "D")
            | runner()
    }

    #[test]
    fn test_scatter3_data() {
        Mpl::new_3d([])
            & scatter3_data([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]).o("marker", "D")
            | runner()
    }

    #[test]
    fn test_scatter3_eq() {
        assert_eq!(
            scatter3([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]).o("marker", "D"),
            scatter3_data([(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]).o("marker", "D"),
        )
    }

    #[test]
    fn test_quiver3() {
        Mpl::new_3d([])
            & quiver3(
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 2.0],
                [1.0, 2.0],
                [1.0, 2.0],
            ).o("pivot", "middle")
            | runner()
    }

    #[test]
    fn test_quiver3_data() {
        Mpl::new_3d([])
            & quiver3_data([
                (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                (1.0, 1.0, 1.0, 2.0, 2.0, 2.0),
            ]).o("pivot", "middle")
            | runner()
    }

    #[test]
    fn test_quiver3_triples() {
        Mpl::new_3d([])
            & quiver3_triples(
                [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
                [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
            ).o("pivot", "middle")
            | runner()
    }

    #[test]
    fn test_quiver3_eq() {
        let norm = quiver3(
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ).o("pivot", "middle");
        let data = quiver3_data([
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 2.0, 2.0, 2.0),
        ]).o("pivot", "middle");
        let triples = quiver3_triples(
            [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
            [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
        ).o("pivot", "middle");
        assert_eq!(norm, data);
        assert_eq!(norm, triples);
    }

    #[test]
    fn test_surface() {
        Mpl::new_3d([])
            & surface(
                [[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 1.0], [2.0, 3.0]],
            ).o("cmap", "rainbow")
            | runner()
    }

    #[test]
    fn test_surface_data() {
        Mpl::new_3d([])
            & surface_data(
                [
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 1.0),
                    (0.0, 1.0, 2.0),
                    (1.0, 1.0, 3.0),
                ],
                2,
            ).o("cmap", "rainbow")
            | runner()
    }

    #[test]
    fn test_surface_flat() {
        Mpl::new_3d([])
            & surface_flat(
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 2.0, 3.0],
                2,
            ).o("cmap", "rainbow")
            | runner()
    }

    #[test]
    fn test_surface_eq() {
        let norm = surface(
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.0, 1.0], [2.0, 3.0]],
        ).o("cmap", "rainbow");
        let data = surface_data(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 1.0),
                (0.0, 1.0, 2.0),
                (1.0, 1.0, 3.0),
            ],
            2,
        ).o("cmap", "rainbow");
        let flat = surface_flat(
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 2.0, 3.0],
            2,
        ).o("cmap", "rainbow");
        assert_eq!(norm, data);
        assert_eq!(norm, flat);
    }

    #[test]
    fn test_trisurf() {
        Mpl::new_3d([])
            & trisurf(
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 2.0, 3.0],
            ).o("cmap", "rainbow")
            | runner()
    }

    #[test]
    fn test_trisurf_data() {
        Mpl::new_3d([])
            & trisurf_data([
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 1.0),
                (0.0, 1.0, 2.0),
                (1.0, 1.0, 3.0),
            ]).o("cmap", "rainbow")
            | runner()
    }

    #[test]
    fn test_trisurf_eq() {
        let norm = trisurf(
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 2.0, 3.0],
        ).o("cmap", "rainbow");
        let data = trisurf_data([
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 2.0),
            (1.0, 1.0, 3.0),
        ]).o("cmap", "rainbow");
        assert_eq!(norm, data);
    }

    #[test]
    fn test_view_init() {
        Mpl::new_3d([])
            & view_init(90.0, 0.0).o("roll", 45.0)
            | runner()
    }

    #[test]
    fn test_zlabel() {
        Mpl::new_3d([])
            & zlabel("zlabel")
            | runner()
    }

    #[test]
    fn test_zlim() {
        Mpl::new_3d([])
            & zlim(Some(-10.0), Some(15.0))
            | runner()
    }

    #[test]
    fn test_zscale() {
        Mpl::new_3d([])
            & zscale(AxisScale::Log)
            | runner()
    }

    #[test]
    fn test_zticklabels() {
        Mpl::new_3d([])
            & zticklabels([0.0, 1.0], ["zero", "one"]).o("minor", true)
            | runner()
    }

    #[test]
    fn test_zticklabels_data() {
        Mpl::new_3d([])
            & zticklabels_data([(0.0, "zero"), (1.0, "one")]).o("minor", true)
            | runner()
    }

    #[test]
    fn test_zticklabels_eq() {
        assert_eq!(
            zticklabels([0.0, 1.0], ["zero", "one"]).o("minor", true),
            zticklabels_data([(0.0, "zero"), (1.0, "one")]).o("minor", true),
        )
    }

    #[test]
    fn test_zticks() {
        Mpl::new_3d([])
            & zticks([0.0, 1.0]).o("minor", true)
            | runner()
    }
}

