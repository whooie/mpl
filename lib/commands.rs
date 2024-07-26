//! Commonly used plotting commands.
//!
//! This module contains types representing many common plotting commands,
//! implementing [`Matplotlib`] and sometimes [`MatplotlibOpts`]. Each can be
//! instantiated using their constructor methods or using a corresponding
//! function from this module for convenience, e.g.
//!
//! ```
//! # use mpl::commands::*;
//! let p1 = Plot::new([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]);
//! let p2 =      plot([0.0, 1.0, 2.0], [0.0, 2.0, 4.0]);
//!
//! assert_eq!(p1, p2);
//! ```

use serde_json as json;
use crate::core::{
    Matplotlib,
    MatplotlibOpts,
    Opt,
    PyValue,
    AsPy,
    PRELUDE,
    INIT,
};

/// Direct injection of arbitrary Python.
///
/// See [`Prelude`] for prelude code.
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

    fn data(&self) -> Option<json::Value> { None }

    fn py_cmd(&self) -> String { self.0.clone() }
}

/// Direct injection of arbitrary Python into the prelude.
///
/// See [`Raw`] for main body code.
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

    fn data(&self) -> Option<json::Value> { None }

    fn py_cmd(&self) -> String { self.0.clone() }
}

/// Default list of imports and library setup, not including `rcParams`.
///
/// This is automatically added to a [`Mpl`][crate::core::Mpl] when it's run if
/// no other commands are present for which [`Matplotlib::is_prelude`] equals
/// `true`.
///
/// See [`PRELUDE`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DefPrelude;

impl Matplotlib for DefPrelude {
    fn is_prelude(&self) -> bool { true }

    fn data(&self) -> Option<json::Value> { None }

    fn py_cmd(&self) -> String { PRELUDE.into() }
}

/// Default initialization of `fig` and `ax` plotting objects.
///
/// See [`INIT`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DefInit;

impl Matplotlib for DefInit {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<json::Value> { None }

    fn py_cmd(&self) -> String { INIT.into() }
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

    fn data(&self) -> Option<json::Value> { None }

    fn py_cmd(&self) -> String {
        format!("plt.rcParams[\"{}\"] = {}", self.key, self.val.as_py())
    }
}

/// Activate or deactivate TeX text.
///
/// ```python
/// plt.rcParams["text.usetex"] = {0}
/// ```
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
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<json::Value> { None }

    fn py_cmd(&self) -> String {
        format!("plt.rcParams[\"text.usetex\"] = {}", self.0.as_py())
    }
}

/// Set the local variable `ax` to a different set of axes.
///
/// ```python
/// ax = {0}
/// ```
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

    fn data(&self) -> Option<json::Value> { None }

    fn py_cmd(&self) -> String { format!("ax = {}", self.0) }
}

/// Set the local variable `fig` to a different figure.
///
/// ```python
/// fig = {0}
/// ```
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

    fn data(&self) -> Option<json::Value> { None }

    fn py_cmd(&self) -> String { format!("fig = {}", self.0) }
}

/// Set the local variable `cbar` to a different colorbar.
///
/// ```python
/// cbar = {0}
/// ```
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

    fn data(&self) -> Option<json::Value> { None }

    fn py_cmd(&self) -> String { format!("cbar = {}", self.0) }
}

/// Set the local variable `im` to a different colorbar.
///
/// ```python
/// im = {0}
/// ```
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

    fn data(&self) -> Option<json::Value> { None }

    fn py_cmd(&self) -> String { format!("im = {}", self.0) }
}

/// A (*x*, *y*) plot.
///
/// ```python
/// ax.plot({x}, {y}, **{opts})
/// ```
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

    fn data(&self) -> Option<json::Value> {
        let x: Vec<json::Value>
            = self.x.iter().copied().map(json::Value::from).collect();
        let y: Vec<json::Value>
            = self.y.iter().copied().map(json::Value::from).collect();
        Some(json::Value::Array(vec![x.into(), y.into()]))
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

    fn data(&self) -> Option<json::Value> {
        let data: Vec<json::Value>
            = self.data.iter().copied().map(json::Value::from).collect();
        Some(json::Value::Array(data))
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

    fn data(&self) -> Option<json::Value> {
        let x: Vec<json::Value>
            = self.x.iter().copied().map(json::Value::from).collect();
        let y: Vec<json::Value>
            = self.y.iter().copied().map(json::Value::from).collect();
        Some(json::Value::Array(vec![x.into(), y.into()]))
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

    fn data(&self) -> Option<json::Value> {
        let x: Vec<json::Value>
            = self.x.iter().copied().map(json::Value::from).collect();
        let y: Vec<json::Value>
            = self.y.iter().copied().map(json::Value::from).collect();
        Some(json::Value::Array(vec![x.into(), y.into()]))
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

/// A bar plot.
///
/// ```python
/// ax.bar({x}, {y}, **{opts})
/// ```
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

    fn data(&self) -> Option<json::Value> {
        let x: Vec<json::Value>
            = self.x.iter().copied().map(json::Value::from).collect();
        let y: Vec<json::Value>
            = self.y.iter().copied().map(json::Value::from).collect();
        Some(json::Value::Array(vec![x.into(), y.into()]))
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

    fn data(&self) -> Option<json::Value> {
        let y: Vec<json::Value>
            = self.y.iter().copied().map(json::Value::from).collect();
        let w: Vec<json::Value>
            = self.w.iter().copied().map(json::Value::from).collect();
        Some(json::Value::Array(vec![y.into(), w.into()]))
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
        let mut x: Vec<f64> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        let mut e: Vec<f64> = Vec::new();
        data.into_iter()
            .for_each(|(xk, yk, ek)| { x.push(xk); y.push(yk); e.push(ek); });
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

    fn data(&self) -> Option<json::Value> {
        let x: Vec<json::Value>
            = self.x.iter().copied().map(json::Value::from).collect();
        let y: Vec<json::Value>
            = self.y.iter().copied().map(json::Value::from).collect();
        let e: Vec<json::Value>
            = self.e.iter().copied().map(json::Value::from).collect();
        Some(json::Value::Array(vec![x.into(), y.into(), e.into()]))
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
                *y2k = (y1 - y2).abs();
            });
        Self { x, y: y1, e: y2, opts }
    }
}

/// Plot with asymmetric error bars.
///
/// ```python
/// ax.errorbar({x}, {y}, [{e_neg}, {e_pos}], **{opts})
/// ```
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
        let mut x: Vec<f64> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        let mut e_neg: Vec<f64> = Vec::new();
        let mut e_pos: Vec<f64> = Vec::new();
        data.into_iter()
            .for_each(|(xk, yk, emk, epk)| {
                x.push(xk); y.push(yk); e_neg.push(emk); e_pos.push(epk);
            });
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

    fn data(&self) -> Option<json::Value> {
        let x: Vec<json::Value>
            = self.x.iter().copied().map(json::Value::from).collect();
        let y: Vec<json::Value>
            = self.y.iter().copied().map(json::Value::from).collect();
        let e_neg: Vec<json::Value>
            = self.e_neg.iter().copied().map(json::Value::from).collect();
        let e_pos: Vec<json::Value>
            = self.e_pos.iter().copied().map(json::Value::from).collect();
        Some(json::Value::Array(
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
        let data: Vec<Vec<f64>>
            = data.into_iter()
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
        let data: Vec<Vec<f64>>
            = Chunks::new(data.into_iter(), size)
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

    fn data(&self) -> Option<json::Value> {
        let data: Vec<json::Value>
            = self.data.iter()
            .map(|row| {
                let row: Vec<json::Value>
                    = row.iter().copied().map(json::Value::from).collect();
                json::Value::Array(row)
            })
            .collect();
        Some(json::Value::Array(data))
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
        let data: Vec<Vec<f64>>
            = data.into_iter()
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
        let data: Vec<Vec<f64>>
            = Chunks::new(data.into_iter(), size)
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

    fn data(&self) -> Option<json::Value> {
        let data: Vec<json::Value>
            = self.data.iter()
            .map(|row| {
                let row: Vec<json::Value>
                    = row.iter().copied().map(json::Value::from).collect();
                json::Value::Array(row)
            })
            .collect();
        Some(json::Value::Array(data))
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
        let z: Vec<Vec<f64>>
            = z.into_iter()
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
        let z: Vec<Vec<f64>>
            = Chunks::new(z.into_iter(), x.len())
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

    fn data(&self) -> Option<json::Value> {
        let x: Vec<json::Value>
            = self.x.iter().copied().map(json::Value::from).collect();
        let y: Vec<json::Value>
            = self.y.iter().copied().map(json::Value::from).collect();
        let z: Vec<json::Value>
            = self.z.iter()
            .map(|row| {
                let row: Vec<json::Value>
                    = row.iter().copied().map(json::Value::from).collect();
                json::Value::Array(row)
            })
            .collect();
        Some(json::Value::Array(vec![x.into(), y.into(), z.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.contour(data[0], data[1], data[2]{}{})",
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
        let z: Vec<Vec<f64>>
            = z.into_iter()
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
        let z: Vec<Vec<f64>>
            = Chunks::new(z.into_iter(), x.len())
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
pub fn new_flat<X, Y, Z>(x: X, y: Y, z: Z) -> Contourf
where
    X: IntoIterator<Item = f64>,
    Y: IntoIterator<Item = f64>,
    Z: IntoIterator<Item = f64>,
{
    Contourf::new_flat(x, y, z)
}

impl Matplotlib for Contourf {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<json::Value> {
        let x: Vec<json::Value>
            = self.x.iter().copied().map(json::Value::from).collect();
        let y: Vec<json::Value>
            = self.y.iter().copied().map(json::Value::from).collect();
        let z: Vec<json::Value>
            = self.z.iter()
            .map(|row| {
                let row: Vec<json::Value>
                    = row.iter().copied().map(json::Value::from).collect();
                json::Value::Array(row)
            })
            .collect();
        Some(json::Value::Array(vec![x.into(), y.into(), z.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("ax.contourf(data[0], data[1], data[2]{}{})",
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
        let data: Vec<Vec<f64>>
            = data.into_iter()
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
        let data: Vec<Vec<f64>>
            = Chunks::new(data.into_iter(), rowlen)
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

    fn data(&self) -> Option<json::Value> {
        let data: Vec<json::Value>
            = self.data.iter()
            .map(|row| {
                let row: Vec<json::Value>
                    = row.iter().copied().map(json::Value::from).collect();
                json::Value::Array(row)
            })
            .collect();
        Some(json::Value::Array(data))
    }

    fn py_cmd(&self) -> String {
        format!("ax.imshow(data{}{})",
            if self.opts.is_empty() { "" } else { ", " },
            self.opts.as_py(),
        )
    }
}

/// A filled area between two horizontal curves.
///
/// ```python
/// ax.fill_between({x}, {y1}, {y2}, **{opts})
/// ```
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
        let mut x: Vec<f64> = Vec::new();
        let mut y1: Vec<f64> = Vec::new();
        let mut y2: Vec<f64> = Vec::new();
        data.into_iter()
            .for_each(|(xk, y1k, y2k)| {
                x.push(xk); y1.push(y1k); y2.push(y2k);
            });
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

    fn data(&self) -> Option<json::Value> {
        let x: Vec<json::Value>
            = self.x.iter().copied().map(json::Value::from).collect();
        let y1: Vec<json::Value>
            = self.y1.iter().copied().map(json::Value::from).collect();
        let y2: Vec<json::Value>
            = self.y2.iter().copied().map(json::Value::from).collect();
        Some(json::Value::Array(vec![x.into(), y1.into(), y2.into()]))
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
        let mut y: Vec<f64> = Vec::new();
        let mut x1: Vec<f64> = Vec::new();
        let mut x2: Vec<f64> = Vec::new();
        data.into_iter()
            .for_each(|(yk, x1k, x2k)| {
                y.push(yk); x1.push(x1k); x2.push(x2k);
            });
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

    fn data(&self) -> Option<json::Value> {
        let y: Vec<json::Value>
            = self.y.iter().copied().map(json::Value::from).collect();
        let x1: Vec<json::Value>
            = self.x1.iter().copied().map(json::Value::from).collect();
        let x2: Vec<json::Value>
            = self.x2.iter().copied().map(json::Value::from).collect();
        Some(json::Value::Array(vec![y.into(), x1.into(), x2.into()]))
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

    fn data(&self) -> Option<json::Value> { None }

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

    fn data(&self) -> Option<json::Value> { None }

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

    fn data(&self) -> Option<json::Value> { None }

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
/// ax.axline({xy}, xy2=None, m={m}, **{opts})
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

    fn data(&self) -> Option<json::Value> { None }

    fn py_cmd(&self) -> String {
        format!("ax.axline({:?}, xy2=None, m={}{}{})",
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

    fn data(&self) -> Option<json::Value> {
        Some(json::Value::Array(
            self.data.iter().copied().map(json::Value::from).collect()))
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

    fn data(&self) -> Option<json::Value> {
        Some(json::Value::Array(
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

    fn data(&self) -> Option<json::Value> {
        Some(json::Value::Array(
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

    fn data(&self) -> Option<json::Value> {
        Some(json::Value::Array(
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

    fn data(&self) -> Option<json::Value> { None }

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

impl Matplotlib for Scale {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<json::Value> { None }

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

impl Matplotlib for Lim {
    fn is_prelude(&self) -> bool { false }

    fn data(&self) -> Option<json::Value> { None }

    fn py_cmd(&self) -> String {
        let ax = format!("{:?}", self.axis).to_lowercase();
        let min
            = self.min.as_ref()
            .map(|x| format!("{}", x))
            .unwrap_or("None".into());
        let max
            = self.max.as_ref()
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

    fn data(&self) -> Option<json::Value> { None }

    fn py_cmd(&self) -> String {
        let min
            = self.min.as_ref()
            .map(|x| format!("{}", x))
            .unwrap_or("None".into());
        let max
            = self.max.as_ref()
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

    fn data(&self) -> Option<json::Value> { None }

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

    fn data(&self) -> Option<json::Value> { None }

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

    fn data(&self) -> Option<json::Value> { None }

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

    fn data(&self) -> Option<json::Value> {
        let v: Vec<json::Value>
            = self.v.iter().copied().map(json::Value::from).collect();
        Some(json::Value::Array(v))
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

    fn data(&self) -> Option<json::Value> {
        let v: Vec<json::Value>
            = self.v.iter().copied().map(json::Value::from).collect();
        Some(json::Value::Array(v))
    }

    fn py_cmd(&self) -> String {
        format!("cbar.set_ticks(data{}{}",
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
        let (v, s): (Vec<f64>, Vec<String>)
            = ticklabels.into_iter()
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

    fn data(&self) -> Option<json::Value> {
        let v: Vec<json::Value>
            = self.v.iter().copied().map(json::Value::from).collect();
        let s: Vec<json::Value>
            = self.s.iter().cloned().map(json::Value::from).collect();
        Some(json::Value::Array(vec![v.into(), s.into()]))
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
        let (v, s): (Vec<f64>, Vec<String>)
            = ticklabels.into_iter()
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

    fn data(&self) -> Option<json::Value> {
        let v: Vec<json::Value>
            = self.v.iter().copied().map(json::Value::from).collect();
        let s: Vec<json::Value>
            = self.s.iter().cloned().map(json::Value::from).collect();
        Some(json::Value::Array(vec![v.into(), s.into()]))
    }

    fn py_cmd(&self) -> String {
        format!("cbar.set_ticks(data[0], label=data[1]{}{})",
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

    fn data(&self) -> Option<json::Value> { None }

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

    fn data(&self) -> Option<json::Value> { None }

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

    fn data(&self) -> Option<json::Value> { None }

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

    fn data(&self) -> Option<json::Value> { None }

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

    fn data(&self) -> Option<json::Value> { None }

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

    fn data(&self) -> Option<json::Value> { None }

    fn py_cmd(&self) -> String {
        format!("ax.grid({}, {})", self.onoff.as_py(), self.opts.as_py())
    }
}

/// Adjust the padding between and around subplots.
///
/// ```python
/// fig.tight_layout(**{opts})
/// ```
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

    fn data(&self) -> Option<json::Value> { None }

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

    fn data(&self) -> Option<json::Value> { None }

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

// streamplot
// surface
// trisurf
// quiver

/// Set the view on a set of 3D axes.
///
/// Angles are in degrees.
///
/// ```python
/// ax.view_init(azim={azim}, elev={elev}, **{opts})
/// ```
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

    fn data(&self) -> Option<json::Value> { None }

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

