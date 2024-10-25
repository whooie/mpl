use std::{
    collections::HashMap,
    fs,
    io::{ self, Write },
    ops::Range,
    path::{ Path, PathBuf },
    process,
    rc::Rc,
};
use rand::distributions::{ Alphanumeric, DistString };
use serde_json as json;
use thiserror::Error;
use crate::commands::Axis2;

#[derive(Debug, Error)]
pub enum MplError {
    #[error("IO error: {0}")]
    IOError(#[from] io::Error),

    #[error("serialization error: {0}")]
    JsonError(#[from] json::Error),

    #[error("script error:\nstdout:\n{0}\nstderr:\n{1}")]
    PyError(String, String),
}
pub type MplResult<T> = Result<T, MplError>;

/// Default prelude to a Matplotlib script.
///
/// ```python
/// import datetime
/// import io
/// import json
/// import os
/// import random
/// import sys
/// import matplotlib
/// matplotlib.use("QtAgg")
/// import matplotlib.path as mpath
/// import matplotlib.patches as mpatches
/// import matplotlib.pyplot as plt
/// import matplotlib.cm as mcm
/// import matplotlib.colors as mcolors
/// import matplotlib.collections as mcollections
/// import matplotlib.ticker as mticker
/// import matplotlib.image as mimage
/// from mpl_toolkits.mplot3d import axes3d
/// import numpy as np
/// ```
pub const PRELUDE: &str
= "\
import datetime
import io
import json
import os
import random
import sys
import matplotlib
matplotlib.use(\"QtAgg\")
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.ticker as mticker
import matplotlib.image as mimage
from mpl_toolkits.mplot3d import axes3d
import numpy as np
";

/// Default initializer for plotting objects, defaulting to a single figure and
/// axis frame.
///
/// ```python
/// fig, ax = plt.subplots()
/// ```
pub const INIT: &str
= "\
fig, ax = plt.subplots()
";

/// An executable element in a Matplotlib script.
pub trait Matplotlib: std::fmt::Debug {
    /// Return `true` if `self` should be considered as a prelude item, which
    /// are execute in the order seen but before any non-prelude items.
    fn is_prelude(&self) -> bool;

    /// Optionally encode some data as JSON, to be made available at `self`'s
    /// call site in the matplotlib script.
    fn data(&self) -> Option<json::Value>;

    /// Write `self` as Python. The (default) local environment will hold the
    /// following variables:
    ///
    /// - `data`: If [`self.data`][Matplotlib::data] returns `Some`, that data
    ///   will be available under this name.
    /// - `fig` and `ax`: The current figure of type `matplotlib.pyplot.Figure`
    ///   and the current set of axes, of type `matplotlib.axes.Axes`.
    fn py_cmd(&self) -> String;
}

/// Convert a Rust value to a Python source code string.
pub trait AsPy {
    fn as_py(&self) -> String;
}

impl AsPy for bool {
    fn as_py(&self) -> String { if *self { "True" } else { "False" }.into() }
}

impl AsPy for i32 {
    fn as_py(&self) -> String { self.to_string() }
}

impl AsPy for f64 {
    fn as_py(&self) -> String { self.to_string() }
}

impl AsPy for String {
    fn as_py(&self) -> String { format!("\"{self}\"") }
}

impl AsPy for &str {
    fn as_py(&self) -> String { format!("\"{self}\"") }
}

/// A primitive Python value or variable to be used in a keyword argument.
#[derive(Clone, Debug, PartialEq)]
pub enum PyValue {
    /// A `bool`.
    Bool(bool),
    /// An `int`.
    Int(i32),
    /// A `float`.
    Float(f64),
    /// A `str`.
    Str(String),
    /// A `list[...]`.
    List(Vec<PyValue>),
    /// A `dict[str, ...]`.
    Dict(HashMap<String, PyValue>),
    /// An arbitrary variable name.
    ///
    /// **Note**: This variant is *not* validated as a Python identifier.
    Var(String),
    /// Python's `None` value.
    None
}

impl From<bool> for PyValue {
    fn from(b: bool) -> Self { Self::Bool(b) }
}

impl From<i32> for PyValue {
    fn from(i: i32) -> Self { Self::Int(i) }
}

impl From<f64> for PyValue {
    fn from(f: f64) -> Self { Self::Float(f) }
}

impl From<String> for PyValue {
    fn from(s: String) -> Self { Self::Str(s) }
}

impl From<&str> for PyValue {
    fn from(s: &str) -> Self { Self::Str(s.into()) }
}

impl<T> From<&T> for PyValue
where T: Clone + Into<PyValue>
{
    fn from(x: &T) -> Self { x.clone().into() }
}

impl From<Vec<PyValue>> for PyValue {
    fn from(l: Vec<PyValue>) -> Self { Self::List(l) }
}

impl From<HashMap<String, PyValue>> for PyValue {
    fn from(d: HashMap<String, PyValue>) -> Self { Self::Dict(d) }
}

impl<T: Into<PyValue>> FromIterator<T> for PyValue {
    fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = T>
    {
        Self::list(iter)
    }
}

impl<S: Into<String>, T: Into<PyValue>> FromIterator<(S, T)> for PyValue {
    fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = (S, T)>
    {
        Self::dict(iter)
    }
}

impl PyValue {
    /// Create a `List` from an iterator.
    pub fn list<I, T>(items: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<PyValue>,
    {
        Self::List(items.into_iter().map(|item| item.into()).collect())
    }

    /// Create a `Dict` from an iterator.
    pub fn dict<I, S, T>(items: I) -> Self
    where
        I: IntoIterator<Item = (S, T)>,
        S: Into<String>,
        T: Into<PyValue>,
    {
        Self::Dict(
            items.into_iter().map(|(s, v)| (s.into(), v.into())).collect())
    }
}

impl AsPy for PyValue {
    fn as_py(&self) -> String {
        match self {
            Self::Bool(b) => if *b { "True".into() } else { "False".into() },
            Self::Int(i) => format!("{i}"),
            Self::Float(f) => format!("{f}"),
            Self::Str(s) => format!("\"{s}\""),
            Self::List(l) => {
                let n = l.len();
                let mut out = String::from("[");
                for (k, v) in l.iter().enumerate() {
                    out.push_str(&v.as_py());
                    if k < n - 1 { out.push_str(", "); }
                }
                out.push(']');
                out
            },
            Self::Dict(d) => {
                let n = d.len();
                let mut out = String::from("{");
                for (j, (k, v)) in d.iter().enumerate() {
                    out.push_str(&format!("\"{}\": {}", k, v.as_py()));
                    if j < n - 1 { out.push_str(", "); }
                }
                out.push('}');
                out
            },
            Self::Var(v) => v.clone(),
            Self::None => "None".into(),
        }
    }
}

/// An optional keyword argument.
#[derive(Clone, Debug, PartialEq)]
pub struct Opt(pub String, pub PyValue);

impl<T: Into<PyValue>> From<(&str, T)> for Opt {
    fn from(kv: (&str, T)) -> Self { Self(kv.0.into(), kv.1.into()) }
}

impl<T: Into<PyValue>> From<(String, T)> for Opt {
    fn from(kv: (String, T)) -> Self { Self(kv.0, kv.1.into()) }
}

impl Opt {
    /// Create a new `Opt`.
    pub fn new<T>(key: &str, val: T) -> Self
    where T: Into<PyValue>
    {
        Self(key.into(), val.into())
    }
}

/// Create a new [`Opt`].
pub fn opt<T>(key: &str, val: T) -> Opt
where T: Into<PyValue>
{
    Opt::new(key, val)
}

impl AsPy for Opt {
    fn as_py(&self) -> String { format!("{}={}", self.0, self.1.as_py()) }
}

impl AsPy for Vec<Opt> {
    fn as_py(&self) -> String {
        let n = self.len();
        let mut out = String::new();
        for (k, opt) in self.iter().enumerate() {
            out.push_str(&opt.as_py());
            if k < n - 1 { out.push_str(", "); }
        }
        out
    }
}

/// Extends [`Matplotlib`] to take optional keyword arguments.
pub trait MatplotlibOpts: Matplotlib {
    /// Apply a single keyword argument.
    fn kwarg<T: Into<PyValue>>(&mut self, key: &str, val: T) -> &mut Self;

    /// Apply a single keyword argument with full ownership of `self`.
    fn o<T: Into<PyValue>>(mut self, key: &str, val: T) -> Self
    where Self: Sized
    {
        self.kwarg(key, val);
        self
    }

    /// Apply a series of keyword arguments with full ownership of `self`.
    fn oo<I>(mut self, opts: I) -> Self
    where
        I: IntoIterator<Item = Opt>,
        Self: Sized,
    {
        opts.into_iter().for_each(|Opt(key, val)| { self.kwarg(&key, val); });
        self
    }
}

fn get_temp_fname() -> PathBuf {
    std::env::temp_dir()
        .join(Alphanumeric.sample_string(&mut rand::thread_rng(), 15))
}

#[derive(Debug)]
struct TempFile(PathBuf, Option<fs::File>);

impl TempFile {
    fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        fs::OpenOptions::new()
            .create(true)
            .append(false)
            .truncate(true)
            .write(true)
            .open(&path)
            .map(|file| Self(path, Some(file)))
    }
}

impl Write for TempFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Some(file) = self.1.as_mut() {
            file.write(buf)
        } else {
            Ok(0)
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(file) = self.1.as_mut() {
            file.flush()
        } else {
            Ok(())
        }
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        drop(self.1.take());
        fs::remove_file(&self.0).ok();
    }
}

/// Main script builder type.
///
/// Plotting scripts are built by combining base `Mpl`s with either individual
/// commands represented by types implementing [`Matplotlib`] (via
/// [`then`][Self::then]) or other `Mpl`s (via [`concat`][Self::concat]). Both
/// operations are overloaded onto the `&` operator.
///
/// When a script is ready to be run, [`Mpl::run`] appends the appropriate IO
/// commands to the end of the file before writing to the OS's default temp
/// directory (e.g. `/tmp` on Linux) and calling the system's default `python3`
/// executable. Files in the temp directory are cleaned up afterwards. `Mpl` can
/// also interact with the [`Run`] objects through both the `&` and `|`
/// operators, where instead of producing a `Result` like `run`, a `panic` is
/// raised on an error. `&` produces a final `Mpl` to be used in later
/// operations, while `|` returns `()`.
///
/// ```ignore
/// use std::f64::consts::TAU;
/// use mpl::{ Mpl, Run, MatplotlibOpts, commands as c };
///
/// let dx: f64 = TAU / 1000.0;
/// let x: Vec<f64> = (0..1000_u32).map(|k| f64::from(k) * dx).collect();
/// let y1: Vec<f64> = x.iter().copied().map(f64::sin).collect();
/// let y2: Vec<f64> = x.iter().copied().map(f64::cos).collect();
///
/// Mpl::default()
///     & c::plot(x.clone(), y1).o("marker", "o").o("color", "b")
///     & c::plot(x,         y2).o("marker", "D").o("color", "r")
///     | Run::Show
/// ```
#[derive(Clone, Debug, Default)]
pub struct Mpl {
    prelude: Vec<Rc<dyn Matplotlib + 'static>>,
    commands: Vec<Rc<dyn Matplotlib + 'static>>,
}
// pub struct Mpl(Vec<Rc<dyn Matplotlib + 'static>>);

impl Mpl {
    /// Create a new, empty plotting script.
    ///
    /// The resulting plot will implicitly pull in
    /// [`DefPrelude`][crate::commands::DefPrelude] and
    /// [`DefInit`][crate::commands::DefInit] when [`run`][Self::run] (or a
    /// synonym) is called if no other objects with [`Matplotlib::is_prelude`]`
    /// == true` are added.
    pub fn new() -> Self { Self::default() }

    /// Create a new plotting script, initializing to a figure with a single set
    /// of 3D axes (of type `mpl_toolkits.mplot3d.axes3d.Axes3D`).
    ///
    /// This pulls in [`DefPrelude`][crate::commands::DefPrelude], but not
    /// [`DefInit`][crate::commands::DefInit].
    ///
    /// Options are passed to the construction of the `Axes3D` object.
    pub fn new_3d<I>(opts: I) -> Self
    where I: IntoIterator<Item = Opt>
    {
        let opts: Vec<Opt> = opts.into_iter().collect();
        Self::default()
            & crate::commands::DefPrelude
            & crate::commands::Init3D { opts: opts.into_iter().collect() }
    }

    /// Like [`new_3d`][Self::new_3d], but call a closure on the new `Mpl`
    /// between prelude and figure/axes initialization.
    pub fn new_3d_with<I, F>(opts: I, f: F) -> Self
    where
        I: IntoIterator<Item = Opt>,
        F: FnOnce(Mpl) -> Mpl,
    {
        f(Self::default() & crate::commands::DefPrelude)
            & crate::commands::Init3D { opts: opts.into_iter().collect() }
    }

    /// Create a new plotting script, initializing to a figure with a regular
    /// grid of plots. All `Axes` objects will be stored in a 2D Numpy array
    /// under the local variable `AX`, and the script will be initially focused
    /// on the upper-left corner of the array, i.e. `ax = AX[0, 0]`.
    ///
    /// This pulls in [`DefPrelude`][crate::commands::DefPrelude], but not
    /// [`DefInit`][crate::commands::DefInit].
    ///
    /// Options are passed to the call to `pyplot.subplots`.
    pub fn new_grid<I>(nrows: usize, ncols: usize, opts: I) -> Self
    where I: IntoIterator<Item = Opt>
    {
        let opts: Vec<Opt> = opts.into_iter().collect();
        Self::default()
            & crate::commands::DefPrelude
            & crate::commands::InitGrid {
                nrows,
                ncols,
                opts: opts.into_iter().collect(),
            }
    }

    /// Like [`new_grid`][Self::new_grid], but call a closure on the new `Mpl`
    /// between prelude and figure/axes initialization.
    pub fn new_grid_with<I, F>(nrows: usize, ncols: usize, opts: I, f: F)
        -> Self
    where
        I: IntoIterator<Item = Opt>,
        F: FnOnce(Mpl) -> Mpl,
    {
        f(Self::default() & crate::commands::DefPrelude)
            & crate::commands::InitGrid {
                nrows,
                ncols,
                opts: opts.into_iter().collect(),
            }
    }

    /// Create a new plotting script, initializing a figure with Matplotlib's
    /// `gridspec`. Keyword arguments are passed to
    /// `pyplot.Figure.add_gridspec`, and each subplot's position in the
    /// gridspec is specified using a [`GSPos`]. All `Axes` objects will be
    /// stored in a 1D Numpy array under the local variable `AX`, and the script
    /// will be initially focused to the subplot corresponding to the first
    /// `GSPos` encountered, i.e. `ax = AX[0]`.
    ///
    /// This pulls in [`DefPrelude`][crate::commands::DefPrelude`], but not
    /// [`DefInit`][crate::commands::DefInit].
    pub fn new_gridspec<I, P>(gridspec_kw: I, positions: P) -> Self
    where
        I: IntoIterator<Item = Opt>,
        P: IntoIterator<Item = GSPos>,
    {
        Self::default()
            & crate::commands::DefPrelude
            & crate::commands::init_gridspec(gridspec_kw, positions)
    }

    /// Like [`new_gridspec`][Self::new_gridspec], but call a closure on the new
    /// `Mpl` between prelude and figure/axes initialization.
    pub fn new_gridspec_with<I, P, F>(gridspec_kw: I, positions: P, f: F)
        -> Self
    where
        I: IntoIterator<Item = Opt>,
        P: IntoIterator<Item = GSPos>,
        F: FnOnce(Mpl) -> Mpl,
    {
        f(Self::default() & crate::commands::DefPrelude)
            & crate::commands::init_gridspec(gridspec_kw, positions)
    }

    /// Add a new command to `self`.
    pub fn then<M: Matplotlib + 'static>(&mut self, item: M) -> &mut Self {
        if item.is_prelude() {
            self.prelude.push(Rc::new(item));
        } else {
            self.commands.push(Rc::new(item));
        }
        self
    }

    /// Combine `self` with `other`, moving all commands marked with
    /// [`is_prelude`][Matplotlib::is_prelude]` == true` to the top (with those
    /// from `self` before those from `other`) but maintaining command order
    /// otherwise.
    pub fn concat(&mut self, other: &Self) -> &mut Self {
        self.prelude.append(&mut other.prelude.clone());
        self.commands.append(&mut other.commands.clone());
        self
    }

    fn collect_data(&self) -> (json::Value, Vec<bool>) {
        let mut has_data =
            vec![false; self.prelude.len() + self.commands.len()];
        let data: Vec<json::Value> =
            self.prelude.iter()
            .chain(self.commands.iter())
            .zip(has_data.iter_mut())
            .flat_map(|(item, item_has_data)| {
                let maybe_data = item.data();
                *item_has_data = maybe_data.is_some();
                maybe_data
            })
            .collect();
        (json::Value::Array(data), has_data)
    }

    fn build_script<P>(&self, datafile: P, has_data: &[bool]) -> String
    where P: AsRef<Path>
    {
        let mut script =
            format!("\
                import json\n\
                datafile = open(\"{}\", \"r\")\n\
                alldata = json.loads(datafile.read())\n\
                datafile.close()\n",
                datafile.as_ref().display(),
            );
        if self.prelude.is_empty() {
            script.push_str(PRELUDE);
            script.push_str(INIT);
        }
        let mut data_count: usize = 0;
        let iter =
            self.prelude.iter()
            .chain(self.commands.iter())
            .zip(has_data);
        for (item, has_data) in iter {
            if *has_data {
                script.push_str(
                    &format!("data = alldata[{}]\n", data_count));
                data_count += 1;
            }
            script.push_str(&item.py_cmd());
            script.push('\n');
        }
        script
    }

    /// Build a Python script, but do not run it.
    pub fn code(&self, mode: Run) -> String {
        let mut tmp_json = get_temp_fname();
        tmp_json.set_extension("json");
        let (_, has_data) = self.collect_data();
        let mut script = self.build_script(&tmp_json, &has_data);
        match mode {
            Run::Show => {
                script.push_str("\nplt.show()");
            },
            Run::Save(outfile) => {
                script.push_str(
                    &format!("\nfig.savefig(\"{}\")", outfile.display()));
            }
            Run::SaveShow(outfile) => {
                script.push_str(
                    &format!("\nfig.savefig(\"{}\")", outfile.display()));
                script.push_str("\nplt.show()");
            },
            Run::Debug => { },
            Run::Build => { },
        }
        script
    }

    /// Build and run a Python script script in a [`Run`] mode.
    pub fn run(&self, mode: Run) -> MplResult<()> {
        let tmp = get_temp_fname();
        let mut tmp_json = tmp.clone();
        tmp_json.set_extension("json");
        let mut tmp_py = tmp.clone();
        tmp_py.set_extension("py");
        let (data, has_data) = self.collect_data();
        let mut script = self.build_script(&tmp_json, &has_data);
        match mode {
            Run::Show => {
                script.push_str("\nplt.show()");
            },
            Run::Save(outfile) => {
                script.push_str(
                    &format!("\nfig.savefig(\"{}\")", outfile.display()));
            },
            Run::SaveShow(outfile) => {
                script.push_str(
                    &format!("\nfig.savefig(\"{}\")", outfile.display()));
                script.push_str("\nplt.show()");
            },
            Run::Debug => { },
            Run::Build => { return Ok(()); },
        }
        let mut data_file = TempFile::new(&tmp_json)?;
        data_file.write_all(json::to_string(&data)?.as_bytes())?;
        data_file.flush()?;
        let mut script_file = TempFile::new(&tmp_py)?;
        script_file.write_all(script.as_bytes())?;
        script_file.flush()?;
        let res =
            process::Command::new("python3")
            .arg(format!("{}", tmp_py.display()))
            .output()?;
        if res.status.success() {
            Ok(())
        } else {
            let stdout: String =
                res.stdout.into_iter().map(char::from).collect();
            let stderr: String =
                res.stderr.into_iter().map(char::from).collect();
            Err(MplError::PyError(stdout, stderr))
        }
    }

    /// Alias for `self.run(Run::Show)`.
    pub fn show(&self) -> MplResult<()> { self.run(Run::Show) }

    /// Alias for `self.run(Run::Save(path))`.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> MplResult<()> {
        self.run(Run::Save(path.as_ref().to_path_buf()))
    }

    /// Alias for `self.run(Run::SaveShow(path))`
    pub fn saveshow<P: AsRef<Path>>(&self, path: P) -> MplResult<()> {
        self.run(Run::SaveShow(path.as_ref().to_path_buf()))
    }
}

/// A single subplot's position in a [`gridspec`][Mpl::new_gridspec].
///
/// The position is specified by two integer ranges representing a 2D slice of
/// the `gridspec`.
///
/// This type also allows for shared axes to be specified in the context of a
/// series of positions as a pair of integers: A given integer `k` refers to the
/// `Axes` object corresponding to the `k`-th position in the series; the first
/// is for the X-axis and the second is for the Y-axis.
///
/// The object
/// ```ignore
/// GSPos { i: 0..3, j: 2..3, sharex: Some(0), sharey: None }
/// ```
/// specifies a subplot covering the first three rows and second column of a
/// grid, sharing its X-axis with the first subplot in an implied sequence and
/// its Y-axis with no other.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GSPos {
    /// Vertical slice.
    pub i: Range<usize>,
    /// Horizontal slice.
    pub j: Range<usize>,
    /// Index of the `Axes` object with which to share the X-axis.
    pub sharex: Option<usize>,
    /// Index of the `Axes` object with which to share the Y-axis.
    pub sharey: Option<usize>,
}

impl GSPos {
    /// Create a new `GSPos` without any shared axes.
    pub fn new(i: Range<usize>, j: Range<usize>) -> Self {
        Self { i, j, sharex: None, sharey: None }
    }

    /// Create a new `GSPos` with shared axes.
    pub fn new_shared(
        i: Range<usize>,
        j: Range<usize>,
        sharex: Option<usize>,
        sharey: Option<usize>,
    ) -> Self
    {
        Self { i, j, sharex, sharey }
    }

    /// Set the axis sharing.
    pub fn share(mut self, axis: Axis2, target: Option<usize>) -> Self {
        match axis {
            Axis2::X => { self.sharex = target; },
            Axis2::Y => { self.sharey = target; },
            Axis2::Both => { self.sharex = target; self.sharey = target; },
        }
        self
    }

    /// Set the X-axis sharing.
    pub fn sharex(mut self, target: Option<usize>) -> Self {
        self.sharex = target;
        self
    }

    /// Set the Y-axis sharing.
    pub fn sharey(mut self, target: Option<usize>) -> Self {
        self.sharey = target;
        self
    }
}

/// Determines the final IO command(s) in the plotting script generated by an
/// [`Mpl`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Run {
    /// Call `pyplot.show` to display interactive figure(s).
    Show,
    /// Call `pyplot.Figure.savefig` to save the plot to a file.
    Save(PathBuf),
    /// `Save` and then `Show`.
    SaveShow(PathBuf),
    /// Perform no plotting IO, just build the script and call Python on it (for
    /// debugging purposes).
    Debug,
    /// Build the script, but don't call Python on it (for debugging purposes).
    Build,
}

impl<T: Matplotlib + 'static> From<T> for Mpl {
    fn from(item: T) -> Self {
        let mut mpl = Self::default();
        mpl.then(item);
        mpl
    }
}

impl std::ops::BitAnd<Mpl> for Mpl {
    type Output = Mpl;

    fn bitand(mut self, mut rhs: Mpl) -> Self::Output {
        self.prelude.append(&mut rhs.prelude);
        self.commands.append(&mut rhs.commands);
        self
    }
}

impl std::ops::BitAndAssign<Mpl> for Mpl {
    fn bitand_assign(&mut self, mut rhs: Mpl) {
        self.prelude.append(&mut rhs.prelude);
        self.commands.append(&mut rhs.commands);
    }
}

impl<T> std::ops::BitAnd<T> for Mpl
where T: Matplotlib + 'static
{
    type Output = Mpl;

    fn bitand(mut self, rhs: T) -> Self::Output {
        self.then(rhs);
        self
    }
}

impl<T> std::ops::BitAndAssign<T> for Mpl
where T: Matplotlib + 'static
{
    fn bitand_assign(&mut self, rhs: T) {
        self.then(rhs);
    }
}

impl std::ops::BitAnd<Run> for Mpl {
    type Output = Mpl;

    fn bitand(self, mode: Run) -> Self::Output {
        match self.run(mode) {
            Ok(_) => self,
            Err(err) => { panic!("error in Mpl::bitand: {err}"); },
        }
    }
}

impl std::ops::BitAndAssign<Run> for Mpl {
    fn bitand_assign(&mut self, mode: Run) {
        match self.run(mode) {
            Ok(_) => { },
            Err(err) => { panic!("error in Mpl::bitand_assign: {err}"); },
        }
    }
}

impl std::ops::BitOr<Run> for Mpl {
    type Output = ();

    fn bitor(self, mode: Run) -> Self::Output {
        match self.run(mode) {
            Ok(_) => (),
            Err(err) => { panic!("error in Mpl::bitor: {err}"); },
        }
    }
}

impl std::ops::BitOr<Run> for &Mpl {
    type Output = ();

    fn bitor(self, mode: Run) -> Self::Output {
        match self.run(mode) {
            Ok(_) => (),
            Err(err) => { panic!("error in Mpl::bitor: {err}"); },
        }
    }
}

