#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FontSize {
    XXSmall,
    XSmall,
    Smaller,
    Small,
    Medium,
    Large,
    Larger,
    XLarge,
    XXLarge,
    Pt(f32),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Alignment {
    Left,
    Center,
    Right,
}

pub trait RcLeaf {
    const KEY: &str;

    fn pyvalue(&self) -> String;
}

/// All items of `matplotlib.rcParams`.
///
/// All field names agree with key names with the exception of `axes.grid`
/// which, due to the existence of `axes.grid.axis` and `axes.grid.which`, has
/// been mapped to `axes.grid.on` in this library. All values returned by
/// [`Default`][RcParams::default] agree with Matplotlib defaults.
///
/// Most of the documentation for the items in this module is copied from the
/// [matplotlib documentation][mpl-customizing].
///
/// [mpl-customizing]: https://matplotlib.org/stable/users/explain/customizing.html
#[derive(Clone, Debug, PartialEq)]
pub struct RcParams {
    /// Agg rendering options.
    pub agg: RcAgg,
    /// Animation options.
    pub animation: RcAnimation,
    /// Axis styling options.
    pub axes: RcAxes,
    /// 3D axis styling options.
    pub axes3d: RcAxes3d,
    /// Agg rendering backend.
    pub backend: RcBackend,
    /// If the rendering backend conflicts with the GUI renderer in interactive
    /// mode, automatically find another, compatible one.
    pub backend_fallback: bool,
    /// Box plot options.
    pub boxplot: RcBoxplot,
    /// Contour plot options.
    pub contour: RcContour,
    /// Date formatting options.
    pub date: RcDate,
    /// Docstring generation options.
    pub docstring: RcDocstring,
    /// Errorbar plotting options.
    pub errorbar: RcErrorbar,
    /// Figure layout and rendering options.
    pub figure: RcFigure,
    /// Set fonts.
    pub font: RcFont,
    /// Plotting grid options.
    pub grid: RcGrid,
    /// Hatch drawing options.
    pub hatch: RcHatch,
    /// Histogram plotting options.
    pub hist: RcHist,
    /// Image plotting options.
    pub image: RcImage,
    /// Run in interactive mode.
    pub interactive: bool,
    /// Set keystroke mappings for the interactive GUI.
    pub keymap: RcKeymap,
    /// Legend drawing options.
    pub legend: RcLegend,
    /// Line drawing options.
    pub lines: RcLines,
    /// Mac OSX backend options.
    pub macosx: RcMacOSX,
    /// Marker drawing options.
    pub markers: RcMarkers,
    /// Font settings for math mode text.
    pub mathtext: RcMathtext,
    /// Patch drawing options.
    pub patch: RcPatch,
    /// Path drawing options.
    pub path: RcPath,
    /// Pixel coloring options.
    pub pcolor: RcPColor,
    /// Pixel coloring options for meshes.
    pub pcolormesh: RcPColormesh,
    /// PDF rendering options.
    pub pdf: RcPdf,
    /// PGF options for latex-style rendering.
    pub pgf: RcPgf,
    /// Polar axes options.
    pub polaraxes: RcPolaraxes,
    /// PS backend options.
    pub ps: RcPs,
    /// Options for `savefig`.
    pub savefig: RcSavefig,
    /// Scatter plot options.
    pub scatter: RcScatter,
    /// SVG rendering options.
    pub svg: RcSvg,
    /// Text drawing options.
    pub text: RcText,
    /// Set the timezone.
    ///
    /// This must be a `pytz` timezone string, e.g. `US/Central` or
    /// `Europe/Paris`.
    pub timezone: String,
    /// TK backend options.
    pub tk: RcTk,
    /// Toolbar options for the interactive GUI.
    ///
    /// One of {`None` (as a string), `toolbar2`, `toolmanager`}.
    pub toolbar: RcToolbar,
    /// WebAgg backend options.
    pub webagg: RcWebagg,
    /// X-axis drawing options.
    pub xaxis: RcXAxis,
    /// X-tick drawing options.
    pub xtick: RcXTick,
    /// Y-axis drawing options.
    pub yaxis: RcYAxis,
    /// Y-tick drawing options.
    pub ytick: RcYTick,
}

impl Default for RcParams {
    fn default() -> Self {
        agg: RcAgg::default(), 
        animation: RcAnimation::default(), 
        axes: RcAxes::default(), 
        axes3d: RcAxes3d::default(),
        backend: RcBackend::default(),
        backend_fallback: true,
        boxplot: RcBoxplot::default(), 
        contour: RcContour::default(), 
        date: RcDate::default(), 
        docstring: RcDocstring::default(), 
        errorbar: RcErrorbar::default(), 
        figure: RcFigure::default(), 
        font: RcFont::default(), 
        grid: RcGrid::default(), 
        hatch: RcHatch::default(), 
        hist: RcHist::default(), 
        image: RcImage::default(), 
        interactive: false,
        keymap: RcKeymap::default(), 
        legend: RcLegend::default(), 
        lines: RcLines::default(), 
        macosx: RcMacOSX::default(), 
        markers: RcMarkers::default(), 
        mathtext: RcMathtext::default(), 
        patch: RcPatch::default(), 
        path: RcPath::default(), 
        pcolor: RcPColor::default(), 
        pcolormesh: RcPColormesh::default(),
        pdf: RcPdf::default(), 
        pgf: RcPgf::default(), 
        polaraxes: RcPolaraxes::default(), 
        ps: RcPs::default(), 
        savefig: RcSavefig::default(), 
        scatter: RcScatter::default(), 
        svg: RcSvg::default(), 
        text: RcText::default(), 
        timezone: "UTC".into(),
        tk: RcTk::default(), 
        toolbar: RcToolbar::default(),
        webagg: RcWebagg::default(), 
        xaxis: RcXAxis::default(), 
        xtick: RcXTick::default(), 
        yaxis: RcYAxis::default(), 
        ytick: RcYTick::default(), 
    }
}

/// Agg rendering options.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct RcAgg {
    /// Agg backend path drawing options.
    pub path: RcAggPath,
}

/// Agg backend path drawing options.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RcAggPath {
    /// Split data sets into chunks of a given size.
    ///
    /// `0` to disable; values in the range 10000 to 100000 can improve speed
    /// significantly and prevent an Agg rendering failure when plotting very
    /// large data sets, especially if they are very gappy. It may cause minor
    /// artifacts, though. A value of 20000 is probably a good starting point.
    ///
    /// Default value: `0`
    pub chunksize: u32,
}

impl Default for RcAggPath {
    fn default() -> Self { Self { chunksize: 0 } }
}

/// Animation options.
#[derive(Clone, Debug, PartialEq)]
pub struct RcAnimation {
    /// Set the size/quality trade-off for the animation.
    ///
    /// Default value: `Auto`
    pub bitrate: RcAnimationBitrate,
    /// Codec to use for writing the animation.
    ///
    /// Default value: `h264`
    pub codec: String,
    /// Additional arguments to pass to the conversion utility.
    ///
    /// Default value: `["-layers", "OptimizePlus"]`
    pub convert_args: Vec<String>,
    /// Path to ImageMagick's convert binary. Unqualified paths are resolved by
    /// `subprocess.Popen`, except that on Windows, we look up the install of
    /// ImageMagick in the registry (as convert is also the name of a system
    /// tool).
    ///
    /// Default value: `convert`
    pub convert_path: String,
    /// Limit, in MB, of size of base64-encoded animation in HTML (i.e. IPython
    /// notebook).
    ///
    /// Default value: `20.0`
    pub embed_limit: f32,
    /// Additional arguments to pass to ffmpeg.
    ///
    /// Default value: `[]`
    pub ffmpeg_args: Vec<String>,
    /// Path to ffmpeg binary. Unqualified paths are resolved by
    /// `subprocess.Popen`.
    ///
    /// Default value: `ffmpeg`.
    pub ffmpeg_path: String,
    /// Controls the frame format used by temp files.
    ///
    /// Default value: `png`
    pub frame_format: String,
    /// Set the method for displaying an animation as HTML in the IPython
    /// notebook.
    ///
    /// Default value: `None`
    pub html: String,
    /// Animation writer "backend".
    ///
    /// Default value: `ffmpeg`.
    pub writer: String,
}

impl Default for RcAnimation {
    fn default() -> Self {
        Self {
            bitrate: RcAnimationBitrate::default(),
            codec: "h264".into(),
            convert_args: vec!["-layers".into(), "OptimizePlus".into()],
            convert_path: "convert".into(),
            embed_limit: 20.0,
            ffmpeg_args: vec![],
            ffmpeg_path: "ffmpeg".into(),
            frame_format: "png".into(),
            html: RcAnimationHtml::default(),
            writer: "ffmpeg".into(),
        }
    }
}

/// Set the size/quality trade-off for the animation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RcAnimationBitrate {
    Auto,
    Set(u32),
}

impl Default for RcAnimationBitrate {
    fn default() -> Self { Self::Auto }
}

/// Set the method for displaying an animation as HTML in the IPython notebook.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RcAnimationHtml {
    None,
    /// Use a HTML5 video tag.
    Html5,
    /// Use a JavaScript animation.
    JsHtml,
}

impl Default for RcAnimationHtml {
    fn default() -> Self { Self::None }
}

/// Axis styling options.
#[derive(Clone, Debug, PartialEq)]
pub struct RcAxes {
    /// Strategy to determine the limits of a plotting frame.
    ///
    /// Default value: `Data`
    pub autolimit_mode: RcAxesAutolimitMode,
    /// Where to to draw axis gridlines and ticks.
    ///
    /// Default value: `Lines`
    pub axisbelow: RcAxesAxisbelow,
    /// Axes edge color.
    ///
    /// Default value: `black`
    pub edgecolor: String,
    /// Axes background color.
    ///
    /// Default value: `white`
    pub facecolor: String,
    /// Formatting options for axis tick labels.
    pub formatter: RcAxesFormatter,
    /// Grid drawing options.
    pub grid: RcAxesGrid,
    /// Axis label color.
    ///
    /// Default value: `black`
    pub labelcolor: String,
    /// Space between axis and label.
    ///
    /// Default value: `4.0`
    pub labelpad: f32,
    /// Font size of the X- and Y-axis labels.
    ///
    /// Default value: `medium`
    pub labelsize: FontSize,
    /// Weight of the X- and Y-axis labels.
    ///
    /// Default value: `normal`
    pub labelweight: String,
    /// Axis edge line width.
    ///
    /// Default value: `0.8`
    pub linewidth: f32,
    /// Color cycle for plot lines as a list of color specs, each of which may
    /// be a single-letter color name, a long color name, or web-style hex
    /// (with leading "#").
    ///
    /// Default value: `[ ... ]`
    pub prop_cycle: Vec<String>,
    /// Axis spine options.
    pub spines: RcAxesSpines,
    /// Color of the title text.
    ///
    /// Default value: `auto`
    pub titlecolor: String,
    /// Alignment of the title.
    ///
    /// Default value: `Center`.
    pub titlelocation: Alignment,
    pub titlesize: String,
    pub titleweight: String,
    pub titley: Option<f32>,
    pub unicode_minus: bool,
    pub xmargin: f32,
    pub ymargin: f32,
    pub zmargin: f32,
}

impl Default for RcAxes {
    fn default() -> Self {
        Self {
            autolimit_mode: RcAxesAutolimitMode::default(),
            axisbelow: RcAxesAxisbelow::default(),
            edgecolor: "black".into(),
            facecolor: "white".into(),
            formatter: RcAxesFormatter::default(),
            grid: RcAxesGrid::default(),
            labelcolor: "black".into(),
            labelpad: 4.0,
            labelsize: FontSize::Medium,
            labelweight: "normal".into(),
            linewidth: 0.8,
            prop_cycle: vec![
                "#1f77b4".into(),
                "#ff7f0e".into(),
                "#2ca02c".into(),
                "#d62728".into(),
                "#9467bd".into(),
                "#8c564b".into(),
                "#e377c2".into(),
                "#7f7f7f".into(),
                "#bcbd22".into(),
                "#17becf".into(),
            ],
            spines: RcAxesSpines::default(),
            titlecolor: "auto".into(),
            titlelocation: RcAxesTitlelocation::default(),
            titlepad: 6.0,
            titlesize: FontSize::Large,
            titleweight: "normal".into(),
            titley: None,
            unicode_minus: true,
            xmargin: 0.05,
            ymargin: 0.05,
            zmargin: 0.05,
        }
    }
}

/// Strategy to determine the limits of a plotting frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RcAxesAutolimitMode {
    /// Use `xmargin` and `ymargin` as is.
    Data,
    /// After application of margins, axis limits are further expanded to the
    /// nearest round number.
    RoundNumbers,
}

impl Default for RcAxesAutolimitMode {
    fn default() -> Self { Self::Data }
}

/// Where to draw axis gridlines and ticks.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RcAxesAxisbelow {
    /// Below patches.
    True,
    /// Above patches but below lines.
    Lines,
    /// Above all.
    False,
}

impl Default for RcAxesAxisBelow {
    fn default() -> Self { Self::Lines }
}

/// Formatting options for axis tick labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RcAxesFormatter {
    /// Use scientific notation if log10 of the axis range is smaller than the
    /// first or larger than the second.
    ///
    /// Default value: `(-5, 6)`
    pub limits: (i32, i32),
    /// Minimum exponent to use in scientific notation.
    ///
    /// Default value: `0`
    pub min_exponent: i32,
    /// When `useoffset` is `true`, the offset will be used when it can remove
    /// at least this number of significant digits from tick labels.
    ///
    /// Default value: `4`
    pub offset_threshold: u32,
    /// When true, format tick labels according to the user's locale. For
    /// example, use "," as a decimal separator in the `fr_FR` locale.
    ///
    /// Default value: `false`
    pub use_locale: bool,
    /// When true, use math text for scientific notation.
    ///
    /// Default value: `false`
    pub use_mathtext: bool,
    /// If true, the tick label formatter will default to labeling ticks
    /// relative to an offset when the data range is small compared to the
    /// minimum absolute value of the data.
    ///
    /// Default value: `true`
    pub useoffset: bool,
}

impl Default for RcAxesFormatter {
    fn default() -> Self {
        Self {
            limits: (-5, 6),
            min_exponent: 0,
            offset_threshold: 4,
            use_locale: false,
            use_mathtext: false,
            useoffset: true,
        }
    }
}

/// Grid drawing options.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RcAxesGrid {
    /// Draw a coordinate grid.
    ///
    /// Default value: `false`
    pub on: bool,
    /// Draw coordinate grids for this axis.
    ///
    /// Default value: `Both`
    pub axis: RcAxesGridAxis,
    /// Draw coordinate grids for these axes ticks.
    ///
    /// Default value: `Major`
    pub which: RcAxesGridWhich,
}

impl Default for RcAxesGrid {
    fn default() -> Self {
        Self {
            on: false,
            axis: RcAxesGridAxis::default(),
            which: RcAxesGridWhich::default(),
        }
    }
}

/// Draw coordinate grids for this axis.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RcAxesGridAxis {
    /// Only the X-axis.
    X,
    /// Only the Y-axis.
    Y,
    /// Both the X- and Y-axes.
    Both,
}

impl Default for RcAxesGridAxis {
    fn default() -> Self { Self::Both }
}

/// Draw coordinate grids for these axes ticks.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RcAxesGridWhich {
    /// Only major ticks.
    Major,
    /// Only minor ticks.
    Minor,
    /// Both major and minor ticks.
    Both,
}

impl Default for RcAxesGridWhich {
    fn default() -> Self { Self::Major }
}

/// Axis spine options.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RcAxesSpines {
    /// Display the bottom axis spine.
    ///
    /// Default value: `true`
    pub bottom: bool,
    /// Display the left axis spine.
    ///
    /// Default value: `true`
    pub left: bool,
    /// Display the right axis spine.
    ///
    /// Default value: `true`
    pub right: bool,
    /// Display the top axis spine.
    ///
    /// Default value: `true`
    pub top: bool,
}

impl Default for RcAxesSpines {
    fn default() -> Self {
        Self {
            bottom: true,
            left: true,
            right: true,
            top: true,
        }
    }
}

/// Alignment of the title.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RcAxesTitleLocation {
    Left,
    Center,
    Right,
}

impl Default for RcAxesTitleLocation {
    fn default() -> Self { Self::Center }
}

/// Agg rendering backend.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RcBackend {
    MacOSX,
    QtAgg,
    Gtk4Agg,
    Gtk3Agg,
    TkAgg,
    WxAgg,
    Agg,
    QtCairo,
    GTK4Cairo,
    GTK3Cairo,
    TkCairo,
    WxCairo,
    Cairo,
    Ps,
    Pdf,
    Svg,
    Template,
}

impl Default for RcBackend {
    fn default() -> Self { Self::QtAgg }
}

/// Toolbar options for the interactive GUI.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RcToolbar {
    None,
    Toolbar2,
    ToolManager,
}

impl Default for RcBackend {
    fn default() -> Self { Self::Toolbar2 }
}

