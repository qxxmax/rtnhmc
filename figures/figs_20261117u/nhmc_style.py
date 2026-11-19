"""Unified figure style for the NHMC manuscript (v10), 2026-11-17.

Import from every plotting script in this folder:  import nhmc_style as NS; NS.apply()
Only styling lives here (fonts, sizes, colours, markers, panel letters, condition tags, legends); no data.

Spec
- sans fonts (DejaVu Sans, mathtext dejavusans); axis labels 8, ticks 7, legend 7, panel letters 9 bold
- no grid, top/right spines off; PDF output, pdf.fonttype 42
- widths: full 7.0 in, single panel 3.4 in; row height ~2.0-2.2 in
- panel letters: bold lowercase a, b, c ... top-left outside each axes (panel_letter)
- condition tags: math text inside the axes, top-left (or top-right), size 7 (tag)
- methods: fixed label / colour / marker (METHOD); reference curves black/dark-grey dotted, bands #d9d9d9, fits dashed dark grey
- lattice size L: fixed sequential Blues (LCOL); kappa: viridis with fixed normalisation [0.20, 0.30] (kcol)
- legends frameless; multi-panel figures: one shared legend centred above the panels (fig_legend)
"""
import matplotlib
import matplotlib.pyplot as plt

FULL_W = 7.0
SINGLE_W = 3.4
ROW_H = 2.1

# ------------------------------------------------------------------ methods
METHOD = {
    'path-IMH': dict(color='#D55E00', marker='s'),
    'RT':       dict(color='#0072B2', marker='o'),
    'pRT':      dict(color='#009E73', marker='^'),
    'HMC':      dict(color='#555555', marker='v', open=True),
    'Wolff':    dict(color='#CC79A7', marker='D', open=True),
    'CMCD':     dict(color='#E69F00', marker='o'),
    'DBS':      dict(color='#56B4E9', marker='o'),
    'NHMC':     dict(color='#0072B2', marker='o'),
}
COL = {k: v['color'] for k, v in METHOD.items()}
MRK = {k: v['marker'] for k, v in METHOD.items()}
COL['IMH'] = COL['path-IMH']          # internal keys of the old scripts map onto the unified labels
MRK['IMH'] = MRK['path-IMH']
LABEL = {'IMH': 'path-IMH', 'imh': 'path-IMH', 'imhG': 'path-IMH', 'RT': 'RT', 'rt': 'RT', 'pRT': 'pRT', 'rtp12': 'pRT'}

# BAR-type estimators (forward / reverse / BAR)
BAR_STYLE = {
    'forward': dict(color='#CC79A7', marker='s', open=True),
    'reverse': dict(color='#E69F00', marker='^', open=True),
    'BAR':     dict(color='#0072B2', marker='o', open=False),
}

REF_COLOR = '0.15'                    # reference curves (TI, exact, Wolff / HMC means): dotted
REF_LS = ':'
BAND = '#d9d9d9'                      # uncertainty bands, |z| < 2 bands
FIT_COLOR = '0.3'                     # fits: dashed dark grey
FIT_LS = '--'
ZERO_COLOR = '0.5'                    # zero lines

# ------------------------------------------------------------------ lattice size L and kappa
LCOL = {4: '#9ecae1', 6: '#6baed6', 8: '#4292c6', 10: '#2b7bba', 12: '#1764ab', 16: '#08306b',
        24: '#807dba', 32: '#3f007d'}
# rectangular N_x x 8 lattices (BAR prefix figure): same sequential blues, ordered by size
NXCOL = {64: '#9ecae1', 128: '#4292c6', 256: '#1764ab', 512: '#08306b'}
KAPPA_NORM = matplotlib.colors.Normalize(0.20, 0.30)
KAPPA_CMAP = plt.get_cmap('viridis')


def kcol(kappa):
    """colour of a coupling kappa (float, e.g. 0.27), fixed normalisation over [0.20, 0.30]."""
    return KAPPA_CMAP(KAPPA_NORM(kappa))


def kappa_colorbar(fig, ax, **kw):
    sm = matplotlib.cm.ScalarMappable(cmap=KAPPA_CMAP, norm=KAPPA_NORM)
    cb = fig.colorbar(sm, ax=ax, **kw)
    cb.set_label(r'$\kappa$', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.5)
    return cb


# ------------------------------------------------------------------ rc
RC = {
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'mathtext.fontset': 'dejavusans',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.columnspacing': 1.2,
    'legend.handletextpad': 0.5, 'legend.borderaxespad': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
    'lines.linewidth': 1.0, 'lines.markersize': 3.5, 'errorbar.capsize': 0,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 200,
}


def apply():
    plt.rcParams.update(RC)


# ------------------------------------------------------------------ annotations
def panel_letter(ax, s, x=None, y=1.02):
    """bold lowercase letter at the top-left, outside the axes. x is measured in inches from the axes' left edge
    (default 0.38 in), so letters line up regardless of the panel width."""
    fig = ax.figure
    w_in = ax.get_position().width * fig.get_figwidth()
    dx = -(0.38 if x is None else x) / max(w_in, 1e-3)
    ax.text(dx, y, s, transform=ax.transAxes, fontweight='bold', fontsize=9, va='bottom', ha='left')


def panel_letters(axes, letters='abcdefghijklmnop', **kw):
    for a, s in zip(axes, letters):
        panel_letter(a, s, **kw)


def tag(ax, s, loc='upper left', **kw):
    """short condition tag (math text) inside the axes; loc 'upper left' / 'upper right' / 'lower left' / 'lower right'."""
    x = 0.04 if 'left' in loc else 0.96
    y = 0.95 if 'upper' in loc else 0.05
    kw.setdefault('fontsize', 7)
    ax.text(x, y, s, transform=ax.transAxes, ha='left' if 'left' in loc else 'right',
            va='top' if 'upper' in loc else 'bottom', **kw)


def fig_legend(fig, handles, labels=None, ncol=None, y=1.0, **kw):
    """one shared frameless legend centred above the panels (anchored at the top edge of the figure)."""
    if labels is None:
        labels = [h.get_label() for h in handles]
    kw.setdefault('frameon', False)
    return fig.legend(handles, labels, loc='upper center', ncol=ncol or len(handles), bbox_to_anchor=(0.5, y), **kw)


# ------------------------------------------------------------------ legend handles
def mhandle(name, line=True, ls='-', ms=4, label=None, filled=None):
    st = METHOD[name]
    filled = (not st.get('open', False)) if filled is None else filled
    return plt.Line2D([], [], color=st['color'], marker=st['marker'], ls=ls if line else '', ms=ms,
                      mfc=st['color'] if filled else 'white', mec=st['color'], label=label or name)


def lhandle(L, label=None, **kw):
    kw.setdefault('lw', 1.2)
    return plt.Line2D([], [], color=LCOL[L], label=label or f'$L={L}$', **kw)


def khandle(kappa, label=None, **kw):
    kw.setdefault('lw', 2)
    return plt.Line2D([], [], color=kcol(kappa), label=label or fr'$\kappa={kappa:.2f}$', **kw)


def save(fig, path):
    """PDF only (vector); bbox tight with a small pad so outside panel letters are kept."""
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
