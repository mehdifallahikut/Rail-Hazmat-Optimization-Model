import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# --- داده‌های Gap ---

# === Small Instances ===
small_m1 = [85, 48, None]
small_m2 = [2, 14, 66]
small_m3 = [0, 0, 0]

# === Medium Instances ===
med_m1 = [72, None, None]
med_m2 = [60, None, None]
med_m3 = [0, 0, 0]

# === Large Instances ===
large_m1 = [None, None, None]
large_m2 = [None, None, None]
large_m3 = [0.7, 2.6, 3.9]

# --- تنظیمات رسم ---
groups = ['Train Len: 100', 'Train Len: 150', 'Train Len: 250']
x = np.arange(len(groups))
width = 0.25

plt.rcParams.update({'font.size': 10, 'font.family': 'sans-serif'})
fig, axs = plt.subplots(1, 3, figsize=(16, 6), dpi=150, sharey=True)


def plot_smart_bar(ax, data, offset, color, label_name):
    plot_vals = []
    bar_colors = []
    hatches = []
    bar_alphas = []

    for v in data:
        if v is None:
            plot_vals.append(100)
            bar_colors.append(color)
            hatches.append('///')
            bar_alphas.append(0.5)
        else:
            plot_vals.append(v)
            bar_colors.append(color)
            hatches.append('')
            bar_alphas.append(0.9)

    bars = ax.bar(x + offset, plot_vals, width, label=label_name,
                  color=bar_colors, edgecolor='black', alpha=0.9, zorder=3)

    for i, (bar, hatch) in enumerate(zip(bars, hatches)):
        bar.set_hatch(hatch)
        if data[i] is None:
            bar.set_alpha(0.6)

    for i, rect in enumerate(bars):
        val = data[i]
        h = rect.get_height()

        if val is None:
            txt = 'No Sol'
            txt_col = '#8a1c1c'
            font_w = 'bold'
            y_pos = 50
            bg_col = 'white'
        elif val == 0:
            txt = '0.0%'
            txt_col = 'green'
            font_w = 'bold'
            y_pos = h + 2
            bg_col = None
        else:
            txt = f'{val}%'
            txt_col = 'black'
            font_w = 'normal'
            y_pos = h + 2
            bg_col = None

        t = ax.text(rect.get_x() + rect.get_width() / 2, y_pos, txt,
                    ha='center', va='bottom', fontsize=9,
                    color=txt_col, fontweight=font_w, rotation=0)

        if bg_col:
            t.set_bbox(dict(facecolor=bg_col, alpha=0.7, edgecolor='none', pad=1))


def setup_subplot(ax, d1, d2, d3, title):
    plot_smart_bar(ax, d1, -width, '#d9534f', 'M1 (Monolithic)')
    plot_smart_bar(ax, d2, 0, '#f0ad4e', 'M2 (Type-Based)')
    plot_smart_bar(ax, d3, width, '#28a745', 'M3 (LBBD)')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15)
    ax.set_ylim(0, 115)
    ax.grid(True, axis='y', ls="-", alpha=0.3, zorder=0)


setup_subplot(axs[0], small_m1, small_m2, small_m3, "Small Instances (3 Blocks)")
setup_subplot(axs[1], med_m1, med_m2, med_m3, "Medium Instances (5 Blocks)")
setup_subplot(axs[2], large_m1, large_m2, large_m3, "Large Instances (7 Blocks)")

axs[0].set_ylabel('Optimality Gap (%)', fontsize=12, fontweight='bold')

legend_elements = [
    Patch(facecolor='#d9534f', edgecolor='black', label='M1 (Monolithic)'),
    Patch(facecolor='#f0ad4e', edgecolor='black', label='M2 (Type-Based)'),
    Patch(facecolor='#28a745', edgecolor='black', label='M3 (LBBD)'),
    Patch(facecolor='white', hatch='///', edgecolor='black', label='No Solution (Hatched)')
]

axs[1].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10)

plt.suptitle("Optimality Gap Analysis (Lower is Better)", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
# plt.savefig('gap_analysis_colored.png', bbox_inches='tight', dpi=300)
plt.show()