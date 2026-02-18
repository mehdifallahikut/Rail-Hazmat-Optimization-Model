import matplotlib.pyplot as plt
import numpy as np

data = [
    {'alpha': 0.00, 'cost': 906.8, 'risk': 1231.36},
    {'alpha': 0.05, 'cost': 906.8, 'risk': 1231.36},
    {'alpha': 0.10, 'cost': 906.8, 'risk': 1231.36},
    {'alpha': 0.15, 'cost': 906.8, 'risk': 1231.36},
    {'alpha': 0.20, 'cost': 906.8, 'risk': 1231.36},
    {'alpha': 0.25, 'cost': 906.8, 'risk': 1231.36},
    {'alpha': 0.30, 'cost': 906.8, 'risk': 1231.36},
    {'alpha': 0.35, 'cost': 906.8, 'risk': 1231.36},
    {'alpha': 0.40, 'cost': 906.8, 'risk': 1231.36},
    {'alpha': 0.45, 'cost': 906.8, 'risk': 1231.36},
    {'alpha': 0.50, 'cost': 906.8, 'risk': 1231.36},
    {'alpha': 0.55, 'cost': 906.8, 'risk': 1231.36},
    {'alpha': 0.60, 'cost': 901.3, 'risk': 1240.86},
    {'alpha': 0.65, 'cost': 902.7, 'risk': 1237.26},
    {'alpha': 0.70, 'cost': 902.7, 'risk': 1237.26},
    {'alpha': 0.75, 'cost': 901.3, 'risk': 1240.86},
    {'alpha': 0.80, 'cost': 901.3, 'risk': 1240.86},
    {'alpha': 0.85, 'cost': 898.2, 'risk': 1258.12},
    {'alpha': 0.90, 'cost': 872.3, 'risk': 1405.88},   # Knee Point
    {'alpha': 0.95, 'cost': 860.0, 'risk': 1588.07},
    {'alpha': 1.00, 'cost': 835.7, 'risk': 2719.79}    # Max Risk
]

alphas = [d['alpha'] for d in data]
costs = [d['cost'] for d in data]
risks = [d['risk'] for d in data]

# --- (Paper Quality Style) ---
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.autolayout': True
})

plt.figure(figsize=(10, 6), dpi=150)

# 1.  (Pareto Frontier Line)
sorted_points = sorted(zip(costs, risks), key=lambda x: x[0], reverse=True)
u_costs, u_risks = zip(*sorted_points)

plt.plot(u_costs, u_risks, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

sc = plt.scatter(costs, risks, c=alphas, cmap='viridis', s=120, edgecolors='black', linewidth=1, zorder=2)

cbar = plt.colorbar(sc)
cbar.set_label(r'Weight Parameter ($\alpha$)', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

plt.xlabel('Total Operational Cost ($)', fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel('Total Societal Risk (Exposure × Prob)', fontsize=12, fontweight='bold', labelpad=10)
plt.title('Pareto Frontier: Cost vs. Risk Trade-off', fontsize=14, fontweight='bold', pad=15)

# 4. (Annotations)

#  (Knee Point) -
plt.annotate(f'Knee Point\n(Best Trade-off)\n($\\alpha$=0.90)',
             xy=(872.3, 1405.88), xytext=(880, 1800),
             arrowprops=dict(facecolor='#1f77b4', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, color='#004080', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.9))

#  (Min Risk)
plt.annotate('Min Risk\n(Conservative)',
             xy=(906.8, 1231.36), xytext=(895, 1100),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             fontsize=9, ha='center')

#  (Max Risk)
plt.annotate('High Risk Spike\n(Cost Priority)',
             xy=(835.7, 2719.79), xytext=(850, 2500),
             arrowprops=dict(facecolor='#d62728', shrink=0.05, width=1.5),
             fontsize=9, color='#d62728', fontweight='bold')

plt.grid(True, linestyle=':', alpha=0.7)

plt.show()