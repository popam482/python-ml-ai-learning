import numpy as np
import matplotlib.pyplot as plt

np.random.seed(101)
hours = np.arange(0, 24)
cpu = np.random.uniform(20, 95, 24)
ram = np.random.uniform(40, 85, 24)
temp = cpu * 0.5 + np.random.normal(30, 5, 24)
disk_io = np.random.randint(100, 1000, 24)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

axes[0, 0].plot(hours, cpu, color='red', label='CPU usage')
axes[0, 0].plot(hours, ram, color='purple', linestyle='--', label='RAM usage')
axes[0, 0].set_title('CPU and RAM usage')

max_idx = np.argmax(cpu)
max_x = hours[max_idx]
max_y = cpu[max_idx]

axes[0, 0].annotate(f'Max CPU: {max_y}',
                    xy=(max_x, max_y),
                    xytext=(max_x + 0.5, max_y + 0.5),
                    arrowprops=dict(facecolor='red', shrink=0.001, width=2, headwidth=5),
                    fontsize=10)

axes[0, 0].set_title('CPU and RAM usage')
axes[0, 0].set_xlabel('Hours')
axes[0, 0].set_ylabel('CPU usage')
axes[0, 0].legend()
axes[0, 0].grid(True)

ax1 = axes[0, 1]
ax2 = ax1.twinx()

line = ax1.plot(
    hours,
    temp,
    color='red',
    linewidth=2,
    label='Temperature'
)

ax1.set_xlabel('Hours')
ax1.set_ylabel('Temperature (C)', color='red')
ax1.tick_params(axis='y', labelcolor='red')

bars = ax2.bar(
    hours,
    disk_io,
    color='blue',
    alpha=0.3,
    label='Disk IO Operations'
)

ax2.set_ylabel('Disk IO Operations', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

ax1.legend([line[0], bars], ['Temperature', 'Disk IO Ops'],
           loc='upper left')

ax1.set_title('Temperature and Disk IO')
ax1.grid(True)

n, bins, patches = axes[1, 0].hist(temp, bins=10, density=False, color='olive', edgecolor='black', alpha=0.8)
axes[1, 0].set_title('Temperature distribution')
axes[1, 0].set_xlabel('Measured temperature')
axes[1, 0].set_ylabel('Frequency')

mean_temp = np.mean(temp)

axes[1, 0].axvline(
    mean_temp,
    color='red',
    linestyle=':',
    linewidth=2,
    label=f'Mean = {mean_temp:.2f}'
)

axes[1, 1].scatter(cpu, temp, color='red', s=disk_io)
axes[1, 1].set_title('CPU usage - temperature scatter')


plt.tight_layout()
plt.suptitle('Charts')

axes[0,0].spines['top'].set_visible(False)
axes[0,0].spines['right'].set_visible(False)

axes[0,1].spines['top'].set_visible(False)
axes[0,1].spines['right'].set_visible(False)

axes[1,0].spines['top'].set_visible(False)
axes[1,0].spines['right'].set_visible(False)

axes[1,1].spines['top'].set_visible(False)
axes[1,1].spines['right'].set_visible(False)

plt.show()

fig.savefig('../exports/system_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')