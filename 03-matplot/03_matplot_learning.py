import matplotlib.pyplot as plt
import numpy as np

minutes = [0, 10, 20, 30, 40, 50]
motor_temperature = [20, 65, 85, 88, 90, 89]

# plt.plot(minutes, motor_temperature)

plt.plot(minutes, motor_temperature, color='green', marker='o', linestyle='--')
plt.title('Car motor temperature')
plt.xlabel('Time')
plt.ylabel('Temperature(C)')
plt.grid(True)

plt.show()

# bar chart

languages = ['C++', 'Java', 'Python']
code_lines = [1000, 3200, 2500]

plt.bar(languages, code_lines, color=['blue', 'orange', 'yellow'])
plt.title('Programming Languages activity')
plt.xlabel('Language')
plt.ylabel('Code Lines')
plt.grid(True)
plt.savefig('language_activity.png')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
fig.suptitle('Principal title', fontsize=14, fontweight='bold')

ax.set_title('Graphic specific title')
ax.set_xlabel('X axis [units]', fontsize=10)
ax.set_ylabel('Y axis [units]', fontsize=10)

plt.show()

# line plot

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, y1, color='green', linestyle='-', linewidth=2.5, label='sin(x)', zorder=1)
ax.plot(x, y2, color='blue', linestyle='-', linewidth=2.5, label='cos(x)', zorder=1)

ax.grid(True)
ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.8)
plt.show()

# scatter plot
np.random.seed(42)
n = 100
pressure = np.random.uniform(1, 100, n)
temperature = pressure * 1.5 + np.random.normal(0, 10, n)
vibration = np.random.uniform(50, 100, n)

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(pressure, temperature, c=temperature, s=vibration * 4, cmap='viridis', alpha=0.7,
                     edgecolors='black', linewidths=1)

cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label('temperature scale(C)')

ax.set_xlabel('Pressure (bar)')
ax.set_ylabel('Thermic response (C)')
ax.set_title('Sensor analysis')

plt.show()

# bar charts

category = ['Mode A', 'Mode B', 'Mode C', 'Mode D']
cpu_execution_time = [120, 150, 200, 180]
io_execution_time = [40, 20, 100, 50]

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(category, cpu_execution_time, label='CPU time', color='orange')
ax.bar(category, io_execution_time, label='IO time', color='red')
ax.legend()
plt.show()

# grouped bar chart

x_indices = np.arange(len(category))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar((x_indices - bar_width) / 2, cpu_execution_time, label='CPU time', color='pink')
ax.bar((x_indices + bar_width) / 2, io_execution_time, label='IO time', color='purple')

ax.set_xticks(x_indices)
ax.set_xticklabels(category)
ax.legend()

plt.show()

# histograms

data_noise = np.random.normal(loc=50, scale=10, size=1000)

fig, ax = plt.subplots(figsize=(10, 6))

n, bins, patches = ax.hist(data_noise, bins=30, density=False, color='olive', edgecolor='black', alpha=0.8)

ax.set_title('Noise signal distribution')
ax.set_xlabel('Measured value')
ax.set_ylabel('Frequency')

plt.show()

# box plots - quartiles, outliers

sensor_data_1 = np.random.normal(100, 10, 200)
sensor_data_2 = np.random.normal(105, 25, 200)
sensor_data_2 = np.append(sensor_data_2, [190, -20])  # outliers

fig, ax = plt.subplots(figsize=(10, 6))

ax.boxplot([sensor_data_1, sensor_data_2], label=['stable sensor', 'sensor with noise'], patch_artist=True,
           boxprops=dict(facecolor='white'), medianprops=dict(color='blue'))

ax.set_ylabel('recorded value')
ax.set_title('Anomaly diagnostics')
ax.grid(axis='y', linestyle='--', linewidth=0.5)

plt.show()

# grid indexing

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

axes[0, 0].plot(x, y1, color='red')
axes[0, 0].plot(x, y2, color='purple')
axes[0, 0].set_title('Sinus')

axes[0, 1].scatter(pressure, temperature, color='blue', s=10)
axes[0, 1].set_title('Pressure/Temp scatter')

axes[1, 0].bar(category, cpu_execution_time, label='CPU time', color='orange')
axes[1, 0].set_title('CPU performance')

axes[1, 1].hist(data_noise, bins=15, color='green')
axes[1, 1].set_title('Noise histogram')

plt.tight_layout()

plt.show()

# ax.twinx()

time = np.arange(0, 24, 1)
mw_consumption = [500, 480, 470, 490, 530, 620, 780, 850, 900, 880, 860, 840,
                  830, 820, 810, 830, 890, 950, 980, 920, 850, 750, 650, 560]
price_euro = [45, 40, 38, 35, 40, 65, 90, 120, 130, 110, 100, 95,
              90, 88, 85, 92, 115, 145, 160, 135, 105, 80, 60, 50]

fig, ax1 = plt.subplots(figsize=(10, 5))

# first axis (left)
color_mw = 'blue'
ax1.set_xlabel('Hour')
ax1.set_ylabel('MW consumption', color=color_mw)
line1 = ax1.plot(time, mw_consumption, color=color_mw, linewidth=2.5, label='Consumption')
ax1.tick_params(axis='y', labelcolor=color_mw)

ax2 = ax1.twinx()
color_price = 'red'
ax2.set_ylabel('Price (EUR/Mwh)', color=color_price)
line2 = ax2.plot(time, price_euro, color=color_price, linewidth=2.5, label='Price')
ax2.tick_params(axis='y', labelcolor=color_price)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True)
ax1.set_title('Consumption vs Price')
ax.grid(True)

plt.show()

# eliminate borders & styling

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y1, color='#2c3e50', linewidth=2)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

plt.show()

# arrow notations

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(time, mw_consumption, color='magenta', linewidth=2)

max_idx = np.argmax(mw_consumption)
max_time = time[max_idx]
max_val = mw_consumption[max_idx]

ax.annotate(f'Peak: {max_val} MW',
            xy=(max_time, max_val),
            xytext=(max_time - 5, max_val - 100),
            arrowprops=dict(facecolor='red', shrink=0.001, width=2, headwidth=5),
            fontsize=10, fontweight='bold', color='red')

ax.set_title('Automatically mark critic points')
plt.show()

# file export
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y1)
ax.set_title('Export test')
fig.savefig('../exports/report.png', dpi=300, bbox_inches='tight', facecolor='white')