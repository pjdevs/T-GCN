import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

# Data and data params
start_date = dt.datetime(2015, 1, 31)
sample_time = 5 * 60
features = pd.read_csv("data/los_speed.csv", header=0).values

# Plot params
road_index = 100
ticks_interval = 12

# Plot
road_speeds = features[:,road_index].T[:100]
times = list(range(0, len(road_speeds)*sample_time, sample_time))
time_ticks = [(start_date + dt.timedelta(seconds=t)).strftime("%H:%M") for t in times]
plot_indices = [i for i in range(len(times)) if i % ticks_interval == 0]

fig, axes = plt.subplots()

axes.plot(times, road_speeds)
axes.set_xticks(np.take(times, plot_indices))
axes.set_xticklabels(np.take(time_ticks, plot_indices))

plt.xlabel("time (hours:minutes)")
plt.ylabel("speed (km/h)")
plt.show()