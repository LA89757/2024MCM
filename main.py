import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import pandas as pd

print(pd.__version__)

# 元胞状态和植被类型
EMPTY = 0
BURNING = 1
BURNT = 2

VEGETATION_TYPES = {
    'grass': {'spread_chance': 0.3, 'humidity': 0.2},
    'trees': {'spread_chance': 0.2, 'humidity': 0.3},
    'bushes': {'spread_chance': 0.4, 'humidity': 0.25}
}

GRID_SIZE = 50

# 初始化网格
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=[('state', 'i4'), ('humidity', 'f4'), ('height', 'f4'), ('vegetation', 'U10'), ('temperature', 'f4'), ('rainfall', 'f4'), ('slope', 'f4')])

# 初始条件
grid['state'][GRID_SIZE//2, GRID_SIZE//2] = BURNING
grid['humidity'] = np.random.rand(GRID_SIZE, GRID_SIZE) * 0.3
grid['height'] = np.random.rand(GRID_SIZE, GRID_SIZE)
grid['vegetation'] = np.random.choice(list(VEGETATION_TYPES.keys()), size=(GRID_SIZE, GRID_SIZE))
grid['temperature'] = np.random.uniform(20, 40, size=(GRID_SIZE, GRID_SIZE))  # 假设温度在20到40度之间
grid['rainfall'] = np.random.uniform(0, 5, size=(GRID_SIZE, GRID_SIZE))  # 假设降雨量在0到5mm之间
grid['slope'] = np.random.uniform(0, 10, size=(GRID_SIZE, GRID_SIZE))  # 假设坡度在0到10度之间

wind_direction = (0, 1)  # 初始化风向
wind_speed = 2  # 初始化风速

def update_wind(wind_direction, wind_speed):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    wind_direction = random.choice(directions)
    wind_speed = np.random.uniform(1, 3)
    return wind_direction, wind_speed

def update_grid(grid, wind_direction, wind_speed, frame):
    new_grid = np.copy(grid)
    # 随机产生新的起火点
    if frame % 10 == 0:  # 每10帧产生一个新的起火点
        new_fire_x = GRID_SIZE//2 + random.randint(-5, 5)
        new_fire_y = GRID_SIZE//2 + random.randint(-5, 5)
        if 0 <= new_fire_x < GRID_SIZE and 0 <= new_fire_y < GRID_SIZE:
            new_grid['state'][new_fire_x, new_fire_y] = BURNING

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid['state'][i, j] == BURNING:
                new_grid['state'][i, j] = BURNT
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE and grid['state'][ni, nj] == EMPTY:
                        # 考虑温度、降雨和坡度的影响
                        temperature_factor = (grid['temperature'][ni, nj] - 20) / 20
                        rainfall_factor = -grid['rainfall'][ni, nj] / 5
                        slope_factor = -grid['slope'][ni, nj] / 10 if (di, dj) == (1, 0) else grid['slope'][ni, nj] / 10
                        veg_type = grid['vegetation'][ni, nj]
                        spread_chance = VEGETATION_TYPES[veg_type]['spread_chance']
                        spread_chance += temperature_factor + rainfall_factor + slope_factor
                        if (di, dj) == wind_direction:
                            spread_chance += 0.1 * wind_speed
                        spread_chance -= grid['humidity'][ni, nj]
                        spread_chance += np.random.uniform(-0.05, 0.05)
                        if np.random.rand() < spread_chance:
                            new_grid['state'][ni, nj] = BURNING
    return new_grid

fig, ax = plt.subplots()
im = ax.imshow(grid['state'], cmap='hot', interpolation='nearest')

def init():
    im.set_data(grid['state'])
    return [im]

def update(frame):
    global grid, wind_direction, wind_speed
    wind_direction, wind_speed = update_wind(wind_direction, wind_speed)
    grid = update_grid(grid, wind_direction, wind_speed, frame)
    im.set_data(grid['state'])
    return [im]

ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True, interval=200)

plt.show()