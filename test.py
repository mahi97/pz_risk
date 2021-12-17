import numpy as np
import matplotlib.pyplot as plt

win_rate = np.array([
    [[0.417, 0.583, 0.],  # 1 vs 1
     [0.255, 0.745, 0.]],  # 1 vs 2
    [[0.579, 0.421, 0.],  # 2 vs 1
     [0.228, 0.324, 0.448]],  # 2 vs 2
    [[0.660, 0.340, 0.],  # 3 vs 1
     [0.371, 0.336, 0.293]]  # 3 vs 2
])

d3 = {}

def get_chance(attack_unit, defense_unit, left):
    global win_rate, d3
    i_a = min(attack_unit - 1, 2)
    i_d = min(defense_unit - 1, 1)
    if (attack_unit, defense_unit, left) in d3:
        c = d3[(attack_unit, defense_unit, left)]
        return c

    c = 0.0
    if left < -defense_unit or left > attack_unit:
        c = 0.0
    elif defense_unit < 0 or attack_unit < 0:
        c = 0.0
    elif attack_unit == 0:
        if left == -defense_unit:
            c = 1.0
        else:
            c = 0.0
    elif defense_unit == 0:
        if left == attack_unit:
            c = 1.0
        else:
            c = 0.0
    else:
        c = win_rate[i_a, i_d, 0] * get_chance(attack_unit, defense_unit - min(min(i_a, i_d) + 1, 2), left) + \
            win_rate[i_a, i_d, 1] * get_chance(attack_unit - 1, defense_unit - min(i_a, 1), left) + \
            win_rate[i_a, i_d, 2] * get_chance(attack_unit - 2, defense_unit, left)
    d3[(attack_unit, defense_unit, left)] = c
    return c

k, j = 3, 1

b = range(-j, k+1)
a = [get_chance(k, j, i) for i in b]

print(np.argmax(a), range(-j, k+1)[np.argmax(a)])

plt.plot(b, a)
plt.show()