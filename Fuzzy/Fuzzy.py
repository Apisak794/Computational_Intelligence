import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Generate universe variables
#   * Quality and service on subjective ranges [0, 10]
#   * Tip has a range of [0, 1] in units of percentage points
x_weight = np.arange(0, 11, 1)
x_number = np.arange(0, 11, 1)
x_option  = np.arange(0, 101, 1)

# Generate fuzzy membership functions
weight_light = fuzz.trimf(x_weight, [0, 0, 2])
weight_medium = fuzz.trimf(x_weight, [0, 10, 10])
weight_heavy = fuzz.trimf(x_weight, [1.5, 7, 10])

number_little = fuzz.trimf(x_number, [0, 0, 2])
number_moderate = fuzz.trimf(x_number, [0, 10, 10])
number_Alot = fuzz.trimf(x_number, [1.5, 7, 10])

option_fill = fuzz.trimf(x_option, [0, 0, 20])
option_nothing = fuzz.trimf(x_option, [0, 70, 100])
option_fritter = fuzz.trimf(x_option, [15, 87, 100])


# Visualize these universes and membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_weight, weight_light, 'b', linewidth=1.5, label='light')
ax0.plot(x_weight, weight_medium, 'g', linewidth=1.5, label='medium')
ax0.plot(x_weight, weight_heavy, 'r', linewidth=1.5, label='heavy')
ax0.set_title('garbage weight')
ax0.legend()

ax1.plot(x_number, number_little, 'b', linewidth=1.5, label='little')
ax1.plot(x_number, number_moderate, 'g', linewidth=1.5, label='moderate')
ax1.plot(x_number, number_Alot, 'r', linewidth=1.5, label='A lot')
ax1.set_title('garbage number')
ax1.legend()

ax2.plot(x_option, option_fill, 'b', linewidth=1.5, label='fill')
ax2.plot(x_option, option_nothing, 'g', linewidth=1.5, label='nothing')
ax2.plot(x_option, option_fritter, 'r', linewidth=1.5, label='fritter')
ax2.set_title('option')
ax2.legend()

garbage_weight = 7
garbage_number = 3

weight_level_light = fuzz.interp_membership(x_weight, weight_light, garbage_weight)
weight_level_medium = fuzz.interp_membership(x_weight, weight_medium, garbage_weight)
weight_level_heavy = fuzz.interp_membership(x_weight, weight_heavy, garbage_weight)

number_level_little = fuzz.interp_membership(x_number, number_little, garbage_number)
number_level_moderate = fuzz.interp_membership(x_number, number_moderate, garbage_number)
number_level_Alot = fuzz.interp_membership(x_number, number_Alot, garbage_number)

# Now we take our rules and apply them. Rule 1 concerns bad food OR service.
# The OR operator means we take the maximum of these two.

# เติมเมื่อ weight or number < 4
active_rule1 = np.fmax( weight_level_light, number_level_little)
option_activation_fill = np.fmin(active_rule1, option_fill)
# เฉยๆเมื่อ weight or number ระหว่าง 4 - 6
active_rule2 = np.fmax(weight_level_medium, number_level_moderate)
option_activation_nothing = np.fmin(active_rule2, option_nothing)
# ทิ้งเมื่อ weight or number > 6
active_rule3 = np.fmax(weight_level_heavy, number_level_Alot)
option_activation_fritter = np.fmin(active_rule3, option_fritter)

option0 = np.zeros_like(x_option)

fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_option, option0, option_activation_fill, facecolor='b', alpha=0.7)
ax0.plot(x_option, option_fill, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_option, option0, option_activation_nothing, facecolor='g', alpha=0.7)
ax0.plot(x_option, option_nothing, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_option, option0, option_activation_fritter, facecolor='r', alpha=0.7)
ax0.plot(x_option, option_fritter, 'r', linewidth=0.5, linestyle='--', )
ax0.set_title('Output membership activity')

aggregated = np.fmax(option_activation_fill, np.fmax(option_activation_nothing,option_activation_fritter))


# Calculate defuzzified result
option = fuzz.defuzz(x_option, aggregated, 'centroid')
option_activation = fuzz.interp_membership(x_option, aggregated, option)  # for plot

print('ความต้องการทิ้งขยะ = ' + np.str(option) + ' % ')
if(option > 60):
    print('ทิ้งขยะออกจากถัง')
elif(option > 54):
    print('ไม่ทำอะไร')
else:
    print('เติมขยะเข้าถัง')
    
# Visualize this
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_option, option_fill, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_option, option_nothing, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_option, option_fritter, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_option, option0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([option, option], [0, option_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

# fritter > 60 , 54 < nothing < 60 ,fill < 54
print(option)
plt.tight_layout()
plt.show()
