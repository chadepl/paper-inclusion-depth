

import numpy as np
from mpmath import mp
import matplotlib.pyplot as plt

############################
# SETUP THE SHAPE ENSEMBLE #
############################

shape_a = np.ones((4, 4), dtype=int)

shape_b = np.zeros((4, 4), dtype=int)
shape_b[1:3, 1:3] = 1

shape_c = np.zeros((4, 4), dtype=int)
shape_c[0:1, 0:1] = 1

shape_d = np.zeros((4, 4), dtype=int)
shape_d[2, 0:2] = 1

shape_e = np.zeros((4, 4), dtype=int)
shape_e[2, 1:2] = 1

shape_field = shape_a + shape_b + shape_c + shape_d + shape_e

########################
# SETUP OVERLAP FIELDS #
########################

in_ids = np.zeros((4, 4), dtype="O")
in_ids = in_ids + shape_a * 2 ** 0
in_ids = in_ids + shape_b * 2 ** 1
in_ids = in_ids + shape_c * 2 ** 2
in_ids = in_ids + shape_d * 2 ** 3
in_ids = in_ids + shape_e * 2 ** 4

out_ids = np.zeros((4, 4), dtype="O")
out_ids = out_ids + (1 - shape_a) * 2 ** 0
out_ids = out_ids + (1 - shape_b) * 2 ** 1
out_ids = out_ids + (1 - shape_c) * 2 ** 2
out_ids = out_ids + (1 - shape_d) * 2 ** 3
out_ids = out_ids + (1 - shape_e) * 2 ** 4

#####################################################
# GET OVERLAPS OUTSIDE AND INSIDE OF TARGET CONTOUR #
#####################################################

shape_in_ids = in_ids[shape_d == 1]
for num in shape_in_ids:
    print(f"{num}: " + "{0:b}".format(num)[::-1])

shape_out_ids = in_ids[shape_d == 0]
for num in shape_out_ids:
    print(f"{num}: " + "{0:b}".format(num)[::-1])

#####################
# FIND VALID SHAPES #
#####################

# valid shapes:
# - the shape is contained: partially inside or fully inside and not outside
# - the shape contains: fully inside and partially outside
print()

# partially inside tells me that at least a pixel of the shape is inside
partial_in = np.bitwise_or.reduce(shape_in_ids)
# fully inside tells me that all pixels of the shape are inside
fully_in = np.bitwise_and.reduce(shape_in_ids)
# partially outside tells me that a pixel of the shape is outside
partial_out = np.bitwise_or.reduce(shape_out_ids)

print(f"{partial_in}: " + "{0:b}".format(partial_in)[::-1])
print(f"{fully_in}: " + "{0:b}".format(fully_in)[::-1])
print(f"{partial_out}: " + "{0:b}".format(partial_out)[::-1])

valid1 = np.bitwise_and(np.bitwise_or(partial_in, fully_in), np.bitwise_not(partial_out))
print(f"{valid1}: " + "{0:b}".format(valid1)[::-1])

valid2 = np.bitwise_and(fully_in, partial_out)
print(f"{valid2}: " + "{0:b}".format(valid2)[::-1])

# invalid shapes:
# partially inside / partially outside
# fully outside and not inside

# only keep valid shapes
# from the initial in and out masks we filter out invalid shapes
# so now it is only > 0 in valid areas
valid_in_mask = np.bitwise_and(in_ids, np.bitwise_or(valid1, valid2))
valid_out_mask = np.bitwise_and(out_ids, np.bitwise_or(valid1, valid2))

########################################
# FIND DEPTHS IN IN/OUT OVERLAP FIELDS #
########################################

unique_vals_out = np.unique(valid_in_mask[shape_d == 1])
unique_vals_in = np.unique(valid_out_mask[shape_d == 1])

# we want to find a way to get, for each of these the number of on bits it has
print(unique_vals_out)
print(unique_vals_in)

int_uvout = np.array([int(obj) for obj in unique_vals_out]).reshape(-1,1)
int_uvin = np.array([int(obj) for obj in unique_vals_in])

print(np.unpackbits(int_uvout.astype(np.uint8), axis=1))
# print(np.unpackbits(int_uvin.astype(np.uint8), axis=1))
#print(np.unpackbits(unique_vals_out.astype(np.uint32).reshape(-1, 1), axis=1))

# Brian Kerninghan algorithm for counting set bits
# can use a vectorized numpy operation but need to figure out how
# to handle the bits given that we have arbitrary precission
def count_set_bits_large(num):
    count = 0
    while num != 0:
        num &= num - 1
        count += 1
    return count

for num in unique_vals_out:
    print(f"- {num}: " + "{0:b}".format(num)[::-1])
    print(count_set_bits_large(num))

for num in unique_vals_in:
    print(count_set_bits_large(num))

num_out = np.array([count_set_bits_large(num) for num in unique_vals_out]).min()
num_in = np.array([count_set_bits_large(num) for num in unique_vals_in]).max()

print(num_out)
print(num_in)


# print(shape_in_ids)
# print(all_in)
# print(all_in)
# print(f"{all_in}: " + "{0:b}".format(all_in)[::-1])
# print(f"{all_out}: " + "{0:b}".format(all_out)[::-1])
# print(f"{in_both}: " + "{0:b}".format(in_both)[::-1])

plt.imshow(shape_field)
plt.show()
#plt.imshow(in_ids.astype(int))


plt.imshow(valid_in_mask.astype(float))
plt.show()

plt.imshow(valid_out_mask.astype(float))
plt.show()