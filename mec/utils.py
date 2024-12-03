from itertools import product

def ranges(*args):
    """
    Generates the Cartesian product of ranges for the given dimensions.
    Each argument specifies the upper limit of a range, starting from 0.

    Args:
        *args: Integers specifying the upper bounds of ranges.

    Returns:
        An iterator over tuples representing the Cartesian product.
    """
    return product(*(range(dim) for dim in args))



# Example of use:
# for x, y, z in ranges(2, 3, 4):
#     print('x'+str(x)+', y'+str(y)+', z'+str(z))