import math
import colorsys

def color(c):
    return int(math.floor(c * 255))
def hsv2rgb(h, v):
    (r, g, b) = colorsys.hsv_to_rgb(h, 1.0, v)
    return (color(r), color(g), color(b))

def get_tuple_colors(num_color):
    return [hsv2rgb(color_id / num_color, 1.0)
            for color_id in range(num_color)]



WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_RED = (136, 0, 21)
RED = (237, 28, 36)
ORANGE = (255, 127, 39)
YELLOW = (255, 242, 0)
GREEN = (34, 177, 76)
TURQUOISE = (0, 162, 232)
INDIGO = (63, 72, 204)
PURPLE = (163, 73, 164)

BROWN = (185, 122, 87)
ROSE = (255, 174, 201)
GOLD = (255, 201, 14)
LIGHT_YELLOW = (239, 228, 176)
LIME = (181, 230, 29)
LIGHT_TURQUOISE = (153, 217, 234)
BLUE_GRAY = (112, 146, 190)
LAVENDER = (200, 191, 231)

BLUE = (0, 0, 255)

BBOX_COLORS = [DARK_RED, RED, ORANGE, YELLOW, GREEN, TURQUOISE, INDIGO, PURPLE,
               BROWN, ROSE, GOLD, LIGHT_YELLOW, LIME, LIGHT_TURQUOISE, BLUE_GRAY, LAVENDER]