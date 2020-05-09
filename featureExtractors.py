import numpy as np
import util
from variables import *

class FeatureExtractor():

    def __init__(self, layout):
        self.layout = layout
        self.rows = self.layout.racetrack.width
        self.cols = self.layout.racetrack.height
        # self.divide_factor = (self.rows * self.cols)
        self.divide_factor = 100.

    def simpleExtractor(self, state):
        return np.array(state)

    def getCollisionFeatures(self, state):
        features = util.Counter()
        current_pos = state[:2]
        float_x, float_y = current_pos
        x, y = int(float_x), int(float_y)

        features["x"] = float_x # / self.divide_factor
        features["y"] = float_y # / self.divide_factor


        for i in range(self.rows):
            if self.layout.racetrack[x-i][y] == WALL_CELL:
                features["closest_left_wall"] = (float_x - x + i)  # / self.divide_factor
                break
        for i in range(self.rows):
            if self.layout.racetrack[x+i][y] == WALL_CELL:
                features["closest_right_wall"] = (x + i - float_x) # / self.divide_factor
                break
        for i in range(self.cols):
            if self.layout.racetrack[x][y+i] == WALL_CELL:
                features["closest_up_wall"] = (y + i - float_y) # / self.divide_factor
                break
        for i in range(self.cols):
            if self.layout.racetrack[x][y-i] == WALL_CELL:
                features["closest_down_wall"] = (float_y - y + i) # / self.divide_factor
                break

        features.divideAll(self.divide_factor)
        return np.array(features.values())
