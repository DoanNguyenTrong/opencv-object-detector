import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


class Localizer:
    def __init__(self, original, transformed):
        # Locate points of the documents or object which you want to transform 
        # Original
        # xo = [334, 165, 931, 1117]
        # yo = [180, 591, 171, 571]
        # Transformed
        # xt = [0, 0, 600, 600]
        # yt = [0, 600, 0, 600]

        self.xo, self.yo = original
        self.xt, self.yt = transformed
        xy_o = np.float32([[x, y] for x, y in zip(self.xo, self.yo)]) 
        xy_t = np.float32([[x, y] for x, y in zip(self.xt, self.yt)])

        # Apply Perspective Transform Algorithm 
        print("Initiate perspective transformation matrix")
        self.M = cv2.getPerspectiveTransform(xy_o, xy_t)

        
        

    def perspective_transform(self, x, y):
        """
        - Wrap perspective transformation
        https://docs.opencv.org/master/da/d54/group__imgproc__transform.html#gaf73673a7e8e18ec6963e3774e6a94b87
        
        (x, y) -> (x', y')
        """
        x_ = (self.M[0,0] * x + self.M[0,1] * y + self.M[0,2])/ (self.M[2,0] * x + self.M[2,1] * y + self.M[2,2])
        y_ = (self.M[1,0] * x + self.M[1,1] * y + self.M[1,2])/ (self.M[2,0] * x + self.M[2,1] * y + self.M[2,2])
        return [int(x_), int(y_)]


    def raw_location(self, bboxes, offset=[0,0]):
        """
            Extract raw location for a group of classes
        """
        locs = []
        
        for box in bboxes:
            x_loc = box[0]+box[2]/2 + offset[0]
            y_loc = box[1]+box[3] + offset[1]
            locs.append([x_loc, y_loc])
        return locs

    def actual_location(self, IDs, scores, locs, cls = None):
        """
            Convert to actual location in the ground coordinate
        """
        loc_abs = []
        scores_ = []
        IDs_ = []
        
        for i in range(len(IDs)):
            x, y = locs[i]
            x_, y_ = self.perspective_transform(x,y)
            if x_ >=np.min(self.xt) and x_< np.max(self.xt) and y_ >=np.min(self.xt) and y_ < np.max(self.yt):
                loc_abs.append([x_, y_])
                scores_.append(scores[i])
                IDs_   .append(IDs[i])
        return [IDs, scores_, loc_abs]
        
    def draw_loc(self, img, x, y, radius=10, thickness=2, color=(255, 0, 0) ):
        """
        Draw a box a round some location
        """
        img = cv2.circle(img, (x,y), radius, color, thickness)
        return img

    def draw_locs(self, img, locs, radius=10, thickness=2, color=(255,0,0)):
        
        for x, y in locs:
            self.draw_loc(img, int(x), int(y), radius=radius, thickness=thickness, color=color)
        return img

