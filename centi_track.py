import numpy as np
import cv2
from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import convolve
import csv
from itertools import repeat
import matplotlib.pyplot as plt
import json
from collections import deque
from pathlib import Path
import time

# 
full_file_path_to_video = "compressed/subB_t3_d4_labelled.mp4"
filename = Path(full_file_path_to_video).name

class centipede:
    ref_norm = [1, 0]
    def __init__(self, segments = 21, ant = 4):
        #we use 22 segemtns and 40 legs model
        self.tracking_started = False
        self.frame_count = 0
        self.antenna_body_tolerance = 0.05
        self.legs = (segments - 1) * 2
        self.ant = ant
        self.segments = segments
        self.contours = None
        self.body_length = None
        self.segment_points = []
        self.leg_points = []
        self.line_thickness = 1 # 6
        self.anchor_point = [0, 0]
        self.end_point = [0, 0]
        self.segment_angles = []
        self.leg_angles = []
        self.is_anchor_switched = False
        self.keep_segments_same = False
        self.csv_left_leg_angles = [list(range(int((self.legs - 2 + self.ant)/2)))]
        self.csv_right_leg_angles = [list(range(int((self.legs - 2 + self.ant)/2)))]
        self.temp_leg_angles = [list(range(self.legs - 2 + self.ant))]
        self.csv_segment_angles = [list(range(self.segments - 1))]
        self.csv_antennae_angles = ["left", "right"]
        self.head = None
        self.head_idx = None
        self.head_tracking_csv = []

    
    def update_centipede(self, process_frame, canvas_frame):
        self.frame_count += 1
        self.frame = process_frame
        self.analyze_frame(process_frame)
        self.draw_body(canvas_frame)
        self.draw_legs(canvas_frame)

    def find_head(self, min_vertex1, min_idx1, min_vertex2, min_idx2):
        comparison_head = global_head if self.head is None else self.head
        min_vertex1_dist = Calculations.calc_dist(min_vertex1, comparison_head)
        min_vertex2_dist = Calculations.calc_dist(min_vertex2, comparison_head)
        if min_vertex1_dist < min_vertex2_dist:
            self.head = min_vertex1
            self.head_idx = min_idx1
        else:
            self.head = min_vertex2
            self.head_idx = min_idx2
        
        self.head_tracking_csv.append(self.head) # track head every frame
            
    def analyze_frame(self, process_frame):
        self.process_midline(process_frame)
        self.generate_segment_points()
        self.get_segment_angles()
        self.extend_segment_points()
        self.remove_antennae()
        self.find_body_gap_dist()
        self.find_legs()
        self.separate_legs()
        self.order_legs()
        self.get_leg_angles()
        self.get_antennae_segments()
        self.separate_antennae()
        self.get_antennae_angles()
        # self.order_legs()
        # left_leg_angles, right_leg_angles = self.find_leg_angles()
        # self.csv_left_leg_angles.append(left_leg_angles) 
        # self.csv_right_leg_angles.append(right_leg_angles) 

        # # self.temp_leg_angles.append(leg_angles)
        # flip_var = -1 if self.track_head_idx == -1 else 1
        # oriented_seg_points = self.segment_points[::flip_var]
        # segment_angles = self.get_segment_angles(oriented_seg_points)
        # self.csv_segment_angles.append(segment_angles)

    def process_midline(self, process_frame):
        contours, _ = cv2.findContours(process_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        arc_lengths = np.array([cv2.arcLength(c, True) for c in contours])
        max_index = np.argmax(arc_lengths)
        filtered_contours = [c for i, c in enumerate(contours) if i != max_index]

        self.leg_contours = np.array(filtered_contours, dtype=object)
        self.midline_contour = contours[max_index]

        epsilon = 0.003 * cv2.arcLength(self.midline_contour, True)  # Approximation accuracy parameter
        midline_polygon = cv2.approxPolyDP(self.midline_contour, epsilon, True)
        self.body_length = cv2.arcLength(midline_polygon, True) / 2

        midline_polygon = midline_polygon[:, 0, :]
        self.midline_polygon = midline_polygon
        polygon_angles = []

        for i in range(len(midline_polygon)): 
            # Get the current vertex and its two neighboring vertices
            pt1 = midline_polygon[i - 1]  # Previous point
            pt2 = midline_polygon[i]      # Current point
            pt3 = midline_polygon[(i + 1) % len(midline_polygon)]  # Next point
            
            angle = Calculations.calc_angle_three_pts(pt1, pt2, pt3)
            polygon_angles.append(angle)
    
        two_min_indices = np.argpartition(polygon_angles, 2)[:2]
        min_vertex1, min_vertex2 = [midline_polygon[i] for i in two_min_indices]
        min_idx1, min_idx2 = two_min_indices

        self.find_head(min_vertex1, min_idx1, min_vertex2, min_idx2)

        return min_vertex1, min_idx1, min_vertex2, min_idx2

    def generate_segment_points(self):
        # If segment lengths should remain unchanged, exit early
        segment_points_list = [self.head]

        desired_segment_length = self.body_length / self.segments
        self.segment_length = desired_segment_length
    

        curr_idx_left = self.head_idx
        curr_idx_right = self.head_idx
        next_idx_right = curr_idx_right
        next_idx_left = curr_idx_left
        for i in range(self.segments):
            curr_point_right = segment_points_list[-1]
            segment_length_right = 0
            while segment_length_right < desired_segment_length:
                next_idx_right = (curr_idx_right + 1) % len(self.midline_polygon)
                next_point_right = self.midline_polygon[next_idx_right]

                travelled_dist_right = Calculations.calc_dist(curr_point_right, next_point_right)
                if segment_length_right + travelled_dist_right > desired_segment_length:
                    break

                segment_length_right += travelled_dist_right
                curr_point_right = next_point_right
                curr_idx_right = next_idx_right
            # Compute the interpolated segment point
            desired_distance_right = desired_segment_length - segment_length_right
            segment_point_right = Calculations.interpolate_points(desired_distance_right, curr_point_right, next_point_right)

            curr_point_left = segment_points_list[-1]
            segment_length_left = 0
            while segment_length_left < desired_segment_length:
                next_idx_left = (curr_idx_left + 1) % len(self.midline_polygon)
                next_point_left = self.midline_polygon[next_idx_left]

                travelled_dist_left = Calculations.calc_dist(curr_point_left, next_point_left)
                if segment_length_left + travelled_dist_left > desired_segment_length:
                    break

                segment_length_left += travelled_dist_left
                curr_point_left = next_point_left
                curr_idx_left = next_idx_left
            # Compute the interpolated segment point
            desired_distance_left = desired_segment_length - segment_length_left
            segment_point_left = Calculations.interpolate_points(desired_distance_left, curr_point_left, next_point_left)

            new_segment_point = Calculations.find_middle_point(segment_point_right, segment_point_left)
            new_segment_point_int = new_segment_point.astype(int)

            segment_points_list.append(new_segment_point_int)

        self.segment_points = np.array(segment_points_list)

    def get_segment_angles(self):
        segment_angles = []
        for i in range(len(self.segment_points) - 2):
            A = self.segment_points[i]
            B = self.segment_points[i + 1]
            C = self.segment_points[i + 2]
            angle = Calculations.calc_angle_three_pts(A, B, C)
            segment_angles.append(angle)
        self.csv_segment_angles.append(segment_angles)

    def extend_segment_points(self):
        extension_weight = 2

        extended_semgent_points = self.segment_points.tolist()
        head_vector = np.array(extended_semgent_points[0]) - np.array(extended_semgent_points[2])
        head_norm = head_vector / np.linalg.norm(head_vector)
        new_head = extension_weight * self.segment_length * head_norm + extended_semgent_points[0]
        new_head = new_head.astype(int)

        tail_vector = np.array(extended_semgent_points[-1]) - np.array(extended_semgent_points[-3])
        tail_norm = tail_vector / np.linalg.norm(tail_vector)
        new_tail = extension_weight * self.segment_length * tail_norm + extended_semgent_points[-1]
        new_tail = new_tail.astype(int)

        extended_semgent_points.insert(0, new_head.tolist())
        extended_semgent_points.append(new_tail.tolist())
        self.extended_segment_points = np.array(extended_semgent_points)

    def is_shoulder(self, point):
        # threshold for finding shoulders
        # could find distance of contour again instead of using body_gap_dist
        threshold = self.body_gap_dist * 1.25
        point = float(point[0]), float(point[1])
        dist = abs(cv2.pointPolygonTest(self.midline_contour, point, True))
        return dist < threshold

    def find_legs(self):
        branches_list = []
        leaves_list = []
        legs = []
        for contour in self.leg_contours:
            skeleton, tl_corner = self.extract_skeleton(contour)
            branches, leaves = self.find_branch_features(skeleton)


            if branches.any():
                branches = np.array([point[::-1] for point in branches])
                branches = branches + tl_corner
            if leaves.any():
                array_leaves = leaves # leaves that refer to array coords
                leaves = np.array([point[::-1] for point in leaves])
                leaves = leaves + tl_corner
            legs.extend(self.match_leg_points(branches, array_leaves, leaves, skeleton, tl_corner))
            

            branches_list.extend(branches)
            leaves_list.extend(leaves)
        self.branches = branches_list
        self.leaves = leaves_list

        self.leg_points = np.array(legs)

    def match_leg_points(self, branches, array_leaves, leaves, skeleton, tl_corner):
        #leg structure: [shoulder, foot]
        legs = []

        for i in range(len(leaves)):
            leaf = leaves[i]
            array_leaf = array_leaves[i]
            # find only vald shoulders
            if self.is_shoulder(leaf):
                foot = Calculations.traverse_binary_image(skeleton, array_leaf)
                foot = foot[::-1] + tl_corner
                legs.append(np.array([leaf, foot]))
        # two_leaves = leaves.shape[0] == 2
        # no_branches = branches.shape[0] == 0
        # if two_leaves and no_branches:
        #     #figure out foot, shoulder later
        #     #problem with 2 legs that look like arches
        #     legs.append(leaves)
        return legs

    def remove_antennae(self):
        head = self.extended_segment_points[0]
        tail = self.extended_segment_points[-1]
        
        head_dists = [Calculations.accurate_point_polygon(contour, head.tolist()) for contour in self.leg_contours]
        tail_dists = [Calculations.accurate_point_polygon(contour, tail.tolist()) for contour in self.leg_contours]

        min_head_dists_idx = np.argpartition(head_dists, 4)[:4]
        min_tail_dists_idx = np.argpartition(tail_dists, 4)[:4]

        calc_diagonal = lambda contour: np.sqrt(sum(cv2.boundingRect(contour)[2:]))


        min_head_lengths = [(calc_diagonal(self.leg_contours[i]), i) for i in min_head_dists_idx]
        min_tail_lengths = [(calc_diagonal(self.leg_contours[i]), i) for i in min_tail_dists_idx]

        min_head_lengths.sort(reverse=True, key=lambda x:x[0])
        min_tail_lengths.sort(reverse=True, key=lambda x:x[0])

        head_antennae_idx = [i for length, i in min_head_lengths[:2]]
        tail_antennae_idx = [i for length, i in min_tail_lengths[:2]]

        antennae_idx = head_antennae_idx + tail_antennae_idx
        antennae_idx = head_antennae_idx

        antennae_idx = min_head_dists_idx[:2]
        

        # min_head_dists_idx = np.argpartition(head_dists, 2)[:2]
        # min_tail_dists_idx = np.argpartition(tail_dists, 2)[:2]
        
        # antennae_idx = np.concatenate((min_head_dists_idx, min_tail_dists_idx), axis=0)
        # still incorrect antennae perception

        # access antennae contours
        self.antennae_contours = np.array(self.leg_contours)[antennae_idx]
        

        new_leg_contours = np.delete(self.leg_contours, antennae_idx, 0)

        self.leg_contours = new_leg_contours



    def extract_skeleton(self, contour):
        blank = np.zeros(self.frame.shape, dtype=np.uint8)
        cv2.drawContours(blank, [contour], -1, 255, thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(contour)
        skeleton_box = blank[y:y+h+1, x:x+w+1] # make box smaller so analysis is faster
        skeleton = skeletonize(skeleton_box > 0)  # Convert to boolean before skeletonization
        return skeleton, (x, y)
    
    def find_branch_features(self, skeleton):
        """Find branch endpoints and branch junctions in a skeletonized tree."""
        # Define a kernel to count neighbors
        kernel = np.array([[1, 1, 1], 
                        [1, 0, 1], 
                        [1, 1, 1]])  # Center pixel gets 10 to distinguish itself

        # Convolve skeleton with kernel to count neighbors
        neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)

        # Find branch points (more than 2 neighbors)
        branches = np.argwhere((skeleton > 0) & (neighbor_count >= 3))

        # Find branch ends (only 1 neighbor)
        leaves = np.argwhere((skeleton > 0) & (neighbor_count == 1))

        return branches, leaves
    
    def separate_legs(self):
        head = self.extended_segment_points[0]
        tail = self.extended_segment_points[-1]
        offset = Calculations.parallel_offset(head, tail, self.body_gap_dist)
        offset = np.array(offset)
        left_offset_segment_points = self.extended_segment_points + offset
        right_offset_segment_points = self.extended_segment_points - offset

        self.offset_left_segment_points = left_offset_segment_points
        self.offset_right_segment_points = right_offset_segment_points
        
        left_legs = []
        right_legs = []
        for leg in self.leg_points:
            shoulder, foot = leg
            left_seg, left_dist, _ = Calculations.closest_segment(left_offset_segment_points, shoulder)
            right_seg, right_dist, _ = Calculations.closest_segment(right_offset_segment_points, shoulder)
            if left_dist < right_dist:
                left_legs.append(leg)
            else:
                right_legs.append(leg)
        
        self.left_leg_points = np.array(left_legs)
        self.right_leg_points = np.array(right_legs)
        
    
    def order_legs(self):
        ordered_left_legs = []
        ordered_right_legs = []

        #antannae
        left_legs = self.left_leg_points
        right_legs = self.right_leg_points

        
        current_point = self.extended_segment_points[0]

        while len(left_legs) > 0:
            left_legs_dists = np.linalg.norm(left_legs[:, 0] - current_point, axis=1)
            min_index = np.argmin(left_legs_dists)
            next_point = left_legs[min_index]
            ordered_left_legs.append(next_point)
            left_legs = np.delete(left_legs, min_index, axis=0)
            current_point = next_point[0]
        
        current_point = self.extended_segment_points[0]

        while len(right_legs) > 0:
            right_legs_dists = np.linalg.norm(right_legs[:, 0] - current_point, axis=1)
            min_index = np.argmin(right_legs_dists)
            next_point = right_legs[min_index]
            ordered_right_legs.append(next_point)
            right_legs = np.delete(right_legs, min_index, axis=0)
            current_point = next_point[0]


        # adjust for antannae
        # short_left_legs = np.array(ordered_left_legs[:4])
        # left_legs_lens = np.linalg.norm(short_left_legs[:, 0] - short_left_legs[:, 1], axis=1)
        # max_len = np.argmin(left_legs_lens)

        # ordered_left_legs[0], ordered_left_legs[max_len] = ordered_left_legs[max_len], ordered_left_legs[0]

        # short_right_legs = np.array(ordered_right_legs[:4])
        # right_legs_lens = np.linalg.norm(short_right_legs[:, 0] - short_right_legs[:, 1], axis=1)
        # max_len = np.argmin(right_legs_lens)

        # ordered_right_legs[0], ordered_right_legs[max_len] = ordered_right_legs[max_len], ordered_right_legs[0]

        self.left_leg_points = np.array(ordered_left_legs)
        self.right_leg_points = np.array(ordered_right_legs)

    def get_leg_angles(self):
        left_leg_angles = []
        right_leg_angles = []
        for leg in self.left_leg_points:
            shoulder, foot = leg
            closest_segment, _, _ = Calculations.closest_segment(self.segment_points, shoulder)
            leg_angle = Calculations.angle_between_lines(leg, closest_segment)
            left_leg_angles.append(leg_angle)
        for leg in self.right_leg_points:
            shoulder, foot = leg
            closest_segment, _, _ = Calculations.closest_segment(self.segment_points, shoulder)
            leg_angle = Calculations.angle_between_lines(leg, closest_segment)
            right_leg_angles.append(leg_angle)
        self.csv_left_leg_angles.append(left_leg_angles)
        self.csv_right_leg_angles.append(right_leg_angles)
        # print(self.frame_count)


    def get_antennae_segments(self):
        # get antennae segments
        min_dist = float('inf')
        max_dist = 0
        antennae_segments = []
        for antennae_contour in self.antennae_contours:
            for point in antennae_contour:
                point = float(point[0][0]), float(point[0][1])
                dist = cv2.pointPolygonTest(self.midline_contour, point, True)
                if abs(dist) < min_dist:
                    min_point = point
                    min_dist = abs(dist)
                if abs(dist) > max_dist:
                    max_point = point
                    max_dist = abs(dist)
            antennae_segments.append([min_point, max_point])
        self.antennae_segments = np.array(antennae_segments)

    def separate_antennae(self):
        # sample 0th antennae, if it is right, then swap positions
        left_extended_head = self.offset_left_segment_points[0]
        right_extended_head = self.offset_right_segment_points[0]
        shoulder, foot = self.antennae_segments[0]
        left_dist = Calculations.calc_dist(shoulder, left_extended_head)
        right_dist = Calculations.calc_dist(shoulder, right_extended_head)
        if right_dist < left_dist:
            self.antennae_segments = self.antennae_segments[::-1]

    def get_antennae_angles(self):
        antennae_angles = []

        for leg in self.antennae_segments:
            shoulder, foot = leg
            closest_segment, _, _ = Calculations.closest_segment(self.segment_points, shoulder)
            leg_angle = Calculations.angle_between_lines(leg, closest_segment)
            antennae_angles.append(leg_angle)
        self.csv_antennae_angles.append(antennae_angles)

    def find_body_gap_dist(self):
        weight = 1.25
        min_distances = []
        # closest_midline_contour_pts = []
        for contour in self.leg_contours:
            min_dist = float('inf')
            min_point = None
            for point in contour:
                point = float(point[0][0]), float(point[0][1])
                dist = cv2.pointPolygonTest(self.midline_contour, point, True)
                min_dist = min(min_dist, abs(dist))
                # if abs(dist) < min_dist:
                #     min_point = point
            min_distances.append(min_dist)
            # closest_midline_contour_pts.append(min_point)
        body_gap_dist = int(np.median(min_distances) * weight)
        self.body_gap_dist = body_gap_dist
        # self.leg_contour_close_pts = closest_midline_contour_pts


    def draw_body(self, frame):
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        drawable_segment_points = [self.segment_points[:, np.newaxis, :]]

        cv2.polylines(frame, drawable_segment_points, isClosed=False, color=(0, 0, 255), thickness=3)
        head = self.segment_points[0]
        cv2.circle(frame, head, 3, (255, 0, 0), -1)
        for point in self.segment_points[1:]:
            cv2.circle(frame, point, 3, (0, 255, 0), -1)

    def draw_legs(self, frame):
        cv2.drawContours(frame, self.antennae_contours, -1, (255, 255, 0), 2)
        # for point in self.branches:
        #     cv2.circle(frame, point, 3, (0, 0, 255), -1)
        for point in self.leaves:
            cv2.circle(frame, point, 3, (0, 255, 0), -1)
        # for point in self.feet:
        #     cv2.circle(frame, point, 3, (255, 0, 0), -1)
        
        for p1, p2 in self.leg_points:
            cv2.line(frame, p1, p2, (0, 150, 150), 2)
        for count, (point1, point2) in enumerate(self.right_leg_points):
                cv2.putText(frame, f"{count}r", point2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for count, (point1, point2) in enumerate(self.left_leg_points):
            cv2.putText(frame, f"{count}l", point2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        

    

class Calculations:

    @staticmethod
    def calc_angle_three_pts(A, B, C):
        # Convert points to vectors
        BA = np.array(A) - np.array(B)
        BC = np.array(C) - np.array(B)

        # Compute dot and cross products
        dot_product = np.dot(BA, BC)
        cross_product = np.cross(BA, BC)
            

        # Compute angle in radians using arctan2
        angle_rad = np.arctan2(np.linalg.norm(cross_product), dot_product)

        # Convert to degrees
        angle_deg = np.degrees(angle_rad)

        return angle_deg
    

    @staticmethod
    def calc_dist(x, y):
        return np.linalg.norm(np.array(x) - np.array(y))

    @staticmethod
    def interpolate_points(distance, point1, point2):
        norm = Calculations.calc_dist(point1, point2)
        
        # Compute the interpolation ratio
        ratio = distance / norm
        
        # Compute the biases
        bias_x = ratio * (point2[0] - point1[0])
        bias_y = ratio * (point2[1] - point1[1])
        
        # Compute the new interpolated point
        new_point = np.array([point1[0] + bias_x, point1[1] + bias_y])
        
        return new_point
    
    @staticmethod
    def find_middle_point(point1, point2):
        new_point = (np.array(point1) + np.array(point2)) / 2
        return new_point
    
    @staticmethod
    def parallel_offset(point1, point2, distance):
        """
        Given a line segment (x1, y1) to (x2, y2), returns the (dx, dy) offset
        to shift the line parallel by the given distance.
        """

        # Compute the direction vector
        x1, y1 = point1
        x2, y2 = point2
        dx, dy = x2 - x1, y2 - y1

        # Compute the perpendicular vector (-dy, dx)
        perp_x, perp_y = -dy, dx

        # Normalize the perpendicular vector
        length = np.hypot(perp_x, perp_y)  # sqrt(perp_x^2 + perp_y^2)
        perp_x /= length
        perp_y /= length

        # Scale by the desired distance
        offset_x = perp_x * distance
        offset_y = perp_y * distance

        return offset_x, offset_y
    
    @staticmethod
    def point_to_segment_distance(P, A, B):
        """Compute the shortest distance from point P to the line segment AB."""
        P, A, B = np.array(P), np.array(A), np.array(B)
        AB = B - A
        AP = P - A
        t = np.dot(AP, AB) / np.dot(AB, AB)  # Projection scalar
        
        if t < 0:  # Closest to A
            closest_point = A
        elif t > 1:  # Closest to B
            closest_point = B
        else:  # Closest on the segment
            closest_point = A + t * AB
    
        return np.linalg.norm(P - closest_point), closest_point
    
    @staticmethod
    def closest_segment(polyline, P):
        """Find the closest segment in the polyline to point P."""
        min_dist = float('inf')
        closest_seg = None
        closest_point = None

        for i in range(len(polyline) - 1):
            A, B = polyline[i], polyline[i + 1]
            dist, pt = Calculations.point_to_segment_distance(P, A, B)
            if dist < min_dist:
                min_dist = dist
                closest_seg = (A, B)
                closest_point = pt

        return closest_seg, min_dist, closest_point
    
    @staticmethod
    def angle_between_lines(leg, segment):
        shoulder, foot = leg
        head_facing_segment, tail_facing_segment = segment
        leg_vector = np.array(foot) - np.array(shoulder)
        segment_vector = np.array(tail_facing_segment) - np.array(head_facing_segment)
        dot_product = np.dot(leg_vector, segment_vector)
        cross_product = np.cross(leg_vector, segment_vector)
        angle_rad = np.arctan2(np.linalg.norm(cross_product), dot_product)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    @staticmethod
    def accurate_point_polygon(contour, point):
        min_distance = float("inf")
        closest_point = None

        for pt in contour:
            dist = cv2.norm(np.array(pt) - np.array(point))  # Euclidean distance
            if dist < min_distance:
                min_distance = dist
                closest_point = tuple(pt)
        return min_distance
    
    @staticmethod
    def traverse_binary_image(image, start):
        #traverse graph to return foot
        #optimize by reducing size of visited set? 
        threshold = 140# 20 degrees in radians is the threshold for angle change
        visited = set()
        max_len = 15
        path = deque([start], maxlen=max_len)
        # past path angle
        fork_queue = []
        h, w = image.shape

        off_image_coord = lambda x: x[0] < 0 or x[1] < 0 or x[0] >= h or x[1] >= w

        neighbors = np.array([(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)])
        end = None

        while True:
            # todo: fix blue being too close to green
            prev = path[-1]
            visited.add(tuple(prev))
            path_vector = path[-1] - path[0]
            path_angle = np.arctan2(path_vector[1], path_vector[0])
            for offset in neighbors:
                neighbor_coords = offset + prev
                if tuple(neighbor_coords) in visited or off_image_coord(neighbor_coords) or image[tuple(neighbor_coords)] == 0:
                    continue
                #calculate angle 
                new_path_vector = neighbor_coords - path[0]
                new_path_angle = np.arctan2(new_path_vector[1], new_path_vector[0])
                new_path_offset = abs(new_path_angle - path_angle)
                fork_queue.append((neighbor_coords, new_path_offset))

            if len(fork_queue) == 0:
                end = path[-1]
                break
            next_point = min(fork_queue, key=lambda x: x[1])[0]

            #check if significant angle change
            if len(path) == max_len:
                mid_idx = max_len // 2
                path_angle = Calculations.calc_angle_three_pts(path[0], path[mid_idx], path[-1])
                if path_angle < threshold:
                    end = path[mid_idx]
                    break
            
            path.append(next_point)
            fork_queue = []
        return end
    



class Figures:
    @staticmethod
    def write_csv(data, data_type="leg_angles"):
        # data_array = np.array([row for row in data], dtype=object)
        output_dir = f"output_files/csvs/{file_title}_{data_type}.csv"
        frames_list = ["frames"] + list(range(1, len(data)))
        result_list = [[prefix] + list(row) for prefix, row in zip(frames_list, data)]
        with open(output_dir, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(result_list)

    @staticmethod
    def heatmapping_data(x_values, y_values, data_type="Leg_Angles"):
        #todo: fix heatmap
        output_dir = f"output_files/heatmaps/{file_title}_{data_type}_heatmap.png"
        max_val = len(x_values)
        y_values = [lst + list(repeat(0, max_val - len(lst))) for lst in y_values]
        y_values = list(zip(*y_values))
        # Create the heatmap
        plt.imshow(y_values, cmap='viridis', interpolation='nearest', aspect='auto', vmin=0, vmax=180)

        # Add a colorbar to show the scale
        plt.colorbar()

        # Add titles and labels
        plt.title(data_type)
        plt.xlabel("Frames")
        plt.ylabel("Position Index")

        plt.savefig(output_dir)
        # plt.rcdefaults()
        plt.show()

    @staticmethod
    def plotting_data(x_values, y_values, data_type="Leg_Angles"):
        output_dir = f"output_files/graphs/{file_title}_{data_type}_graph.png"
        y_values = list(zip(*y_values))
        fig, axes = plt.subplots(5, 5, figsize=(15, 12), sharey=True)
        for i, ax in enumerate(axes.flat):
            if i >= len(y_values):
                break
            x_values = list(range(len(y_values[i])))
            ax.plot(x_values, y_values[i])
            ax.set_title(f"{i}")
            # ax.set_xticks([])
            # ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(output_dir)


def process_frame(frame):
    #  Params: raw frame
    #  Creates a frame to show and a frame to analyze
    #  Returns a black

    height, width, _ = frame.shape
    min_dim = min(height, width)
    dim_thresh = 350
    scale_factor = 1
    if min_dim < dim_thresh:
        scale_factor = 2
    big_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(big_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mid_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # adaptive thresholding to use different threshold 
    # values on different regions of the frame.
    blur = ~blur

    ret, bw = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    bw = cv2.medianBlur(bw, 3)

    eroded = bw
    opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel, iterations=15)
    bw = opened

    height, width = bw.shape
    return bw, big_frame, (height, width)


def export_csv(centi):
    # export leg angles
    left_leg_angles = centi.csv_left_leg_angles
    right_leg_angles = centi.csv_right_leg_angles
    Figures.write_csv(left_leg_angles, "left_leg_angles")
    Figures.write_csv(right_leg_angles, "right_leg_angles")

    # export segment angles
    segment_angles = centi.csv_segment_angles
    Figures.write_csv(segment_angles, "segment_angles")

    # export head tracking
    head_tracking = centi.head_tracking_csv
    Figures.write_csv(head_tracking, "head_tracking")

    # export antennae angles
    antennae_angles = centi.csv_antennae_angles
    Figures.write_csv(antennae_angles, "antennae_angles")

def export_heatmap(centi):
    # export leg angles
    left_leg_angles = centi.csv_left_leg_angles
    right_leg_angles = centi.csv_right_leg_angles
    Figures.heatmapping_data(left_leg_angles, "left_leg_angles")
    Figures.heatmapping_data(right_leg_angles, "right_leg_angles")

    # export segment angles
    segment_angles = centi.csv_segment_angles
    Figures.heatmapping_data(segment_angles, "segment_angles")

    # export antennae angles
    antennae_angles = centi.csv_antennae_angles
    Figures.heatmapping_data(antennae_angles, "antennae_angles")


# filename = "polya_t3_d11_skel (2).avi"
# filename = "polya_t1_d11_skel.avi"
# filename = "subB_t3_d4_labelled.mp4"
file_path = f"processed_videos/{filename}"
# filename = "Screen Recording 2024-10-11 at 8.41.48 AM.mov"
# filename = "sub_F_t2_d3_skeleton.avi"
file_title = filename.split(".")[0]
file_title_raw = '_'.join(file_title.split("_")[:-1])

with open("head.json", 'r') as json_file:
    head_dict = json.load(json_file)
    global_head = head_dict[file_title_raw]

cap = cv2.VideoCapture(file_path)

ret, first_frame = cap.read()
processed_frame, big_frame, (height, width) = process_frame(first_frame)
centi = centipede()

if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)

video_name = f"output_files/videos/{file_title}_labelled.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, fps, (int(width), int(height))) 
start_time = time.time()

#around 44 contours
while (cap.isOpened()):

    ret, frame = cap.read()
    if ret == False:
        break
    processed_frame, canvas_frame, (height, width) = process_frame(frame)
    centi.update_centipede(processed_frame, canvas_frame)

    cv2.imshow('Frame', canvas_frame)
    cv2.imshow("Processed", processed_frame)

    video.write(canvas_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # cv2.waitKey(0)

end_time = time.time()
print(f"Processing time: {end_time - start_time} seconds")
# release the video capture object
cap.release()
video.release()

cv2.destroyAllWindows()

export_csv(centi)
# export_heatmap(centi)