# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 00:11:52 2021
The code was taken and modified from the following project on GitHub"
https://github.com/yitao-yu/PythonORBSlAM
"""
from open3d import *
import open3d as o3d
from ORB_slam_class import Frame
from obj_load import *
import numpy as np
import cv2
from copy import copy
import math

# inicializace
clicked = 0
first = 0
points_c = np.array([[200, 200],
                     [200, 400],
                     [400, 400],
                     [400, 200]])
points_co = np.array([[200, 200],
                      [200, 400],
                      [400, 400],
                      [400, 200]])


# transformace bodu
def transform(x, y, z, homography):
    p = np.array([x, y, z]).reshape(3, 1)
    new = np.dot(homography, p)
    return new


# reakce na kliknuti
def on_click(event, x, y, p1, p2):
    global clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = 1
    if event == cv2.EVENT_RBUTTONDOWN:
        clicked = 0
    pass


# vykresleni modelu
def show_model(img, obj, projection, color, a, b, c):
    vertices = obj.vertices
    scale_matrix = np.eye(3)
    h, w = 600, 600

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w/2, p[1] + h/2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)

        cv2.fillConvexPoly(img, imgpts, (a, b, c))
    return img


# vypocet projekcni matice pro zobrazovani objektu
def projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    print(projection)
    return np.dot(camera_parameters, projection)


# hlavni cast programu
def Visualize(cap, params):
    Visualize.cap = cap
    Visualize.params = params

    def update(vis):
        cap = Visualize.cap
        params = Visualize.params

        if cap.isOpened():
            ret, image = cap.read()
            if ret:
                frame = Frame(copy(image))
                frame = frame.execute(params)
                # Visualizing
                # Visualizing feature points
                points = frame.points4d

                if not (points is None):
                    if len(np.array(points).shape) != 2:
                        pass
                    else:
                        points = np.array(points)[:, :3]
                        pc = geometry.PointCloud()
                        pc.points = utility.Vector3dVector(points)
                        pc.paint_uniform_color([1.0, 0.0, 1.0])  # green
                        vis.add_geometry(pc)

                lastpose = frame.last_pose
                currpose = frame.curr_pose

                if lastpose is None:
                    lastpose = currpose

                botloc = np.array(currpose[:3, 3].T)
                botpc = geometry.PointCloud()
                botpc.points = utility.Vector3dVector(
                    np.array(botloc) + 0.2 * np.random.rand(3, 3))  # Covenience for visualization
                botpc.paint_uniform_color([1.0, 0, 1.0])  # blue
                vis.add_geometry(botpc)

                ctr = vis.get_view_control()
                camera = ctr.convert_to_pinhole_camera_parameters()
                camera.extrinsic = lastpose
                camera = ctr.convert_from_pinhole_camera_parameters(camera)  # adjusting camera pose

                # okno s nalezenymi body
                cv2.namedWindow("points")
                cv2.moveWindow("points", 1280, 18)
                cv2.imshow("points", frame.image)
                key = cv2.waitKey(1)

                global points_co
                img = copy(image)

                # prepocitavani bodu a jejich vykreslovani
                global first
                if first == 1:
                    points_fin = []
                    for i in range(4):
                        hom = frame.h
                        x = points_co[i, 0]
                        y = points_co[i, 1]
                        z = 1
                        x_wa = transform(x, y, z, hom)[0, 0]
                        y_wa = transform(x, y, z, hom)[1, 0]
                        z_wa = transform(x, y, z, hom)[2, 0]
                        points_fin.append([x_wa, y_wa, z_wa])
                    points_fin = np.array(points_fin)
                    points_2d = np.array([[points_fin[0, 0] / points_fin[0, 2], points_fin[0, 1] / points_fin[0, 2]],
                                          [points_fin[1, 0] / points_fin[1, 2], points_fin[1, 1] / points_fin[1, 2]],
                                          [points_fin[2, 0] / points_fin[2, 2], points_fin[2, 1] / points_fin[2, 2]],
                                          [points_fin[3, 0] / points_fin[3, 2], points_fin[3, 1] / points_fin[3, 2]]])
                    cv2.circle(image, (int(points_2d[0, 0]), int(points_2d[0, 1])), 3, (255, 0, 255), -1)
                    cv2.circle(image, (int(points_2d[1, 0]), int(points_2d[1, 1])), 3, (255, 0, 255), -1)
                    cv2.circle(image, (int(points_2d[2, 0]), int(points_2d[2, 1])), 3, (255, 0, 255), -1)
                    cv2.circle(image, (int(points_2d[3, 0]), int(points_2d[3, 1])), 3, (255, 0, 255), -1)
                    points_2d = np.array([[int(points_2d[0, 0]), int(points_2d[0, 1])],
                                          [int(points_2d[1, 0]), int(points_2d[1, 1])],
                                          [int(points_2d[2, 0]), int(points_2d[2, 1])],
                                          [int(points_2d[3, 0]), int(points_2d[3, 1])]])
                    global points_c
                    hom, mask = cv2.findHomography(points_c, points_2d)

                    # vyber modelu
                    global clicked
                    if clicked == 0:
                        obj = OBJ(os.path.join('Deer/Deer.obj'), swapyz=True)
                        a = 19
                        b = 39
                        c = 139
                    else:
                        obj = OBJ(os.path.join('Wolf/Wolf.obj'), swapyz=True)
                        a = 128
                        b = 128
                        c = 128

                    img = copy(image)
                    dst = points_2d
                    # img = cv2.polylines(img, [np.int32(dst)], True, (1, 0, 1), 1)
                    # trans = frame.trans[:3, :]
                    projection = projection_matrix(params['K'], hom)
                    # projection =
                    img = show_model(img, obj, projection, False, a, b, c)
                    points_co = points_2d
                else:
                    first = 1

                # vykresleni 3D modelu
                cv2.namedWindow("model")
                cv2.moveWindow("model", 640, 18)
                cv2.imshow('model', img)
                cv2.setMouseCallback("model", on_click)
                key = cv2.waitKey(1)

            else:
                ctr = vis.get_view_control()
                ctr.rotate(10.0, 0.0)
                cap.release()
                cv2.destroyAllWindows()

        return False

    # vykreslovani 3D mapy
    vis = o3d.visualization.Visualizer()
    vis.register_animation_callback(update)
    vis.create_window(width=params["width"], height=params["height"])

    vis.run()
    vis.destroy_window()
    cap.release()
    cv2.destroyAllWindows()
    pass


# main
if __name__ == "__main__":
    from cam_params import paramsdic

    print("Start")
    cap = cv2.VideoCapture(1)
    Visualize(cap, paramsdic)
