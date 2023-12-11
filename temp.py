import json
import cv2
import numpy as np
import threading
import open3d as o3d
import re
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from os.path import isfile, exists, join, splitext
from os import listdir
from pytransform3d.transform_manager import TransformManager
from pytransform3d import transformations as pt
from pytransform3d.plot_utils import make_3d_axis
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from enum import IntEnum


import time

#import os
#os.environ['QT_QPA_PLATFORM'] = 'offscreen'

class FakeDetector:
    def __init__(self):
        self.setup()

    def setup(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.parameters =  cv2.aruco.DetectorParameters()
        #self.parameters.minDistanceToBorder =  1
        #self.parameters.adaptiveThreshWinSizeMin = 3
        #self.parameters.adaptiveThreshWinSizeMax = 15
        #self.parameters.adaptiveThreshWinSizeStep = 3
        #self.parameters.minMarkerPerimeterRate = 0.01
        #self.parameters.maxMarkerPerimeterRate = 8.0
        #self.parameters.minCornerDistanceRate = 0.01
        #self.parameters.minMarkerDistanceRate  = 0.01
        #self.parameters.maxErroneousBitsInBorderRate = 0.65
        #self.parameters.cornerRefinementMaxIterations = 1500
        #self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.useAruco3Detection = True

        self.printedsizeofmarker = 0.039
        
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

        calibration_matrix_path = "./data/calibration_matrix.npy"
        distortion_coefficients_path = "./data/distortion_coefficients.npy"
    
        self.k = np.load(calibration_matrix_path)
        self.d = np.load(distortion_coefficients_path)

    def pose_estimation(self,frame, aruco_dict_type=None, matrix_coefficients=None, distortion_coefficients=None) ->	np.ndarray:

        """
        pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients)

        pose estimation of Aruco markers

        Parameters
        ----------
        frame : np.ndarray
            input image.
        aruco_dict_type : np.ndarray
            dictionary of markers indicating the type of markers.
        matrix_coefficients : np.ndarray
            intrinsic matrix of the calibrated camera.
        distortion_coefficients : np.ndarray
            distortion coefficients associated with the camera.

        Returns
        -------
        np.ndarray
            input image with the axis of the detected Aruco markers drawn on it
        dictionary
            output detections stuff (namely the R matrix and t vector from aruco pose)
        """
        myDict = {}

        if aruco_dict_type is None:
            aruco_dict_type = cv2.aruco.DICT_5X5_100
        if matrix_coefficients is None:
            matrix_coefficients = self.k
        if distortion_coefficients is None:
            distortion_coefficients = self.d

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = frame.copy()
        corners, ids, rejected_img_points = self.detector.detectMarkers(gray)

        # If markers are detected
        myDict = dict()
        if len(corners) > 0:
            ids = ids.flatten()
            #print('here')
            #print(ids)
            for (markerCorner, markerID) in zip(corners, ids):
            #for i in range(0, len(ids)):
                #print('////')
                #print(markerID)
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorner, self.printedsizeofmarker, matrix_coefficients,
                                                                            distortion_coefficients)
                
                localkey = markerID

                #if markerID==13:
                #    dfghjk
                #print('----')
                #print(localkey)
                #print('----')           
                myDict[str(markerID)]=(rvec, tvec)
                
                #print(cv2.Rodrigues(rvec))
                # Draw a square around the markers
                #cv2.aruco.drawDetectedMarkers(frame, corners) 

                # Draw Axis
                #cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  
                frame = cv2.drawFrameAxes( frame, matrix_coefficients, distortion_coefficients, rvec, tvec, length=0.01 )

        #print(myDict)
        return frame, myDict

    def get_detector(self):
        return self

    def showdepth(self, input_depth_img):
        cimg = input_depth_img.copy()
        cimg = cimg / (2.0**12.0)
        cimg = 255.0 * cimg
        cimg = cimg.astype(np.uint8)
        cimg = cv2.applyColorMap(cimg, cv2.COLORMAP_JET)

        #cv2.imshow("depth", cimg)

    def showcolor(self, input_color_img):
       
        #img_out = input_color_img.copy()
        output, dictout = self.pose_estimation(input_color_img)#, cv2.aruco.DICT_4X4_50, self.detector.k, self.detector.d)
        #cv2.imshow("color", output)

        return dictout

class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5

class Camera:
    def __init__(self):
        

        self.pipeline = rs.pipeline()
        config = rs.config()
        color_profiles, depth_profiles = self.get_profiles()

        w, h, fps, fmt = depth_profiles[4] #4
        config.enable_stream(rs.stream.depth, w, h, fmt, fps)

        w, h, fps, fmt = color_profiles[28] #28
        config.enable_stream(rs.stream.color, w, h, fmt, fps)

        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()

        depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

        depth_scale = depth_sensor.get_depth_scale()
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_profiles(self):
        ctx = rs.context()
        devices = ctx.query_devices()

        color_profiles = []
        depth_profiles = []
        for device in devices:
            name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            #print('Sensor: {}, {}'.format(name, serial))
            #print('Supported video formats:')
            for sensor in device.query_sensors():
                for stream_profile in sensor.get_stream_profiles():
                    stream_type = str(stream_profile.stream_type())

                    if stream_type in ['stream.color', 'stream.depth']:
                        v_profile = stream_profile.as_video_stream_profile()
                        fmt = stream_profile.format()
                        w, h = v_profile.width(), v_profile.height()
                        fps = v_profile.fps()

                        video_type = stream_type.split('.')[-1]
                        #print('  {}: width={}, height={}, fps={}, fmt={}'.format(
                        #    video_type, w, h, fps, fmt))
                        if video_type == 'color':
                            color_profiles.append((w, h, fps, fmt))
                        else:
                            depth_profiles.append((w, h, fps, fmt))

        return color_profiles, depth_profiles

    def getimgs(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        profile_local = frames.get_profile()
        intrinsics = profile_local.as_video_stream_profile().get_intrinsics()
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
  

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        # Get color
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        # Get depth
        depth_frame = aligned_frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image



    def getimg(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get color
        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        return color_image

class FakeCamera:
    def __init__(self):
        path_to_config = "./data/readrosbag.json"
        with open(path_to_config) as json_file:
            config = json.load(json_file)
            self.initialize_config(config)
            self.check_folder_structure(config['path_dataset'])

        config['debug_mode'] = False
        config['device'] = 'cpu:0'

        self.ii = 0

        self.config = config

        [self.color_files, self.depth_files] = self.get_rgbd_file_lists(self.config["path_dataset"])
        self.n_files = len(self.color_files)
        
        #temp_det = FakeDetector()
        #self.detector = temp_det.get_detector()

        

    def sorted_alphanum(self, file_list_ordered):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(file_list_ordered, key=alphanum_key)

    def add_if_exists(self, path_dataset, folder_names):
        for folder_name in folder_names:
            if exists(join(path_dataset, folder_name)):
                path = join(path_dataset, folder_name)
                return path
        raise FileNotFoundError(
            f"None of the folders {folder_names} found in {path_dataset}")

    def get_rgbd_folders(self, path_dataset):
        path_color = self.add_if_exists(path_dataset, ["image/", "rgb/", "color/"])
        path_depth = join(path_dataset, "depth/")
        return path_color, path_depth

    def get_rgbd_file_lists(self, path_dataset):
        path_color, path_depth = self.get_rgbd_folders(path_dataset)
        color_files = self.get_file_list(path_color, ".jpg") + \
                self.get_file_list(path_color, ".png")
        depth_files = self.get_file_list(path_depth, ".png")
        return color_files, depth_files

    def get_file_list(self, path, extension=None):
        if extension is None:
            file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
        else:
            file_list = [
                path + f
                for f in listdir(path)
                if isfile(join(path, f)) and splitext(f)[1] == extension
            ]
        file_list = self.sorted_alphanum(file_list)
        return file_list

    def check_folder_structure(self, path_dataset):
        if isfile(path_dataset) and path_dataset.endswith(".bag"):
            return
        path_color, path_depth = self.get_rgbd_folders(path_dataset)
        assert exists(path_depth), \
                "Path %s is not exist!" % path_depth
        assert exists(path_color), \
                "Path %s is not exist!" % path_color

    def set_default_value(self, config, key, value):
        if key not in config:
            config[key] = value

    def initialize_config(self, config):

        # set default parameters if not specified
        self.set_default_value(config, "depth_map_type", "redwood")
        self.set_default_value(config, "n_frames_per_fragment", 100)
        self.set_default_value(config, "n_keyframes_per_n_frame", 5)
        self.set_default_value(config, "depth_min", 0.3)
        self.set_default_value(config, "depth_max", 3.0)
        self.set_default_value(config, "voxel_size", 0.05)
        self.set_default_value(config, "depth_diff_max", 0.07)
        self.set_default_value(config, "depth_scale", 1000)
        self.set_default_value(config, "preference_loop_closure_odometry", 0.1)
        self.set_default_value(config, "preference_loop_closure_registration", 5.0)
        self.set_default_value(config, "tsdf_cubic_size", 3.0)
        self.set_default_value(config, "icp_method", "color")
        self.set_default_value(config, "global_registration", "ransac")
        self.set_default_value(config, "python_multi_threading", True)

        # `slac` and `slac_integrate` related parameters.
        # `voxel_size` and `depth_min` parameters from previous section,
        # are also used in `slac` and `slac_integrate`.
        self.set_default_value(config, "max_iterations", 5)
        self.set_default_value(config, "sdf_trunc", 0.04)
        self.set_default_value(config, "block_count", 40000)
        self.set_default_value(config, "distance_threshold", 0.07)
        self.set_default_value(config, "fitness_threshold", 0.3)
        self.set_default_value(config, "regularizer_weight", 1)
        self.set_default_value(config, "method", "slac")
        self.set_default_value(config, "device", "CPU:0")
        self.set_default_value(config, "save_output_as", "pointcloud")
        self.set_default_value(config, "folder_slac", "slac/")
        self.set_default_value(config, "template_optimized_posegraph_slac",
                        "optimized_posegraph_slac.json")

        # path related parameters.
        self.set_default_value(config, "folder_fragment", "fragments/")
        self.set_default_value(config, "subfolder_slac",
                        "slac/%0.3f/" % config["voxel_size"])
        self.set_default_value(config, "template_fragment_posegraph",
                        "fragments/fragment_%03d.json")
        self.set_default_value(config, "template_fragment_posegraph_optimized",
                        "fragments/fragment_optimized_%03d.json")
        self.set_default_value(config, "template_fragment_pointcloud",
                        "fragments/fragment_%03d.ply")
        self.set_default_value(config, "folder_scene", "scene/")
        self.set_default_value(config, "template_global_posegraph",
                        "scene/global_registration.json")
        self.set_default_value(config, "template_global_posegraph_optimized",
                        "scene/global_registration_optimized.json")
        self.set_default_value(config, "template_refined_posegraph",
                        "scene/refined_registration.json")
        self.set_default_value(config, "template_refined_posegraph_optimized",
                        "scene/refined_registration_optimized.json")
        self.set_default_value(config, "template_global_mesh", "scene/integrated.ply")
        self.set_default_value(config, "template_global_traj", "scene/trajectory.log")

        if config["path_dataset"].endswith(".bag"):
            assert os.path.isfile(config["path_dataset"]), (
                f"File {config['path_dataset']} not found.")
            print("Extracting frames from RGBD video file")
            config["path_dataset"], config["path_intrinsic"], config[
                "depth_scale"] = extract_rgbd_frames(config["path_dataset"])

    def read_image(self, c_path, c_type):

        if c_type == 0:
            img = cv2.imread(c_path)
        elif c_type == 1:
            img = o3d.io.read_image(c_path)
        else:
            sys.exit("trouble")

        if img is None:
            sys.exit("trouble")
        
        if c_type == 0:
            imageprobe = img.copy()
            b,g,r = cv2.split(imageprobe) 
            image = cv2.merge([r, g, b]) 
        elif c_type == 1:
            image = np.asarray(img).copy()
        else:
            sys.exit("trouble")
        
        #print(image.dtype)
        return image

    def getimgs(self):
        self.color_img = self.read_image(self.color_files[self.ii], 0)
        self.depth_img = self.read_image(self.depth_files[self.ii], 1)

        walla = self.color_img.copy()
        wallo = self.depth_img.copy()

        proto_probe = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        #print(proto_probe.intrinsic_matrix)
        #print('>>>>>>>>>>>>>>>>>>>')
        #asdasdasd

        #self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(proto_probe.width, proto_probe.height, proto_probe.fx, proto_probe.fy, proto_probe.cx, proto_probe.cy)
  
        #self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(proto_probe.width, proto_probe.height, roto_probe.intrinsic_matrix)

        self.pinhole_camera_intrinsic = proto_probe

        #self.showdepth()
        #self.showcolor()
        #k = cv2.waitKey(30)

        self.ii += 1
        if (self.ii == self.n_files):
            self.ii = 0

        return walla, wallo

class FakeBoard:
    def __init__(self):
        self.setup()

    def setup(self):
        self.group_a = {
            '1':0,
            '2':1,
            '3':2,
            '4':3,
            '5':4,
            '6':5,
            '8':6,
            '9':7,
            '10':8,
            '11':9,
            '12':10,
            '13':11
        }
        self.group_b = {
            '14':0,
            '15':1,
            '16':2,
            '17':3,
            '18':4,
            '19':5,
            '21':6,
            '22':7,
            '23':8,
            '24':9,
            '25':10,
            '26':11
        }
        self.tm = TransformManager()
        self.tm = self.grid_transforms(self.tm)

    def grid_transforms(self, in_stuff):
        AA = 0.0196
        aaa = [     2,    3,     4,    5,    6,    8,    9,   10,   11,   12,   13]
        bbb = [  4*AA, 8*AA,  2*AA, 6*AA,    0, 8*AA, 2*AA, 6*AA,    0, 4*AA, 8*AA]
        ccc = [     0,    0,  3*AA, 3*AA, 6*AA, 6*AA, 9*AA, 9*AA,12*AA,12*AA,12*AA]

        rr_m_temp = np.identity(3)

        for zzz, yyy, xxx in  zip(aaa, bbb, ccc):
            #print(yyy, xxx)

            tt_v_temp = np.array([-yyy, xxx, 0])

            xx_to_cam = pt.transform_from(rr_m_temp,tt_v_temp)
            in_stuff.add_transform(str(zzz), str(1), xx_to_cam)


        #print('fdghjkl')
        in_stuff.check_consistency()
        #print(in_stuff.check_consistency())
        return in_stuff
        

    def do_stuff(self, in_stuff):

        #start = time.time()
            
        

        f=14
        button = False
        print(*in_stuff.keys())

        #in_stuff = self.grid_transforms(in_stuff)

        
        #tm.add_transform("marker x", "cam", marker2cam)
        for ckey in in_stuff.keys():
            if not button:
                button = True
            aa, bb = in_stuff[ckey]
            rr_v_to_consider = aa[0][0]
            tt_v_to_consider = bb[0][0]

            #start = time.time()
            #print("hello")
            rr_m_to_consider, pp = cv2.Rodrigues(aa[0][0])
            #end = time.time()
            #print(end - start)

            #print('---')
            #print(rr_v_to_consider)            
            #print('---')
            #print(tt_v_to_consider)
            #print('---')
            #print(rr_m_to_consider)
            #print('---')
            #print(pp)
            #print('---')
            #print(ckey)
            #print('---')
            xx_to_cam = pt.transform_from(rr_m_to_consider,tt_v_to_consider)
            #self.tm.add_transform(ckey, "cam", xx_to_cam)
            self.tm.add_transform("cam", ckey, xx_to_cam)
            #self.tm.check_consistency()
        
        
        #if button:
            #plt.figure(figsize=(10, 5))

            #self.ax = make_3d_axis(2, 121)
            #self.ax = self.tm.plot_frames_in("cam", ax=self.ax, alpha=0.6)
            #self.ax.view_init(30, 20)

            #self.ax = make_3d_axis(3, 122)
            #self.ax = self.tm.plot_frames_in("A", ax=ax, alpha=0.6)
            #self.ax.view_init(30, 20)

            #plt.show()
        
        #print(self.tm.check_consistency())
        #self.tm.check_consistency()
        #self.tm.write_png('./graph.png')
        #end = time.time()
        #print(end - start)
        return self.tm, button




       
        

class Experiment:
    def __init__(self):
        self.setup()
        self.viopt = False
        self.temp_h = {}
        self.taskm = []
        self.ref2cam = np.eye(4)

    def setup(self):        
        self.c_camera = FakeCamera()
        #self.c_camera = Camera()
        self.c_detector = FakeDetector() 
        self.c_board = FakeBoard() 

    def run(self):    
        
        gui.Application.instance.initialize()

        self.window = gui.Application.instance.create_window('3d', width=640, height=480)
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.widget.scene.show_ground_plane(True, rendering.Scene.GroundPlane.XY)
        self.window.add_child(self.widget)
        self.mat = rendering.MaterialRecord()
        self.mat.shader = 'defaultLitTransparency'
        self.mat.line_width = 5.0
                         #'defaultUnlit'
                         #'defaultLit'
                         #'unlitLine'
                         #'defaultLitTransparency'

        #mat = rendering.MaterialRecord()
        #mat.shader = "unlitLine"
        #mat.line_width = 5.0

        self.mat.point_size = 9.0

        #self.thread_main()
        threading.Thread(target=self.thread_main).start()
        gui.Application.instance.run()

    def update_geometry(self):
        print('pkk')
        #print(time.ctime())
        self.widget.scene.clear_geometry()




        if True:
            hc_width = 640
            hc_height = 480
            hc_intrinsic_matrix = np.array( [[612.602783203125, 0, 0],
                                [0, 612.6766967773438, 0],
                                [327.29217529296875, 239.1042938232422, 1]])

            hc_extrinsic_matrix = np.array( [[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])

        
        
        
        #standardCameraParametersObj  = self.widget.scene.get_view_control().convert_to_pinhole_camera_parameters()
        # cameraLines = open3d.geometry.LineSet.create_camera_visualization(intrinsic=standardCameraParametersObj.intrinsic, extrinsic=standardCameraParametersObj.extrinsic)
        #cameraLines = open3d.geometry.LineSet.create_camera_visualization(view_width_px=WIDTH, view_height_px=HEIGHT, intrinsic=standardCameraParametersObj.intrinsic.intrinsic_matrix, extrinsic=standardCameraParametersObj.extrinsic)
        if True:
            #frustum = o3d.geometry.LineSet.create_camera_visualization(
            #        view_width_px=hc_width,#color.columns, 
            #        view_height_px=hc_height,#color.rows, 
            #        intrinsic=standardCameraParametersObj.intrinsic.intrinsic_matrix, #,#hc_intrinsic_matrix,#intrinsic.numpy(),
            #        extrinsic=standardCameraParametersObj.extrinsic),#np.linalg.inv(hc_extrinsic_matrix),#np.linalg.inv(T_frame_to_model.cpu().numpy()), 
                    #1.0)
            print("----")
            print(self.c_camera.pinhole_camera_intrinsic)
            print("----")
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                    hc_width, 
                    hc_height, 
                    self.c_camera.pinhole_camera_intrinsic.intrinsic_matrix, #hc_intrinsic_matrix, #,#hc_intrinsic_matrix,#intrinsic.numpy(),
                    hc_extrinsic_matrix,#np.linalg.inv(hc_extrinsic_matrix),#np.linalg.inv(T_frame_to_model.cpu().numpy()), 
                    0.3)
            #frustum.paint_uniform_color([0.0, 1.0, 0.000])
            frustum = frustum.transform(self.ref2cam)
            matf = rendering.MaterialRecord()
            matf.shader = "unlitLine"
            matf.line_width = 5.0
            self.widget.scene.add_geometry("cam", frustum, matf)
            #self.widget.scene.add_geometry(frustum)

        #self.temp_h
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(list(self.temp_h.values()))
        #pcd.points = o3d.utility.Vector3dVector(list([[0,1,2],[3,4,5],[6,7,8]]))
        #f=14
        self.widget.scene.add_geometry('vipo', pcd, self.mat)




        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.03)        
        #toviewpoint = self.taskm.get_transform("cam", "1")
        frame = frame.transform(self.ref2cam)
        self.widget.scene.add_geometry("frame", frame, self.mat)



        
        #self.widget.scene.add_geometry("cam", frustum, self.mat)
        #self.widget.scene.add_geometry(frustum)

    def side_probe(self, in_transformation_manager):
        probe_dict = in_transformation_manager.to_dict()
        #probe_len = in_transformation_manager.connected_components()
        #probe_mtx = np.zeros((probe_len, 3))
        probe_point = pt.vector_to_point([0, 0, 0])
        probe_d = {}

        for ii, jj in enumerate(probe_dict['nodes']):
            #print(ii)
            probe_2_ref = in_transformation_manager.get_transform(jj, "1")
            probe_tip = pt.transform(probe_2_ref, pt.vector_to_point([probe_point[0],probe_point[1],probe_point[2]]))[:-1]
            probe_d[jj] = probe_tip
            #print(probe_tip)
        
        return probe_d

    #def getcamdelta(self):
    #    toviewpoint = self.taskm.get_transform("cam", "1")
    #    return toviewpoint

    def thread_main(self):
        

        if self.viopt:
            plt.ion()
            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(121,projection='3d')
            ax2 = fig.add_subplot(122,projection='3d')

        #ax = make_3d_axis(2, 121)
        
        #ax.view_init(30, 20)

        #ax = make_3d_axis(3, 122)
        
        #ax.view_init(30, 20)

        

        while True:
            start = time.time()
            current_color, current_depth = self.c_camera.getimgs()
            #self.c_detector.showdepth(current_depth)
            current_det_dict = self.c_detector.showcolor(current_color)
            herehere, therethere = self.c_board.do_stuff(current_det_dict)

           
            if therethere:
                self.taskm = herehere
                self.ref2cam = self.taskm.get_transform("cam", "1")

                #ax = fig.add_subplot(projection='3d')
                if self.viopt:
                    ax1 = fig.add_subplot(121,projection='3d')
                    ax2 = fig.add_subplot(122,projection='3d')

                
                #ax = make_3d_axis(2, 121)
                #ax.view_init(30, 20)

                for itr in current_det_dict.keys():
                    #print(itr)
                    #ee2object = herehere.get_transform(str(itr), "cam")
                    #print('sss')
                    #print(itr)
                    #print((itr == '1'))

                    #print('/////-----------------------------------------------')
                    #print(herehere.connected_components())
                    #print(herehere.to_dict()['nodes'])
                    self.temp_h = self.side_probe(herehere)
                    #print('-----------------------------------------------/////')

                    #ax2 = herehere.plot_frames_in("1", ax=ax2, alpha=0.6, s=0.01)

                    if self.viopt:
                        ax2 = herehere.plot_connections_in("1", ax=ax2, ax_s=0.01)
                        
                        ax2.view_init(elev=90, azim=-90)
                        ax2.set_xlim((-0.1, 0.2))
                        ax2.set_ylim((-0.1, 0.4))
                        ax2.set_zlim((0, 0.4))

                        #ax1 = herehere.plot_frames_in("1", ax=ax1, alpha=0.6, s=0.01)
                        ax1 = herehere.plot_connections_in("1", ax=ax1, ax_s=0.01)

                        ax1.view_init(elev=0., azim=-90)
                        ax1.set_xlim((-0.1, 0.2))
                        ax1.set_ylim((-0.1, 0.4))
                        ax1.set_zlim((0, 0.4))
                    

                    '''
                    if not (itr == '1'):
                        a='a'
                        #ee2object = herehere.get_transform(str(itr), "1")
                        #ee2object = herehere.get_transform("cam", str(itr))
                        #origin_of_A_pos = pt.vector_to_point([0, 0, 0])
                        #kcdo = pt.transform(ee2object, origin_of_A_pos)[:-1]

                        kcdo = pt.vector_to_point([0, 0, 0])



                        object2ref = herehere.get_transform(str(itr), "1")
                        #object2ref = herehere.get_transform("1", str(itr))
                        kcd = pt.transform(object2ref, pt.vector_to_point([kcdo[0],kcdo[1],kcdo[2]]))[:-1]

                        #print(origin_of_A_in_B_xyz)
                        #plt.scatter(kcd[0],kcd[1],kcd[2])
                        ax.scatter(kcd[0],kcd[1], kcd[2], marker='^')
                        #plt.scatter(kcd[0],kcd[1], marker='x')
                        #plt.axis('equal')
                        #plt.xlim(-0.02, 0.05)
                        #plt.ylim(-0.3, 0.1)
                        #ax.set_xlim((-0.5, 0.5))
                        #ax.set_ylim((-0.5, 0.5))
                        #ax.set_zlim((0.0, .3))
                    else:
                        a='a'
                        ee2object = herehere.get_transform("cam", "1")
                        origin_of_A_pos = pt.vector_to_point([0, 0, 0])
                        pkk = pt.transform(ee2object, origin_of_A_pos)[:-1]
                        #plt.scatter(pkk[0],pkk[1],pkk[2])
                        print((pkk[0],pkk[1],pkk[2]))

                        AA = 0.0196
                        BB = 1
                        aaa = [     2,    3,     4,    5,    6,    8,    9,   10,   11,   12,   13]
                        bbb = [  4*AA, 8*AA,  2*AA, 6*AA,    0, 8*AA, 2*AA, 6*AA,    0, 4*AA, 8*AA]
                        ccc = [     0,    0,  3*AA, 3*AA, 6*AA, 6*AA, 9*AA, 9*AA,12*AA,12*AA,12*AA]
                        ddd = [     0,    0,     0,    0,    0,    0,    0,    0,    0,    0,    0]
                        eee = [    BB,   BB,    BB,   BB,   BB,   BB,   BB,   BB,   BB,   BB,   BB]

                        for pp,oo,ii,uu in zip(bbb, ccc, ddd, eee):
                            print(';;;;;')
                            print(pp,oo,ii,uu)
                            ax.scatter(pp, oo, ii, marker='o')                        
                        #plt.scatter(np.array(ccc).T,np.array(bbb).T,np.array(ddd).T)
                        #plt.scatter(pkk[0],-pkk[1], marker='*')
                        #plt.axis('equal')
                        #print(origin_of_A_in_B_xyz)
                        #ax.scatter(origin_of_A_in_B_xyz[0],origin_of_A_in_B_xyz[1],origin_of_A_in_B_xyz[2])
                    '''

                #ax = herehere.plot_frames_in("cam", ax=ax, alpha=0.6, s=10)
                
            #print(current_det_dict)
                
            #kk = cv2.waitKey(30)
                #plt.ion()
                #plt.show()
                if self.viopt:
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.1)
                    plt.clf()
            #print('farc')
            gui.Application.instance.post_to_main_thread(self.window, self.update_geometry)
            end = time.time()
            print(1/(end - start))
            print((end - start))

if __name__ == "__main__":
    probetrial = Experiment()
    probetrial.run()
