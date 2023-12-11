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
       
        img_out = input_color_img.copy()
        output, dictout = self.pose_estimation(img_out)#, cv2.aruco.DICT_4X4_50, self.detector.k, self.detector.d)
        #cv2.imshow("color", output)

        return dictout

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
            rr_m_to_consider, pp = cv2.Rodrigues(aa[0][0])

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
            self.tm.check_consistency()
        
        
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
        self.tm.check_consistency()
        self.tm.write_png('./graph.png')
        return self.tm, button




       
        

class Experiment:
    def __init__(self):
        self.setup()

    def setup(self):        
        self.c_camera = FakeCamera()
        self.c_detector = FakeDetector() 
        self.c_board = FakeBoard() 

    def run(self):    
        
        #gui.Application.instance.initialize()

        #self.window = gui.Application.instance.create_window('3d', width=640, height=480)
        #self.widget = gui.SceneWidget()
        #self.widget.scene = rendering.Open3DScene(self.window.renderer)
        #self.window.add_child(self.widget)

        self.thread_main()
        threading.Thread(target=self.thread_main).start()
        #gui.Application.instance.run()

    def update_geometry(self):
        f=14

    def thread_main(self):
        plt.ion()
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121,projection='3d')
        ax2 = fig.add_subplot(122,projection='3d')

        #ax = make_3d_axis(2, 121)
        
        #ax.view_init(30, 20)

        #ax = make_3d_axis(3, 122)
        
        #ax.view_init(30, 20)

        

        while True:
            current_color, current_depth = self.c_camera.getimgs()
            self.c_detector.showdepth(current_depth)
            current_det_dict = self.c_detector.showcolor(current_color)
            herehere, therethere = self.c_board.do_stuff(current_det_dict)

            if therethere:
                #ax = fig.add_subplot(projection='3d')
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

                    ax2 = herehere.plot_frames_in("1", ax=ax2, alpha=0.6, s=0.01)
                    ax2.view_init(elev=90, azim=-90)
                    ax2.set_xlim((-0.1, 0.2))
                    ax2.set_ylim((-0.1, 0.4))
                    ax2.set_zlim((0, 0.4))

                    ax1 = herehere.plot_frames_in("1", ax=ax1, alpha=0.6, s=0.01)
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
            #gui.Application.instance.post_to_main_thread(self.window, self.update_geometry)
            #kk = cv2.waitKey(30)
                #plt.ion()
                #plt.show()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1)
                plt.clf()

if __name__ == "__main__":
    probetrial = Experiment()
    probetrial.run()
