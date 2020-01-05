#DODO Detector
import os
import cv2
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from abc import ABCMeta,abstractmethod
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from warnings import warn

class objectDetector():
    #Base class for object detectors used by the package

    __metaclass__ = ABCMeta

    def __init__(self):
        #Create logger
        self._logger = logging.getLogger('dodo_detector')
        self._logger.setLevel(logging.DEBUG)
        #Create file handler which logs even debug messages
        self._fh = logging.FileHandler('/tmp/dodo_detector.log')
        self._fh.setLevel(logging.DEBUG)
        #Create console handler with a higher log level
        self._ch = logging.StreamHandler()
        self._ch.setLevel(logging.DEBUG)
        #Create formatter and add it to handlers
        self._formatter = logging.Formatter('[%(asctime)s - %(name)s]: %(levelname)s: %(message)s')
        self._fh.setFormatter(self._formatter)
        self._ch.setFormatter(self._formatter)
        #Add handlers to the logger
        self._logger.addHandler(self._fh)
        self._logger.addHandler(self._ch)

    @abstractmethod
    def from_image(self,frame):
        """Detects objects in an image
        :param frame: A numpy.ndarray containing the image where objects will be detected.
        :return: a tuple containing the image, with objects marked by rectangles.
        """
        pass

    def _detect_from_stream(self,get_frame,stream):
        """
        This internal methods detects objects from images retrieved from a stream, given a method that extracts frames from this
        :param get_frame: a method that extracts frames from the stream
        :param stream: an object representing a stream of images
        """
        ret,frame = get_frame(stream)

        while ret:
            marked_frame,objects = self.from_image(frame)
            cv2.imshow("detection",marked_frame)
            if cv2.waitkey(1) == 27:
                break #ESC to quit
            ret,frame = get_frame(stream)
        cv2.destroyAllWindows()

    def from_camera(self,camera_id = 0):
        """
        Detects objects in frames from a camera feed
        :param camera_id: the ID of the camera in the system
        """
        def get_frame(stream):
            frame = stream.read()
            ret = True
            return ret,frame
        stream = WebcamVideoStream(src=camera_id)
        stream.start()
        self._detect_from_stream(get_frame,stream)
        stream.stop()

    def from_video(self,filepath):
        """
        Detects objects in frames from a video file
        :param filepath: the path to the video file
        """
        def get_frame(stream):
            ret,frame = stream.read()
            return ret,frame
        stream = cv2.VideoCapture()
        stream.open(filename = filepath)

        self._detect_from_stream(get_frame,stream)

    @staticmethod
    def is_rgb(im):
        return len(im.shape) == 3 and im.shape[2] == 3

    @staticmethod
    def to_rgb(im):
        w,h = im.shape[0],im.shape[1]
        ret = np.empty((w,h,3),dtype = np.uint8)
        ret[:,:,2] = ret[:,:,1] = ret[:,:,0] = im
        return ret


class KeypointObjectDetector(objectDetector):
    """
    Object detector based o keypoints. This class depends on openCV SIFT and SURF feature detection algorithms,
    as well as the brute-force and FLANN-based feature matchers.
    :param database_path: Path to the top-level directory containing subdirectories , each subdirectory named after a class of objects and containing images of that object.
    :param detector_type: either SURF or SIFT or RootSIFT
    :param matcher_type: either BF for brute-force or 'FLANN' for flann based matcher.
    :param min_points: minimum number of keypoints necessary for an object to be considered detected in an image
    """

    @property
    def detector_type(self):
        return self._detector_type

    @detector_type.setter
    def detector_type(self,value):
        #create the detector
        if value in ['SIFT','RootSIFT']:
            self.detector = cv2.xfeatures2d.SIFT_create()
        elif value == 'SURF':
            self.detector = cv2.features2d.SURF_create()
        else:
            raise ValueError('Invalid detector type')
        self._detector_type = value

    @property
    def matcher_type(self):
        return

    @matcher_type.setter
    def matcher_type(self,value):
        #get which openCV feature matcher the user wants.
        if value == 'BF':
            self.matcher = cv2.BFMatcher()
        elif value == 'FLANN':
            flann_index_kdtree = 0
            index_params = dict(algorithm = flann_index_kdtree,trees=5)
            search_params = dict(checks=50) #or pass an empty dictionary
            self.matcher = cv2.FlannBasedMatcher(index_params,search_params)
        else:
            raise ValueError('Invalid matcher type')
        self._matcher_type = value

    @property
    def database_path(self):
        return self._database_path

    @property
    def categories(self):
        return self._categories

    @database_path.setter
    def database_path(self,value):
        #get the directory where object textures are stored
        self._database_path = value
        if self._database_path[-1] != '/':
            self._database_path += '/'
        #store object classes in a list
        #each directory in the object database corresponds to a class
        self._categories = [os.path.basename(d) for d in os.listdir(self._database_path)]
        #minimum object dimensions in pixels
        self.min_object_height = 10
        self.min_object_width = 10
        self.min_object_area = self.min_object_height * self.min_object_width
        #initialize the frame counter for each object class at 0
        self.object_counters = {}
        for ob in self.categories:
            self.object_counters[ob] = 0
        #load features for each texture and store the image
        #its keypoints and corresponding descriptor
        self.object_features = {}
        for obj in self.categories:
            self.object_features[obj] = self._load_features(obj)

    def __init__(self,database_path,detector_type='RootSIFT',matcher_type='BF',min_points = 10, logging = False):
        super(objectDetector,self).__init__()
        self.current_frame = 0
        #These things are properties, take a look at their setters in this class
        self.detector_type = detector_type
        self.matcher_type = matcher_type
        self.database_path = database_path
        self.min_points = min_points

    def _load_features(self,object_name):
        """
        Given the name of an object class from the image database directory, this method iterates through all the images contained in that directory extracting their keypoints and
        descriptors using the desired feature detector
        :param object_name: the name of the object class, whose image directory is contained inside the image database directory
        :returns: a list of tuples, each tuple containing the processed image as a greyscale numpy.ndarray, its keypoints and descriptors
        """
        img_files = [
            os.path.join(self.database_path + object_name + '/' + f) for f in os.listdir(self.database_path + object_name + '/')
            if os.path.isfile(os.path.join(self.database_path + object_name + '/' ,f))]
        pbar = tqdm(desc = object_name, total = len(img_files)) #Progress bar

        #extract the keypoints from all images in the database
        features = []
        for img_file in img_files:
            pbar.update()
            img = cv2.imread(img_file)

            #scaling_factor = 640/img.shape[0]
            if img.shape[0] > 1000:
                img = cv2.resize(img,(0,0),fx=0.3,fy=0.3)

            #find keypoints and descriptors with the selected feature detector
            keypoints,descriptors = self._detectAndCompute(img)
            features.append((img,keypoints,descriptors))
        return features

    def _detect_object(self,name,img_features,scene):
        """
        name: Name of the object class
        img_features: A list of tuples, each tuple containing three elements: an image, its keypoints and descriptor
        scene: the image where the object name will be detected
        returns a tuple containing two elements the homography matrix and the coordinates of a rectangle containing the object
        """
        scene_kp,scene_descs = self._detectAndCompute(scene)
        for img_feature in img_features:
            obj_image,obj_keypoints,obj_descriptors = img_feature

            if obj_descriptors is not None and len(obj_descriptors) > 0 and scene_descs is not None and len(scene_descs)>0:
                matches = self.matcher.knnMatch(obj_descriptors,scene_descs,k=2)
                #Store all the good matches as per Lowe's ratio test
                good = []
                for match in matches:
                    if len(match) == 2:
                        m,n = match
                        if m.distance < 0.7 * n.distance:
                            good.append(m)
                #If an object is detected
                if len(good) > self.min_points:
                    self.object_counters[name] += 1
                    source_points = np.float32([obj_keypoints[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                    destination_points = np.float32([scene_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                    M,mask = cv2.findHomography(source_points,destination_points,cv2.RANSAC,5.0)

                    if M is None:
                        break

                    if (len(obj_image.shape) == 2):
                        h,w = obj_image.shape
                    else:
                        h,w,_ = obj_image.shape

                    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

                    #dst contains the coordinates of the vertices of the drawn rectangle
                    dst = np.int32(cv2.perspectiveTransform(pts,M))

                    #get the min and max x and y coordinates of the object
                    xmin = min(dst[x,0][0] for x in range(4))
                    xmax = max(dst[x,0][0] for x in range(4))
                    ymin = min(dst[x,0][1] for x in range(4))
                    ymax = max(dst[x,0][1] for x in range(4))

                    #Transform homography into a simpler data structure
                    dst = np.array([point[0] for point in dst],dtype = np.int32)

                    #Returns the homography and a rectangle containing the object
                    return dst,[xmin,xmax,ymin,ymax]
        return None,None

    def _detectAndCompute(self,image):
        """
        Detects keypoints and generates descriptors according to the desired algorithm
        image: a numpy.ndarray containing the image whose keypoints and descriptors will be processed
        """
        keypoints,descriptors = self.detector.detectAndCompute(image,None)
        if self.detector_type == 'RootSIFT' and len(keypoints) > 0:
            #Transforms SIFT descriptors into RootSIFT descriptors
            #Apply the Hellinger kernel by first L1-normalizing
            #and taking the square root
            eps = 1e-7
            descriptors /= (descriptors.sum(axis=1,keepdims=True) + eps)
            descriptors = np.sqrt(descriptors)

        return keypoints,descriptors

    def from_image(self,frame):
        self.current_frame += 1
        #Operations on the frame come here
        detected_objects = {}
        for object_name in self.object_features:
            homography,rct = self._detect_object(object_name,self.object_features[object_name],frame)

            if rct is not None:
                xmin = rct[0]
                ymin = rct[1]
                xmax = rct[2]
                ymax = rct[3]

                if object_name not in detected_objects:
                    detected_objects[object_name] = []

                detected_objects[object_name].append({'box':(ymin,xmin,ymax,xmax)})
                text_point = (homography[0][0],homography[1][1])
                homography = homography.reshape((-1,1,2))
                cv2.polylines(frame,[homography],True,(0,255,255),10)
                cv2.putText(frame,object_name + ':' + str(self.object_counters[object_name]),text_point,xv2.FONT_HERSHEY_COMPLEX_SMALL,1.2,(0,0,0),2)
        return frame,detected_objects


class singleShotDetector(objectDetector):
    """
    Object detector powered by the TensorFlow object detection API
    path_to_frozen_graph: path to the frozen inference graph file, a file with a '.pb' extension
    path_to_labels: path to the label map, a text file with the '.pbtxt' extension
    num_classes: number of object classes that is detected.
    confidence: a value between 0 and 1 representing the confidence level the network has in the detection to consider it an actual detection
    """

    def __init__(self,path_to_frozen_graph,path_to_labels,num_classes = None,confidence = 0.8):
        super(objectDetector,self).__init__()

        if not 0< confidence <= 1:
            raise ValueError ("Confidence must be between 0 and 1")

        self._detection_graph = tf.Graph()
        with self._detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_frozen_graph,'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def,name='')

        #label maps map indices to category names, so that when our convolution
        #network predicts 5, we know that this corresponds to airplane.
        #Here we use internal utility functions, but anything that returns a
        #dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(path_to_labels)

        #This is the workaround to guess the number of classes by the contents of the label map
        #it may not be perfect
        label_map_contents = open(path_to_labels,'r').read()
        suggested_num_classes = label_map_contents.count('name:')
        if num_classes is None:
            num_classes = suggested_num_classes
        elif num_classes != suggested_num_classes:
            warn('Suggested number of classes according to label map is {0}. Will use {1} as manually passed.'.format(suggested_num_classes,num_classes))
        categories = label_map_util.create_category_index(categories)
        self._category_index = label_map_util.create_category_index(categories)

        self._categories = {}
        self._categories_public = {}
        for tmp in categories:
            self._categories[int(tmp['id'])] = tmp['name']
            self._categories_public.append(tmp['name'])
        self._confidence = confidence

        #Create a sesion that will be used until the detector is set on fire by the gc
        self._session = tf.Session(graph = self._detection_graph)

    @property
    def confidence(self):
        return self._confidence

    @property
    def categories(self):
        return self._categories_public

    @confidence.setter
    def confidence(self,value):
        self._confidence = value

    def from_image(self,frame):
        #object recognition begins here
        if not objectDetector.is_rgb(frame):
            frame = objectDetector.to_rgb(frame)
        height,width,z = frame.shape

        image_np_expanded = np.expand_dims(frame,axis=0)
        image_tensor = self._detection_graph.get_tensor_by_name('image_tensor:0')
        #Each box represents a part of the image where a particular object is detected
        boxes = self._detection_graph.get_tensor_by_name('detection_boxes:0')
        #Each score represents how level of confidence for each of the objects.
        #Score is shown on the result image together with the class label.
        scores = self._detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self._detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self._detection_graph.get_tensor_by_name('num_detections:0')

        #Actual detection
        boxes,scores,classes,num_detections = self._session.run([],feed_dict = {image_tensor:image_np_expanded})

        #Count the number of scores are above the designated threshold
        worthy_detections = sum(score >= self._confidence for score in scores[0])
        detected_objects = {}

        for x in range(worthy_detections):
            class_name = self._categories[int(classes[0][x])]
            box_objects = boxes[0][x]

            ymin = int(box_objects[0]*height)
            xmin = int(box_objects[1]*width)
            ymax = int(box_objects[2]*height)
            xmax = int(box_objects[3]*width)

            if class_name not in detected_objects:
                detected_objects[class_name] = []

            detected_objects[class_name].append({'box':(ymin, xmin, ymax, xmax),'confidence':scores[0][x]})

        #Visualization of the results of a detection
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self._category_index,
            use_normalized_coordinates = True,
            line_thickness = 8,
            min_score_thresh = self_confidence)

        return frame,detected_objects
    
    
 
        

    
        
