import numpy as np
import tensorflow as tf
import cv2
import time
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from alexnet import AlexNet
from caffe_classes import class_names
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    pose = 'faceFront'
    #pose = input('Enter the pose(Front, Back, or SidePose): ')
    model_path = 'C:/Users/Asus/3D Objects/Human Detection/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    video = cv2.VideoCapture('C:/Users/Asus/3D Objects/Human Detection/Pose_Alex/v1.mp4')

    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    val = 0
    imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('FastRcnn_TLD_jeet(12).mp4',fourcc,12, (720,1280))

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]
    #print(tracker_type)

 
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x = tf.placeholder(tf.float32, [1, 227, 227, 3])
        keep_prob = tf.placeholder(tf.float32)
        print("3")
        #create model with default config ( == no skip_layer and 1000 units in the last layer)
        model = AlexNet(x, keep_prob, 3, [])

        #define activation of last layer as score
        score = model.fc8

        #create op to calculate softmax 
        softmax = tf.nn.softmax(score)
        sess.run(tf.global_variables_initializer())
        model.load_initial_weights(sess)

        while True:

     
            # Exit if video not opened.
            if not video.isOpened():
                print ("Could not open video")
                sys.exit()
         
            # Read first frame.
            ok, frame = video.read()
            if not ok:
                print ('Cannot read video file')
                sys.exit()
             
            # Define an initial bounding box
            bbox = (287, 23, 86, 320)
             
            # Uncomment the line below to select a different bounding box
            # bbox = cv2.selectROI(frame, False)
            val = 0
            #print(bbox)
            
            r, img = video.read()
            boxes, scores, classes, num = odapi.processFrame(img)
            temp =  boxes[0]
            print(temp[0])
            print(boxes)
            print(boxes[0])

            bbox =  [temp[1],temp[0],temp[3]-temp[1],temp[2]-temp[0]]
            print("bbox:",tuple(bbox))

            ok = tracker.init(frame, tuple(bbox))
            bbox = tuple(bbox)

            w = bbox[2]
            h = bbox[3]
            crop_img = img[int(bbox[1]):int(bbox[1]+h), int(bbox[0]):int(bbox[0]+w)]
            crop_img = cv2.resize(crop_img.astype(np.float32), (227, 227))
            crop_img_r = crop_img.reshape((1,227,227,3))
            #cv2.imwrite(str(val)+"cropped.jpg", crop_img)

            crop_img_r -= imagenet_mean
                                    
            # Reshape as needed to feed into model
            crop_img_r = crop_img_r.reshape((1,227,227,3))
                                
            # Run the session and calculate the class probability
            probs = sess.run(softmax, feed_dict={x: crop_img_r, keep_prob: 1})
                                
            # Get the class name of the class with the highest probability
            class_name = class_names[np.argmax(probs)]
            print(class_name)
            cv2.imshow("Check_Pose", frame)
            cv2.waitKey(1)
            if (class_name == pose):
                break;

        while True:
            # Read a new frame
            ok, frame = video.read()
            if not ok:
                break
             
            # Start timer
            timer = cv2.getTickCount()
     
            # Update tracker
            ok, bbox = tracker.update(frame)

     
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
     
            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                
                w = bbox[2]
                h = bbox[3]
                crop_img = img[int(bbox[1]):int(bbox[1]+h), int(bbox[0]):int(bbox[0]+w)]
                #cv2.imwrite(str(val)+"cropped.jpg", crop_img)
                val = val + 1
                        

            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                
            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
         
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
            out.write(frame)

            # Display result
            cv2.imshow("Tracking", frame)
             
            # Exit if ESC pressed
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'): break

            
        out.release()
        video.release()
            #cv2.destroyAllWindows()
            


    
##    while True:
##        # Read a new frame
##        ok, frame = video.read()
##        if not ok:
##            break
##         
##        # Start timer
##        timer = cv2.getTickCount()
## 
##        # Update tracker
##        ok, bbox = tracker.update(frame)
##
## 
##        # Calculate Frames per second (FPS)
##        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
## 
##        # Draw bounding box
##        if ok:
##            # Tracking success
##            p1 = (int(bbox[0]), int(bbox[1]))
##            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
##            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
##            
##            w = bbox[2]
##            h = bbox[3]
##            crop_img = img[int(bbox[1]):int(bbox[1]+h), int(bbox[0]):int(bbox[0]+w)]
##            #cv2.imwrite(str(val)+"cropped.jpg", crop_img)
##            val = val + 1
##                    
##
##        else :
##            # Tracking failure
##            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
##            
##        # Display tracker type on frame
##        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
##     
##        # Display FPS on frame
##        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
##        out.write(frame)
##
##        # Display result
##        cv2.imshow("Tracking", frame)
##         
##        # Exit if ESC pressed
##        k = cv2.waitKey(1)
##        if k & 0xFF == ord('q'): break
##
##        
##    out.release()
##    video.release()
##        #cv2.destroyAllWindows()
