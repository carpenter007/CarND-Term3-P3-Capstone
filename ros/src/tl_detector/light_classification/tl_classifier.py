from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self):
        PATH_TO_MODEL = 'frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            # Works up to here.
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)
        return

    def get_classification(self, img):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
            confidence = round(scores[0][0]*100)
            if (confidence < 50.0):
                signal = TrafficLight.UNKNOWN
                signal_str = 'Unknown'
                confidence = 100 - confidence
            elif (classes[0][0] == 1):
                signal = TrafficLight.GREEN
                signal_str = 'Green'
            elif (classes[0][0] == 2):
                signal = TrafficLight.RED
                signal_str = 'Red'
            elif (classes[0][0] == 3):
                signal = TrafficLight.YELLOW
                signal_str = 'Yellow'
            else:
                signal = TrafficLight.UNKNOWN
                signal_str = 'Unknown'
            rospy.logdebug('signal:%s  confidence:%6.6f. ', signal_str, confidence)
        
        #TODO implement light color prediction
        return signal
