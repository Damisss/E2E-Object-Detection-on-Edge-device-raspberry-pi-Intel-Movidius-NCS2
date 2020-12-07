#Usage
# python detect_realtime.py --topo models/frozen_inference_graph.xml --weights models/frozen_inference_graph.bin 
import cv2
import argparse
import numpy as np 
from nn.IR_loader import InferenceNetwork
import datetime 
import pickle

ap = argparse.ArgumentParser()
#parse arguments
ap.add_argument('-i', '--input', help='Path to input video')
ap.add_argument('-t', '--topo', required=True, help='Path to intermediate xml file (this file contains the topography of the network)')
ap.add_argument('-w', '--weights', required=True, help='Path to intermediate bin file (this file contains models\' weights ')
ap.add_argument('-c', '--confidence', default=.6, type=float, help='Minimum proba to filter weak detections')

args = vars(ap.parse_args())

class AsyncInference ():
    @staticmethod
    def start ():
        try:
            CLASSES=pickle.loads(open('models/labels.pickle', 'rb').read())
            CLASSES[0] = 'backgound'
            COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
            net, execNet = InferenceNetwork.load(args['topo'], args['weights'])
            #prepare blob
            inputBlob = next(iter(net.inputs))
            outputBlob = next(iter(net.outputs))
            # set default batch size 
            net.batch_size = 1
            # grab number of input blobs, number channel, height and width of input blob
            n, c, h, w = net.inputs[inputBlob].shape
            
            # requestds ids
            currentRequestId, nextRequestId = 0, 1
            # input image heightand width
            H, W = None, None
            
            # Use camera if there si no input video
            if not args.get('input', False):
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(args['input'])
                assert cap.isOpened(), "Can't open " + args['input']
            
            # read first frame
            ret, frame = cap.read()
            H, W = frame.shape[:2]

            start = datetime.datetime.now()
            num_frame = 0
            while cap.isOpened():
                ret, nextFrame = cap.read()
                # stop while loop if is there is no frame available 
                if not ret:
                    break
               
                
                key = cv2.waitKey(1) & 0xFF 
                #stop while loop if q is pressed
                if  key == ord('q'):
                    break
                
                if H is None or W is None:
                    H, W = nextFrame.shape[:2]

                # data preprocessing
                inFrame = cv2.resize(nextFrame, (300, 300))
                # change image data shape from HWC to CHW
                inFrame = inFrame.transpose((2, 0, 1)) 
                # reshape image
                inFrame = inFrame.reshape((n, c, h, w))
                # start async 
                execNet.start_async(request_id=nextRequestId, inputs={inputBlob: inFrame})
                if execNet.requests[currentRequestId].wait(-1) == 0:
                    results = execNet.requests[currentRequestId].outputs[outputBlob]

                    for obj in results[0][0]:
                       confidence = obj[2]
                       indx = int(obj[1])

                       if confidence > args['confidence']:
                           #bounding box coordinate
                           startX = int(obj[3] * W)
                           startY = int(obj[4] * H)
                           endX = int(obj[5] * W)
                           endY = int(obj[6] * H)
                           text = '{} {:.2f}%'.format(CLASSES[indx], confidence * 100)
                           y = startY - 15 if startY - 15 > 15 else startY + 15
                           font=cv2.FONT_HERSHEY_SIMPLEX
                           cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[indx], 2)
                           cv2.putText(frame, str(text), (startX, y), font, .5,  COLORS[indx], 2)
                           
                num_frame += 1
                elaps = (datetime.datetime.now() - start).total_seconds()
                fps = num_frame/elaps
                cv2.putText(frame, 'Average FPS: {:.2f}'.format(fps), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .5,  (255, 0, 0), 2) 
                # show image
                cv2.imshow('Birds', frame)
                currentRequestId, nextRequestId = nextRequestId, currentRequestId
                frame = nextFrame
                H, W = frame.shape[:2]
            
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            raise e


if __name__ == '__main__':
    AsyncInference.start()