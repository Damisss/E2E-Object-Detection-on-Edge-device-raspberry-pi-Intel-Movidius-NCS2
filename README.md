# E2E-Object-Detection-in-Edge-device-Intel-Movidius-NCS2.
<img src="/result.gif" width="400" height="400"/>

This project shows how to train custom object detection model using TensorFlow Object Detection API. It also, demostrate how to deploy the trained model to Intel Movidius Neural Compute Stick (NCS2).
env-setup file provides a step by step guide for stting up environment on ubuntu machine. It also provides the commands for starting training process, evaluation and exporting inference graph. It is import to underline that the python script files such as xml_to_csv.py and generate_tfrecord.py files attached to this repo were provided by TFOD team.

# Convert TensorFlow Object Detection to Openvino Intermediate representation
Please follow [Openvino](https://docs.openvinotoolkit.org/latest/index.html) to setup your env. 
After setting up the env, please run below command provided by openvino team to generate Intermediate representation graph (model topography and weights files ) 
- ***python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py -m frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config  --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --data_type FP16 --generate_deprecated_IR_V7***

# Setting up Raspberry pi
Please follow this [Pyimagesearch](https://www.pyimagesearch.com/2018/02/12/getting-started-with-the-intel-movidius-neural-compute-stick/) to setup the rasberry pi for Movidius NCS2

# Inference:
To run real time inference:
cd deployment:
 - python detect_realtime.py --topo models/frozen_inference_graph.xml --weights models/frozen_inference_graph.bin 

# References
- https://www.pyimagesearch.com/raspberry-pi-for-computer-vision/
- https://docs.openvinotoolkit.org/latest/index.html
- https://www.pyimagesearch.com/2018/02/12/getting-started-with-the-intel-movidius-neural-compute-stick/

