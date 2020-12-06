# E2E-Object-Detection-in-Edge-device-Intel-Movidius-NCS2.

This project shows how to train custom object detection model using TensorFlow Object Detection API and model deployment in Intel-Movidius-NCS2.
env-setup file contains the environment steps on unbutu machine and how to start training process, evaluation and exporting inference graph.

# Convert TensorFlow Object Detection to Openvino Intermediate representation
Please follow [Openvino](https://docs.openvinotoolkit.org/latest/index.html) to setup your env. 
After setting up the env, please run below command from openvino to generate Intermediate representation graph (model topography and weights files ) 
- ***python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py -m frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config  --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --data_type FP16 --generate_deprecated_IR_V7***
# Setting up Raspberry pi
Please follow this [Pyimagesearch](https://www.pyimagesearch.com/2018/02/12/getting-started-with-the-intel-movidius-neural-compute-stick/) to setup the rasberry pi for Movidius NCS2
# References
- https://www.pyimagesearch.com/raspberry-pi-for-computer-vision/
- https://docs.openvinotoolkit.org/latest/index.html
- https://www.pyimagesearch.com/2018/02/12/getting-started-with-the-intel-movidius-neural-compute-stick/

