python D:/Projects/HelmetDetectionProject/my_package/object_detection/model_main_tf2.py --model_dir=D:/Projects/HelmetDetectionProject/my_package/models/helmet_detection_1 --pipeline_config_path=D:/Projects/HelmetDetectionProject/models/my_package/helmet_detection_1/pipeline.config --checkpoint_dir=D:/Projects/HelmetDetectionProject/my_package/models/helmet_detection_1


python D:/Projects/HelmetDetectionProject/my_package/object_detection/exporter_main_v2.py ^
  --input_type image_tensor ^
  --pipeline_config_path D:/Projects/HelmetDetectionProject/my_package/models/helmet_detection_1/pipeline.config  ^
  --trained_checkpoint_dir D:/projects/HelmetDetectionProject/my_package/models/helmet_detection_1
  --output_directory D:/Projects/HelmetDetectionProject/my_package/models/exported_model



python D:/projects/HelmetDetectionProject/my_package/object_detection/exporter_main_v2.py ^
--input_type image_tensor ^
--pipeline_config_path D:/Projects/HelmetDetectionProject/my_package/models/helmet_detection_1/pipeline.config ^
--trained_checkpoint_dir D:/Projects/HelmetDetectionProject/my_package/models/helmet_detection_1 ^
--output_directory D:/Projects/HelmetDetectionProject/my_package/models/exported_model




######### Eval Script.py #################
python helmet_detector/object_detection/model_main_tf2.py ^
  --model_dir=helmet_detector/models/helmet_detection_ssd_mnet_v2 ^
  --pipeline_config_path=helmet_detector/models/helmet_detection_ssd_mnet_v2/pipeline.config ^
  --checkpoint_dir=helmet_detector/models/helmet_detection_ssd_mnet_v2 ^
  --eval_timeout=0



### Tensorboard logs

python helmet_detector/object_detection/model_main_tf2.py ^
  --model_dir=helmet_detector/models/helmet_detection_ssd_mnet_v2 ^
  --pipeline_config_path=helmet_detector/models/helmet_detection_ssd_mnet_v2/pipeline.config ^
  --checkpoint_dir=helmet_detector/models/helmet_detection_ssd_mnet_v2 ^
  --eval_timeout=0


tensorboard --logdir=helmet_detector/models/helmet_detection_ssd_mnet_v2



tensorboard --logdir=D:/Projects/HelmetDetectionProject/my_package/models/helmet_detection_ssd_mnet_v2/train
