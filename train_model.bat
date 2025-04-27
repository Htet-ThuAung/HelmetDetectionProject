@echo off
D:/Projects/tf-gpu/Scripts/python.exe D:/Projects/HelmetDetectionProject/my_package/object_detection/model_main_tf2.py ^
--model_dir=D:/Projects/HelmetDetectionProject/my_package/models/helmet_detection ^
--pipeline_config_path=D:/Projects/HelmetDetectionProject/my_package/models/helmet_detection/pipeline.config ^
--num_train_steps=50000 ^
--checkpoint_every_n=500
pause



