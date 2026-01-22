cd /d %~dp0\..
python -m src.main_finetune --model convnextv2_tiny --batch_size 2 --epochs 1 ^
    --data_set image_folder ^
    --data_path data_row\fake_imagenet\train ^
    --eval_data_path data_row\fake_imagenet\val ^
    --nb_classes 2 ^
    --output_dir outputs\finetune_fake ^
    --log_dir outputs\finetune_fake\tb ^
    --num_workers 0 --device cpu