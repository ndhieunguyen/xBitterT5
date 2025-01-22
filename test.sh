CUDA_VISIBLE_DEVICES=0 python3 test.py \
        --model_ckpt "cbbl-skku-org/xBitterT5-720" \
        --test_csv "data/BTP720/test.csv" \
        --device "cuda" \
        --batch_size 1