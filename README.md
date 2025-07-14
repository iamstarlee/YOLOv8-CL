# 0. 多卡并行训练和推理
```python
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 --master_port=25641 train.py
```
```python
CUDA_VISIBLE_DEVI=0 python predict.py
```

# bad guy is ...