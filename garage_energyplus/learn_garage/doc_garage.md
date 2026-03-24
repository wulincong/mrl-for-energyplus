## 在GPU/CPU上运行实验

tf仅使用CPU（tf默认使用GPU加速）

```
export CUDA_VISIBLE_DEVICES=-1  # CPU only
python path/to/my/experiment/launcher.py
```

torch使用GPU加速

```
import torch
from garage.torch import set_gpu_mode

# ...

if torch.cuda.is_available():
    set_gpu_mode(True)
else:
    set_gpu_mode(False)
algo.to()

```


