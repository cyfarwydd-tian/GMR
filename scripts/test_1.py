import numpy as np
d = np.load("./motion_data/tennis_smplx_global.npz", allow_pickle=True)
print("betas shape:", d["betas"].shape)   # 你应该会看到 (..., 20)
