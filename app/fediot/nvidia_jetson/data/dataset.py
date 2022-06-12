import numpy as np
import pandas as pd
import os


device_list = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]

benign_data = pd.concat(
    [
        pd.read_csv(os.path.join("./N-BaIoT", device, "benign_traffic.csv"))[:5000]
        for device in device_list
    ]
)
np.savetxt("./N-BaIoT/min_dataset.txt", np.array(benign_data).min(axis=0))
np.savetxt("./N-BaIoT/max_dataset.txt", np.array(benign_data).max(axis=0))
