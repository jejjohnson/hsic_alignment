from typing import Optional
import scipy.io as scio


class MIData:
    """MI Data
    
    
    Dataset
    -------
    trials = 1:5
    samples = 50, 100, 500, 1_000, 5_000
    dimensions = 2, 3, 10, 50, 100
    std = 1:11
    nu = 1:9
    """

    def __init__(self, distribution: Optional["gauss"] = None) -> None:

        self.distribution = distribution
        self.data_path = "/media/disk/erc/papers/2019_HSIC_ALIGN/data/mi_distributions/"

        if self.distribution == "gauss":
            self.dist_path = f"{self.data_path}MI_gaus/"
        elif self.distribution == "tstudent":
            self.dist_path = f"{self.data_path}MI_tstu/"
        else:
            raise ValueError(f"Unrecognized Dataset: {distribution}")

    def get_data(self, samples=50, dimensions=2, std=1, trial=1, nu=1):

        if self.distribution == "gauss":
            dat = scio.loadmat(
                f"{self.dist_path}DATA_MI_gaus_nd_{dimensions}_"
                f"Ns_{samples}_std_{std}_tryal_{trial}.mat"
            )

            return dat["X"], dat["Y"], float(dat["MI_ori_nats"][0][0])
        elif self.distribution == "tstudent":
            dat = scio.loadmat(
                f"{self.dist_path}DATA_MI_tstu_nd_{dimensions}_"
                f"Ns_{samples}_tryal_{trial}_nu_{nu}.mat"
            )

            return dat["X"], dat["Y"], float(dat["MI_ori_nats"][0][0])
        else:
            raise ValueError(f"Unrecognized distribution '{self.distribution}'")
