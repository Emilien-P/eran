import sys, os
from typing import Optional
import numpy as np
import onnxruntime.backend as rt

sys.path.insert(0, 'eran/ELINA/python_interface/')
sys.path.insert(0, 'eran/deepg/code/')

from .eran import ERAN
from .read_net_file import *
from .ai_milp import *
from .config import config
from .constraint_utils import *


def init_domain(d):
     if d == 'refinezono':
         return 'deepzono'
     elif d == 'refinepoly':
         return 'deeppoly'
     else:
         return d


def normalize(image, means, stds):
    # normalization taken out of the network
    if len(means) == len(image):
        for i in range(len(image)):
            image[i] -= means[i]
            image[i] /= stds[i]
    else:
        raise ValueError()


class ONNXAnalyzer:
    def __init__(self,
                 netname: str,
                 domain: str = config.domain,
                 timeout_lp: float = config.timeout_lp,
                 timeout_milp: float = config.timeout_milp,
                 use_default_heuristic: bool = config.use_default_heuristic,
                 default_epsilon: float = 0.01,
                 complete: bool = config.complete
                 ):
        ext = os.path.splitext(netname)[-1]
        self.runnable = None
        if ext == ".onnx":
            model, is_conv = read_onnx_net(netname)
            self.runnable = rt.prepare(model, 'CPU')
        elif ext in [".pyt", ".txt"]:
            # We assume 32x32x3 shape
            num_pixels = 32*32*3 
            # the means and stds are stored externally with the dataset to test
            model, is_conv, _, _ = read_tensorflow_net(netname, num_pixels, True)
        else:
            raise ValueError(f"Unsupported network format {ext}")
        
        self.eran = ERAN(model, is_onnx=(ext == ".onnx"))
        self.domain = domain
        self.timeout_lp = timeout_lp
        self.timeout_milp = timeout_milp
        self.use_default_heuristic = use_default_heuristic
        self.default_epsilon = default_epsilon
        self.complete = complete
        self.model = model

    def verify(self, image: np.ndarray, label: int, epsilon: Optional[float] = None, mean=[0.5, 0.5, 0.5], std=[1, 1, 1]) -> Optional[bool]:
        #assert np.min(image) >= 0 and np.max(image) <= 1, "Image pixel values should be in [0, 1]"
        if epsilon is None:
            epsilon = self.default_epsilon
        specLB = np.copy(image)
        specUB = np.copy(image)

        normalize(specLB, mean, std)
        normalize(specUB, mean, std)

        specLB = specLB.transpose(1, 2, 0)
        specUB = specUB.transpose(1, 2, 0)

        pred_label, nn, nlb, nub = self.eran.analyze_box(specLB,
                                                         specUB,
                                                         init_domain(self.domain),
                                                         self.timeout_lp,
                                                         self.timeout_milp,
                                                         self.use_default_heuristic)

        print(f"verified {nlb[-1]} {nub[-1]}")
        print(f"True label is {label}")
        print(f"ERAN analyze box predicted label {pred_label}")
        if self.runnable:
            pred_label2 = self.runnable.run(np.expand_dims(specLB.transpose(2, 0, 1), axis=0))
            print(f"onnx runtime predicted label {pred_label2}")

        if label == pred_label:
            specLB = np.clip(np.copy(image) - epsilon, 0, 1)
            specUB = np.clip(np.copy(image) + epsilon, 0, 1)

            normalize(specLB, mean, std)
            normalize(specUB, mean, std)
            specLB = specLB.transpose(1, 2, 0)
            specUB = specUB.transpose(1, 2, 0)

            perturbed_label, _, nlb, nub = self.eran.analyze_box(specLB, specUB, self.domain, self.timeout_lp,
                                                                 self.timeout_milp, self.use_default_heuristic)
            print("nlb ", nlb[len(nlb) - 1], " nub ", nub[len(nub) - 1])
            if perturbed_label == label:
                print("img verified", label)
                return True, nlb, nub
            else:
                if self.complete:
                    constraints = get_constraints_for_dominant_label(label, 10)
                    verified_flag, adv_image = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
                    if verified_flag:
                        return True, nlb, nub
                    else:
                        cex_label, _, _, _ = self.eran.analyze_box(adv_image, adv_image, 'deepzono', config.timeout_lp,
                                                              config.timeout_milp, config.use_default_heuristic)
                        if cex_label != label:
                            print("adversarial image ", adv_image, "cex label", cex_label, "correct label ", label)
                else:
                    return False, nlb, nub
        else:
            print("img not considered, correct_label", label, "classified label ", pred_label)
            return None, None, None
