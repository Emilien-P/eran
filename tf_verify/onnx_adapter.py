import sys, os
from typing import Optional
import numpy as np

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
        assert os.path.splitext(netname)[-1] == ".onnx", "unrecognized netname extension"
        model, is_conv = read_onnx_net(netname)
        self.eran = ERAN(model, is_onnx=True)
        self.domain = domain
        self.timeout_lp = timeout_lp
        self.timeout_milp = timeout_milp
        self.use_default_heuristic = use_default_heuristic
        self.default_epsilon = default_epsilon
        self.complete = complete

    def verify(self, image: np.ndarray, label: int, epsilon: Optional[float] = None) -> Optional[bool]:
#        assert np.min(image.numpy()) >= 0 and np.max(image.numpy()) <= 1, "Image pixel values should be in [0, 1]"
        if epsilon is None:
            epsilon = self.default_epsilon
        specLB = np.copy(image)
        specUB = np.copy(image)

        pred_label, nn, nlb, nub = self.eran.analyze_box(specLB,
                                                         specUB,
                                                         init_domain(self.domain),
                                                         self.timeout_lp,
                                                         self.timeout_milp,
                                                         self.use_default_heuristic)
        if label == pred_label:
            specLB = np.clip(image - epsilon, 0, 1)
            specUB = np.clip(image + epsilon, 0, 1)
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
