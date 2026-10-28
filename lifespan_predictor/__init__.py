"""寿命预测器 Lifespan Predictor —— 模拟估算，不用于诊断。"""

from .exposures import Profile
from .plan import Adherence, ChangePlan, make_plan
from .scenarios import build_scenarios
from .simulate import MODEL_VERSION, SimulationResult, simulate

__version__ = "0.2.0"
__all__ = [
    "Adherence",
    "ChangePlan",
    "Profile",
    "SimulationResult",
    "MODEL_VERSION",
    "build_scenarios",
    "make_plan",
    "simulate",
    "__version__",
]
