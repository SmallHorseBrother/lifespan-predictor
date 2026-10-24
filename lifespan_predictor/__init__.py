"""寿命预测器 Lifespan Predictor —— 模拟估算，不用于诊断。"""

from .exposures import Profile
from .scenarios import build_scenarios
from .simulate import MODEL_VERSION, SimulationResult, simulate

__version__ = "0.1.0"
__all__ = ["Profile", "SimulationResult", "MODEL_VERSION", "build_scenarios", "simulate", "__version__"]
