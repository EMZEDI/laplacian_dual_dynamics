from src.trainer.trainer import Trainer
from src.trainer.laplacian_encoder import LaplacianEncoderTrainer
from src.trainer.generalized_gdo import GeneralizedGraphDrawingObjectiveTrainer
from src.trainer.generalized_augmented import GeneralizedAugmentedLagrangianTrainer
from src.trainer.al_dos import AugmentedLagrangianRegularizedTrainer
from src.trainer.al import AugmentedLagrangianTrainer
from src.trainer.quadratic_penalty_ggdo import QuadraticPenaltyGGDOTrainer
from src.trainer.sqp import StopGradientQuadraticPenaltyTrainer as SQPTrainer
from src.trainer.cqp import (
    CoefficientSymmetryBreakingQuadraticPenaltyTrainer as CQPTrainer,
)
from src.trainer.gnn import LaplacianGNNTrainer