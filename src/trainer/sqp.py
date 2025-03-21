from itertools import product
from typing import Tuple

import jax
import jax.numpy as jnp

from src.trainer.generalized_augmented import (
    GeneralizedAugmentedLagrangianTrainer,
)


class StopGradientQuadraticPenaltyTrainer(GeneralizedAugmentedLagrangianTrainer):
    def compute_orthogonality_loss(self, params, error_matrix_dict):
        # Get params
        dual_variables = params['duals']
        barrier_coefficients = params['barrier_coefs']
        quadratic_error_matrix = error_matrix_dict['quadratic_errors']

        # Compute dual loss
        dual_loss = 0.0

        # Compute barrier loss
        quadratic_error = quadratic_error_matrix.sum()
        barrier_loss = (
            jax.lax.stop_gradient(barrier_coefficients[0, 0]) * quadratic_error
        )

        # Generate dictionary with dual variables and errors for logging
        dual_dict = {
            f'beta({i},{j})': dual_variables[i, j]
            for i, j in product(range(self.d), range(self.d))
            if i >= j
        }
        barrier_dict = {
            f'barrier_coeff': barrier_coefficients[0, 0],
        }
        dual_dict.update(barrier_dict)

        return dual_loss, barrier_loss, dual_dict

    def update_barrier_coefficients(self, params, *args, **kwargs):
        """Update barrier coefficients using some approximation
        of the barrier gradient in the modified lagrangian.
        """
        barrier_coefficients = params['barrier_coefs']
        quadratic_error_matrix = params['quadratic_errors']
        updates = jnp.clip(quadratic_error_matrix, a_min=0, a_max=None).mean()

        # Calculate updated coefficients
        updated_barrier_coefficients = (
            barrier_coefficients + self.lr_barrier_coefs * updates
        )

        # Clip coefficients to be in the range [min_barrier_coefs, max_barrier_coefs]
        updated_barrier_coefficients = jnp.clip(
            updated_barrier_coefficients,
            a_min=self.min_barrier_coefs,
            a_max=self.max_barrier_coefs,
        )  # TODO: Cliping is probably not the best way to handle this

        # Update params, making sure that the coefficients are lower triangular
        params['barrier_coefs'] = updated_barrier_coefficients
        return params

    def loss_function(self, params, train_batch, **kwargs) -> Tuple[jnp.ndarray]:
        loss, aux = super().loss_function(params, train_batch, **kwargs)
        barrier_coefficient = params['barrier_coefs'][0, 0]

        # Normalize loss by barrier coefficient
        if self.use_barrier_normalization:
            loss /= jax.lax.stop_gradient(barrier_coefficient)

        return loss, aux

    def update_duals(self, params):
        """Leave duals unchanged."""
        return params

    def init_additional_params(self, *args, **kwargs):
        additional_params = {
            'duals': jnp.zeros((self.d, self.d)),
            'barrier_coefs': jnp.tril(self.barrier_initial_val * jnp.ones((1, 1)), k=0),
            'dual_velocities': jnp.zeros((self.d, self.d)),
            'errors': jnp.zeros((self.d, self.d)),
            'quadratic_errors': jnp.zeros((1, 1)),
        }
        return additional_params
