import torch
import torch.nn as nn
import torch.nn.functional as F


class ALLOLoss(nn.Module):
    def __init__(
        self,
        d: int,
        barrier_initial_val: float = 2.0,
        lr_duals: float = 0.01,
        lr_barrier_coefs: float = 0.01,
        min_duals: float = -1000.0,
        max_duals: float = 1000.0,
        min_barrier_coefs: float = 0.1,
        max_barrier_coefs: float = 1000.0,
        error_update_rate: float = 0.05,
        device=None,
    ):
        super().__init__()
        self.d = d
        self.coefficient_vector = torch.ones(d, device=device)
        self.device = (
            device
            if device is not None
            else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # Augmented Lagrangian parameters
        self.lr_duals = lr_duals
        self.lr_barrier_coefs = lr_barrier_coefs
        self.min_duals = min_duals
        self.max_duals = max_duals
        self.min_barrier_coefs = min_barrier_coefs
        self.max_barrier_coefs = max_barrier_coefs
        self.error_update_rate = error_update_rate

        # Initialize error estimates
        self.error_estimates = torch.zeros((d, d), device=self.device)
        self.quadratic_errors_estimates = torch.zeros((d, d), device=self.device)

        # For metrics tracking
        self.metrics = {}

    def compute_graph_drawing_loss(
        self, start_representation: torch.Tensor, end_representation: torch.Tensor
    ):
        """Compute the graph drawing loss as the weighted distance between start and end representations."""
        # Compute mean squared differences per dimension
        graph_induced_norms = ((start_representation - end_representation) ** 2).mean(0)
        # Dot with the coefficient vector to weight the contributions
        loss = torch.dot(graph_induced_norms, self.coefficient_vector)

        # Store metrics for later retrieval
        for i in range(self.d):
            self.metrics[f'graph_norm({i})'] = graph_induced_norms[i].item()

        return loss, graph_induced_norms

    def compute_orthogonality_error_matrix(
        self, representation_batch_1: torch.Tensor, representation_batch_2: torch.Tensor
    ):
        """Compute error matrices for orthonormality constraints."""
        n = representation_batch_1.shape[0]

        # Compute inner product matrices - use detach() for one side to match JAX implementation
        inner_product_matrix_1 = (
            torch.mm(representation_batch_1.t(), representation_batch_1.detach()) / n
        )
        inner_product_matrix_2 = (
            torch.mm(representation_batch_2.t(), representation_batch_2.detach()) / n
        )

        # Compute error vs identity matrix (for orthonormality)
        eye = torch.eye(self.d, device=representation_batch_1.device)
        error_matrix_1 = torch.tril(inner_product_matrix_1 - eye)
        error_matrix_2 = torch.tril(inner_product_matrix_2 - eye)

        # Average the errors from both representations
        error_matrix = 0.5 * (error_matrix_1 + error_matrix_2)

        # Compute quadratic error matrix (element-wise multiplication)
        quadratic_error_matrix = error_matrix**2

        # Return dictionary of error matrices
        error_matrix_dict = {
            'errors': error_matrix,
            'quadratic_errors': quadratic_error_matrix,
        }

        # Store metrics
        self.metrics['max_error'] = torch.abs(error_matrix).max().item()
        self.metrics['mean_error'] = torch.abs(error_matrix).mean().item()

        return error_matrix_dict

    def compute_orthogonality_loss(self, duals, barrier_coef, error_matrix):
        """Compute dual and barrier losses."""
        # Compute dual loss (using the current dual variables)
        dual_loss = (duals * error_matrix['errors']).sum()

        # Compute barrier loss (quadratic penalty)
        barrier_loss = barrier_coef * error_matrix['quadratic_errors'].sum()

        return dual_loss, barrier_loss

    def update_error_estimates(self, error_matrix_dict):
        """Update running estimates of errors."""
        # Update error estimates with exponential moving average
        self.error_estimates = (
            1 - self.error_update_rate
        ) * self.error_estimates + self.error_update_rate * error_matrix_dict['errors']

        # Update quadratic error estimates
        self.quadratic_errors_estimates = (
            (1 - self.error_update_rate) * self.quadratic_errors_estimates
            + self.error_update_rate * error_matrix_dict['quadratic_errors']
        )

        return {
            'errors': self.error_estimates,
            'quadratic_errors': self.quadratic_errors_estimates,
        }

    def update_barrier_coefficient(self, current_barrier_coef):
        """Update barrier coefficient based on current errors."""
        # Get the mean of the quadratic error estimates
        # - Large errors mean we should increase the barrier coefficient
        # - Small errors mean we can potentially reduce it
        quadratic_error = self.quadratic_errors_estimates.mean()

        # Compute update (increase when errors are large)
        update = self.lr_barrier_coefs * quadratic_error

        # Update barrier coefficient
        updated_barrier = current_barrier_coef + update

        # Clip to valid range
        updated_barrier = torch.clamp(
            updated_barrier, min=self.min_barrier_coefs, max=self.max_barrier_coefs
        )

        # Log change
        self.metrics['barrier_coef_update'] = update.item()

        return updated_barrier

    def update_duals(self, current_duals, dual_velocities, barrier_coef=None):
        """Update dual variables (Lagrange multipliers) based on current errors."""
        # Get error matrix from estimates
        error_matrix = self.error_estimates

        # Use adaptive learning rate scaled by barrier coefficient if provided
        lr = self.lr_duals
        if barrier_coef is not None:
            # Use a more conservative scaling
            lr = lr * torch.sqrt(torch.clamp(barrier_coef, min=1.0))

        global_step = getattr(self, '_global_step', 0)
        damping = max(0.5, 1.0 - global_step / 50000)  # Decrease from 1.0 to 0.5

        # Compute updates using momentum-based approach for smoother convergence
        # The update is based on the error estimates
        updates = error_matrix * lr * damping
        updated_duals = current_duals + updates

        # Clip duals to valid range
        updated_duals = torch.clamp(
            updated_duals, min=self.min_duals, max=self.max_duals
        )

        # Ensure lower triangular structure is maintained
        updated_duals = torch.tril(updated_duals)

        # Update velocities with momentum for more stable convergence
        lr_vel = self.lr_duals * 0.1  # Lower rate for velocities
        updates_diff = updated_duals - current_duals
        updated_velocities = dual_velocities + lr_vel * (updates_diff - dual_velocities)

        # Log change
        self.metrics['dual_update_mean'] = updates.abs().mean().item()

        return updated_duals, updated_velocities

    def forward(
        self,
        start_representation: torch.Tensor,
        end_representation: torch.Tensor,
        constraint_representation_1: torch.Tensor,
        constraint_representation_2: torch.Tensor,
        duals: torch.Tensor = None,
        barrier_coef: torch.Tensor = None,
        **kwargs,
    ):
        """Forward pass for computing the augmented Lagrangian loss."""
        # Reset metrics dictionary
        self.metrics = {}

        # Ensure we have dual parameters and barrier coefficient
        if duals is None or barrier_coef is None:
            raise ValueError('Dual parameters and barrier coefficient must be provided')

        # Compute graph drawing loss
        graph_loss, graph_norms = self.compute_graph_drawing_loss(
            start_representation, end_representation
        )

        # Compute orthogonality error matrices
        error_matrix_dict = self.compute_orthogonality_error_matrix(
            constraint_representation_1, constraint_representation_2
        )

        # Update error estimates (EMA)
        self.update_error_estimates(error_matrix_dict)

        # Compute orthogonality loss terms
        dual_loss, barrier_loss = self.compute_orthogonality_loss(
            duals, barrier_coef, error_matrix_dict
        )

        # Compute total loss
        total_loss = graph_loss + dual_loss + barrier_loss

        # Store high-level metrics
        self.metrics['total_loss'] = total_loss.item()
        self.metrics['graph_loss'] = graph_loss.item()
        self.metrics['dual_loss'] = dual_loss.item()
        self.metrics['barrier_loss'] = barrier_loss.item()
        self.metrics['barrier_coef'] = barrier_coef.item()

        # Calculate constraint violation
        constraint_violation = torch.abs(error_matrix_dict['errors']).mean().item()
        self.metrics['constraint_violation'] = constraint_violation

        return total_loss
