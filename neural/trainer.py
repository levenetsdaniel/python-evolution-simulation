import numpy as np

from core.training_buffer import TrainingBuffer
from neural.advisor import CatBoostAdvisor


def train_advisor(
        buffer: TrainingBuffer,
        advisor: CatBoostAdvisor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Train the advisor on buffer samples and return the training arrays.

    The target genome is reconstructed from the pre-mutation genome
    and the recorded mutation deltas.
    """
    if len(buffer.samples) == 0:
        raise ValueError("Can't train advisor: training buffer is empty.")

    X, mutation_deltas, weights = buffer.to_numpy()

    genome_size = len(buffer.samples[0].pre_mutation_genome)
    pre_genomes = X[:, :genome_size]

    target_genomes = np.clip(pre_genomes + mutation_deltas, 0.0, 1.0)

    advisor.fit(X, target_genomes, weights)

    return X, target_genomes, weights