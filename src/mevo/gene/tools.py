import typing as tp

from mevo.gene.gene_type import GeneShape


def convert_gene_shape(shape: GeneShape) -> tp.Sequence[int]:
    if isinstance(shape, (int, float)):
        shape = (1, int(shape))
    elif len(shape) == 1:
        shape = (1, shape[0])
    return shape
