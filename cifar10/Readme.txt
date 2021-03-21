Libraries/Dependencies required: Same as DARTS

1] Architecture Search: python arch_search.py

Copy the resulting architecture into genotypes.py
----------------------------------------------------------------------------------------------------------------------------
eg.) 
RESULT = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('dil_conv_5x5', 1), ('skip_connect', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

DARTS = RESULT
----------------------------------------------------------------------------------------------------------------------------
2] Architecture Evaluation: python train.py