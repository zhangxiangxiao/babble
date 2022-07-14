"""Evaluator for babble model."""

import jax.numpy as jnp
from xjax.xeval import *

def Evaluator():
    """Categorical classification evaluator for babble."""
    eval_func, initial_states = CategoricalEval()
    def evaluate(inputs, net_outputs, states):
        targets = jnp.transpose(inputs[0], (1, 0))
        dec_outputs = jnp.transpose(net_outputs[0], (1, 0))
        dec_result, states = eval_func((None, targets), dec_outputs, states)
        return dec_result, states
    return vmap(EvaluatorTuple(evaluate, initial_states))
