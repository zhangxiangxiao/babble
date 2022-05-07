"""Evaluator for babble model."""

import jax.numpy as jnp
from xjax.xeval import EvaluatorTuple, CategoricalEval

def Evaluator():
    """Categorical classification evaluator for babble."""
    eval_func, initial_states = CategoricalEval()
    def evaluate(inputs, net_outputs, states):
        targets = jnp.transpose(inputs[0], (1, 0))
        tar_outputs = jnp.transpose(net_outputs[0], (1, 0))
        dec_outputs = jnp.transpose(net_outputs[1], (1, 0))
        tar_result, states = eval_func((None, targets), tar_outputs, states)
        dec_result, states = eval_func((None, targets), dec_outputs, states)
        return (tar_result, dec_result), states
    return EvaluatorTuple(evaluate, initial_states)
