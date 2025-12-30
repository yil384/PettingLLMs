import logging
import copy
import io
import sys
import time
import typing
import multiprocessing as mp
from typing import Any, Dict, Optional, Tuple, List, Union

from pettingllms.multi_agent_env.code.agents.code_v_agent import VerilogGenerationAgent
from pettingllms.multi_agent_env.code.agents.code_c_agent import SystemCGenerationAgent
from pettingllms.multi_agent_env.code.agents.testbench_agent import TestbenchGenerationAgent
from pettingllms.multi_agent_env.code.agents.verification_agent import VerificationAgent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.code.code_utils import (
        load_problem_batch,
        extract_code_from_response,
        evaluate_code_against_tests,
    )
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class CodeEnvState:
    problem: str=None
    golden_verilog_code: str=None
    golden_systemc_code: str=None
    generated_verilog_code_history: List[str]=field(default_factory=list)
    generated_systemc_code_history: List[str]=field(default_factory=list)
    generated_verilog_code: str=None
    generated_systemc_code: str=None
    both_codes_generated: bool=False
    # Test stimulus (testbench) - can come from dataset or be generated
    test_input: List[str]=None  # Test input vectors/stimulus for hardware verification
    test_output: List[str]=None  # Expected test outputs for hardware verification
    generated_testbench: str=None  # Generated testbench code (if any agent generates it)
    # Port information extracted from Verilog code
    extracted_ports: Dict=field(default_factory=dict)
    # Functional equivalence verification results
    verification_result: Dict=field(default_factory=dict)  # Result from verify_functional_equivalence
    equivalence_verified: bool=False  # True if verification was run
    is_equivalent: bool=False  # True if Verilog and SystemC are functionally equivalent
    match_ratio: float=0.0  # Ratio of matching outputs (0.0 to 1.0)
    verification_details: str=""  # Human-readable verification details
    # For backward compatibility, keep some old fields but mark as deprecated
    generated_code_history: List[str]=field(default_factory=list)  # Deprecated
    generated_code: str=None  # Deprecated

class CodeEnv(Env):
    """
    Environment for hardware code generation tasks with dual-agent interaction.
    
    This environment coordinates between Verilog code generation and SystemC code generation agents,
    where both agents generate hardware design code for the same problem.
    """

    def __init__(
        self, 
        env_idx: int,
        rollout_idx: int,
        max_turns: int,
        config: dict | None = None,
    ):
        """
        Initialize the code test environment.

    
        """
        super().__init__(env_idx=env_idx, rollout_idx=rollout_idx, config=config)
        self.state=CodeEnvState()
        
        self.backend = "ray_docker" 


   

    def reset(self):
        self.state.generated_verilog_code=None
        self.state.generated_systemc_code=None
        self.state.generated_verilog_code_history=[]
        self.state.generated_systemc_code_history=[]
        self.state.both_codes_generated=False
        self.state.generated_testbench=None
        # Reset port info and verification state
        self.state.extracted_ports={}
        self.state.verification_result={}
        self.state.equivalence_verified=False
        self.state.is_equivalent=False
        self.state.match_ratio=0.0
        self.state.verification_details=""
        # Note: test_input and test_output are kept from problem loading
        # Deprecated fields for backward compatibility
        self.state.generated_code=None
        self.state.generated_code_history=[]


class CodeEnvBatch:
    def __init__(self, env_idx_list: List[int], env_indices: List[int], rollout_idx_list: List[int], samples: int, max_turns: int, config: dict, mode="train", *, env_workers: List=None):
        # Convert env_indices to list for safety
        safe_env_indices = list(env_indices) if not isinstance(env_indices, list) else env_indices
        
        if mode=="train":
            dataset_name = getattr(config.env, "dataset", "apps") if hasattr(config, "env") and hasattr(config.env, "dataset") else "apps"
            self.problem_list=load_problem_batch(safe_env_indices, dataset_name=dataset_name, mode="train")
        else:
            benchmark_name=getattr(config.env,"benchmark") if hasattr(config,"env") and hasattr(config.env,"benchmark") else "test"
            #difficulty=getattr(config,"difficulty") if hasattr(config,"difficulty") else "difficult"
            self.problem_list=load_problem_batch(safe_env_indices,mode=mode,benchmark_name=benchmark_name)
            samples=1
        self.env_list=[]
        if mode=="validate":
            rollout_idx_list=range(len(self.problem_list)*samples)
   

        for i,problem in enumerate(self.problem_list):
            # Load test stimulus if available in dataset (for hardware verification)
            test_input = problem.get("test_input", None)
            test_output = problem.get("test_output", None)
            state=CodeEnvState(
                problem=problem["question"],
                test_input=test_input,
                test_output=test_output
            )
            for s in range(samples):
                env=CodeEnv(env_idx=i, rollout_idx=rollout_idx_list[i*samples+s], max_turns=max_turns, config=None)
                env.state=copy.deepcopy(state)
                self.env_list.append(env)
        if len(self.env_list)!=len(rollout_idx_list):
            raise ValueError(f"len(self.env_list)!=len(rollout_idx_list), {len(self.env_list)}!={len(rollout_idx_list)}")