"""
Verification Agent for functional equivalence testing between Verilog and SystemC.

This agent runs simulations on both implementations using the same test stimulus
and compares outputs to verify functional equivalence.
"""

import copy
import logging
import json
from typing import Any, Dict, Optional

from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.code.agents.testbench_agent import extract_verilog_ports, generate_test_stimulus

logger = logging.getLogger(__name__)


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class VerificationAgent(Agent):
    """
    Agent that performs functional equivalence verification between Verilog and SystemC.
    
    This agent:
    1. Extracts port information from the generated Verilog code
    2. Uses generated or auto-generated test stimulus
    3. Runs both Verilog and SystemC simulations
    4. Compares outputs to determine functional equivalence
    5. Provides feedback for the next iteration if not equivalent
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """
        Initialize the Verification Agent.
        """
        super().__init__()
        self.rollout_idx = rollout_idx
        self.if_trained = False  # This agent doesn't need LLM training
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def reset(self):
        super().reset()
        self.verification_result = None

    def update_from_env(self, turn_idx: int, env_data: Env):
        """
        Check if both codes are ready for verification.
        """
        self.env_data = env_data
        state = getattr(env_data, "state", None)
        
        verilog_code = getattr(state, "generated_verilog_code", None)
        systemc_code = getattr(state, "generated_systemc_code", None)
        
        # Skip if codes are not ready
        if not verilog_code or not systemc_code:
            self.skip_current_turn = True
            self.current_prompt = {"text": "", "image": None}
            return
        
        # Check if codes seem valid (not error messages)
        if verilog_code.startswith("We can not extract") or systemc_code.startswith("We can not extract"):
            self.skip_current_turn = True
            self.current_prompt = {"text": "", "image": None}
            return
        
        self.skip_current_turn = False
        
        # Prepare verification summary prompt (for logging purposes)
        formatted_prompt = (
            f"Verification Agent: Ready to verify functional equivalence.\n\n"
            f"Verilog code length: {len(verilog_code)} chars\n"
            f"SystemC code length: {len(systemc_code)} chars\n"
            f"Turn: {turn_idx}\n"
        )
        
        self.current_prompt = {"text": formatted_prompt, "image": None}

    def update_from_model(self, response: str):
        """
        This agent doesn't use LLM, so this is a no-op.
        """
        # No LLM response to process
        self.current_action = "verify"
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        Execute functional equivalence verification.
        """
        state = env_data.state
        
        verilog_code = state.generated_verilog_code
        systemc_code = state.generated_systemc_code
        
        # Skip if no env_worker or codes not ready
        if env_worker is None:
            logger.warning("VerificationAgent: No env_worker provided, skipping verification")
            return
        
        if not verilog_code or not systemc_code:
            logger.warning("VerificationAgent: Codes not ready, skipping verification")
            return
        
        # Extract ports from Verilog code
        try:
            ports = extract_verilog_ports(verilog_code)
            state.extracted_ports = ports
        except Exception as e:
            logger.warning(f"Failed to extract ports: {e}")
            ports = {"inputs": [], "outputs": [], "clock_ports": [], "reset_ports": []}
            state.extracted_ports = ports
        
        # Get or generate test stimulus
        test_stimulus = None
        
        # Try to use generated testbench
        if state.generated_testbench:
            try:
                test_stimulus = json.loads(state.generated_testbench)
            except json.JSONDecodeError:
                pass
        
        # Auto-generate if not available
        if not test_stimulus:
            test_stimulus = generate_test_stimulus(
                ports,
                num_vectors=50,
                num_sequences=5,
                sequence_length=30
            )
            state.generated_testbench = json.dumps(test_stimulus, indent=2)
        
        # Run functional equivalence verification
        try:
            from pettingllms.multi_agent_env.code.code_worker import _await_ray_object_ref
            
            obj_ref = env_worker.verify_equivalence.remote(
                verilog_code=verilog_code,
                systemc_code=systemc_code,
                ports=ports,
                test_stimulus=test_stimulus,
                timeout=120.0
            )
            
            result = await _await_ray_object_ref(obj_ref, timeout_seconds=150.0)
            
            # Store results
            state.verification_result = result
            state.equivalence_verified = True
            state.is_equivalent = result.get("equivalent", False)
            state.match_ratio = result.get("match_ratio", 0.0)
            state.verification_details = result.get("details", "")
            
            self.verification_result = result
            
            if state.is_equivalent:
                logger.info(f"Functional equivalence VERIFIED: {state.verification_details}")
                self.success = True
                env_data.success = True
            else:
                logger.info(f"Functional equivalence NOT verified: {state.verification_details}")
                self.success = False
                # Don't set env_data.success = False here, let other agents continue
                
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            state.equivalence_verified = False
            state.verification_details = f"Verification error: {e}"
            self.success = False

    def calculate_reward(self, env_data: Env):
        """
        Calculate reward based on verification results.
        
        Reward scheme:
        - Full match (equivalent): 2.0
        - Partial match: match_ratio * 1.5
        - Both codes exist but verification failed: 0.3
        - Codes missing: 0.0
        """
        state = env_data.state
        
        if state.is_equivalent:
            self.agent_reward = 2.0
        elif state.equivalence_verified:
            self.agent_reward = state.match_ratio * 1.5
        elif state.generated_verilog_code and state.generated_systemc_code:
            self.agent_reward = 0.3
        else:
            self.agent_reward = 0.0

    def reset(self):
        """
        Reset the agent's internal state for a new episode.
        """
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None
        self.verification_result = None
        self.skip_current_turn = False

