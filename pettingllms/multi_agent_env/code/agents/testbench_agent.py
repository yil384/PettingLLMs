import copy
import logging
import re
import json
import random
import math
from typing import Any, List, Dict, Tuple, Optional
from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env

logger = logging.getLogger(__name__)


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


def extract_verilog_ports(verilog_code: str) -> Dict[str, Any]:
    """
    Extract port information from Verilog code.
    Returns: {
        'inputs': [(port_name, width), ...],
        'outputs': [(port_name, width), ...],
        'clock_ports': [(port_name, edge), ...],  # edge: 'posedge' or 'negedge'
        'reset_ports': [(port_name, polarity, sync), ...]  # polarity: 0 or 1, sync: True/False
    }
    """
    # Remove comments
    note_pattern = r"(//[^\n]*|/\*[\s\S]*?\*/)"
    clean_code = re.sub(note_pattern, "", verilog_code)
    clean_code = re.sub(r"(?:\s*?\n)+", "\n", clean_code)
    
    ports = {
        'inputs': [],
        'outputs': [],
        'clock_ports': [],
        'reset_ports': []
    }
    
    # Find module definition
    module_pattern = r"module\s+([a-zA-Z_][a-zA-Z0-9_$]*)\s*(?:#\s*\([^)]*\))?\s*\(([\s\S]*?)\)\s*;"
    module_match = re.search(module_pattern, clean_code, re.DOTALL)
    
    if not module_match:
        return ports
    
    port_list = module_match.group(2)
    
    # Extract port declarations
    # Pattern for: input/output/inout [width] port_name
    port_decl_pattern = r"(input|output|inout)\s*(?:wire|reg)?\s*(?:\[(\d+):(\d+)\])?\s*([a-zA-Z_][a-zA-Z0-9_$]*)"
    
    for match in re.finditer(port_decl_pattern, port_list):
        direction = match.group(1)
        msb = match.group(2)
        lsb = match.group(3)
        port_name = match.group(4)
        
        # Calculate width
        if msb and lsb:
            width = abs(int(msb) - int(lsb)) + 1
        else:
            width = 1
        
        if direction == 'input':
            ports['inputs'].append((port_name, width))
        elif direction == 'output':
            ports['outputs'].append((port_name, width))
    
    # Detect clock ports (common names: clk, clock, clk_*, etc.)
    clock_patterns = [r'\b(clk|clock|clk_[a-zA-Z0-9_$]*)\b', r'\b([a-zA-Z_][a-zA-Z0-9_$]*clk[a-zA-Z0-9_$]*)\b']
    for port_name, _ in ports['inputs']:
        for pattern in clock_patterns:
            if re.search(pattern, port_name, re.IGNORECASE):
                # Try to detect edge from always blocks
                edge = 'posedge'  # default
                always_pattern = rf"always\s+@\s*\(\s*(?:posedge|negedge)\s+{re.escape(port_name)}"
                if re.search(always_pattern, clean_code, re.IGNORECASE):
                    if re.search(rf"negedge\s+{re.escape(port_name)}", clean_code, re.IGNORECASE):
                        edge = 'negedge'
                ports['clock_ports'].append((port_name, edge))
                break
    
    # Detect reset ports (common names: rst, reset, rst_*, etc.)
    reset_patterns = [r'\b(rst|reset|rst_[a-zA-Z0-9_$]*|areset|nreset)\b', 
                     r'\b([a-zA-Z_][a-zA-Z0-9_$]*rst[a-zA-Z0-9_$]*)\b']
    for port_name, _ in ports['inputs']:
        if port_name in [p[0] for p in ports['clock_ports']]:
            continue  # Skip if it's a clock
        for pattern in reset_patterns:
            if re.search(pattern, port_name, re.IGNORECASE):
                # Default: active high, synchronous
                polarity = 1  # active high
                sync = True
                # Try to detect from code
                if 'n' in port_name.lower() or 'neg' in port_name.lower():
                    polarity = 0  # active low
                ports['reset_ports'].append((port_name, polarity, sync))
                break
    
    return ports


def generate_test_stimulus(
    ports: Dict[str, Any],
    num_vectors: int = 100,
    num_sequences: int = 10,
    sequence_length: int = 50
) -> Dict[str, Any]:
    """
    Generate test stimulus based on port information.
    Returns test vectors (for combinational) or test scenarios (for sequential).
    """
    inputs = ports.get('inputs', [])
    outputs = ports.get('outputs', [])
    clock_ports = ports.get('clock_ports', [])
    reset_ports = ports.get('reset_ports', [])
    
    # Filter out clock and reset from regular inputs
    clock_names = {p[0] for p in clock_ports}
    reset_names = {p[0] for p in reset_ports}
    regular_inputs = [(name, width) for name, width in inputs 
                     if name not in clock_names and name not in reset_names]
    
    stimulus = {
        'type': 'sequential' if clock_ports else 'combinational',
        'test_vectors': [],
        'test_scenarios': []
    }
    
    if clock_ports:
        # Sequential circuit: generate test scenarios
        for seq_idx in range(num_sequences):
            clock_name = clock_ports[0][0]
            clock_edge = clock_ports[0][1]
            
            scenario = {
                'clock_cycles': sequence_length,
                clock_name: ['0', '1'] * (sequence_length // 2) + (['0'] if sequence_length % 2 else [])
            }
            
            # Add reset sequences
            for reset_name, polarity, _ in reset_ports:
                reset_seq = [str(polarity)] + [str(1 - polarity)] * (sequence_length - 1)
                scenario[reset_name] = reset_seq
            
            # Add random inputs for each cycle
            for port_name, width in regular_inputs:
                port_seq = []
                for cycle in range(sequence_length):
                    if width <= 32:
                        val = random.randint(0, (1 << width) - 1)
                        port_seq.append(format(val, f'0{width}b'))
                    else:
                        # For wide signals, generate multiple random values
                        num_words = math.ceil(width / 32)
                        val_parts = [random.randint(0, (1 << 32) - 1) for _ in range(num_words)]
                        # Combine into binary string
                        full_val = 0
                        for i, part in enumerate(val_parts):
                            full_val |= (part << (i * 32))
                        # Truncate to width
                        mask = (1 << width) - 1
                        full_val &= mask
                        port_seq.append(format(full_val, f'0{width}b'))
                scenario[port_name] = port_seq
            
            stimulus['test_scenarios'].append(scenario)
    else:
        # Combinational circuit: generate test vectors
        for vec_idx in range(num_vectors):
            vector = {}
            
            for port_name, width in regular_inputs:
                if width <= 32:
                    val = random.randint(0, (1 << width) - 1)
                    vector[port_name] = format(val, f'0{width}b')
                else:
                    # For wide signals
                    num_words = math.ceil(width / 32)
                    val_parts = [random.randint(0, (1 << 32) - 1) for _ in range(num_words)]
                    full_val = 0
                    for i, part in enumerate(val_parts):
                        full_val |= (part << (i * 32))
                    mask = (1 << width) - 1
                    full_val &= mask
                    vector[port_name] = format(full_val, f'0{width}b')
            
            stimulus['test_vectors'].append(vector)
    
    return stimulus


class TestbenchGenerationAgent(Agent):
    """
    Agent specialized for generating testbench/test stimulus for hardware designs.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """
        Initialize the Testbench Generation Agent.
        """
        super().__init__()
        self.rollout_idx = rollout_idx
        # Accept other unrelated keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def reset(self):
        super().reset()

    def update_from_env(self, turn_idx: int, env_data: Env):
        """
        Update the agent's internal prompt after an environment step.
        """
        # Save environment data
        self.env_data = env_data

        state = getattr(env_data, "state", None)

        question = getattr(state, "problem", None)
        verilog_code = getattr(state, "generated_verilog_code", None)
        systemc_code = getattr(state, "generated_systemc_code", None)
        
        # Try to extract port information from Verilog code
        port_info = None
        if verilog_code:
            try:
                port_info = extract_verilog_ports(verilog_code)
            except Exception as e:
                logger.warning(f"Failed to extract ports from Verilog: {e}")
        
        formatted_prompt = ""
        
        if turn_idx == 0:
            # Initial testbench generation
            if verilog_code:
                formatted_prompt = (
                    f"You are a helpful assistant that generates testbench and test stimulus for hardware designs.\n\n"
                    f"Given a Verilog design, your task is to generate comprehensive test stimulus.\n\n"
                    f"Problem Description:\n{question}\n\n"
                    f"Verilog Code:\n```verilog\n{truncatefn(verilog_code, 2000)}\n```\n\n"
                )
                
                if port_info and (port_info.get('inputs') or port_info.get('outputs')):
                    formatted_prompt += (
                        f"Extracted Port Information:\n"
                        f"- Input ports: {[(p[0], p[1]) for p in port_info.get('inputs', [])]}\n"
                        f"- Output ports: {[(p[0], p[1]) for p in port_info.get('outputs', [])]}\n"
                    )
                    if port_info.get('clock_ports'):
                        formatted_prompt += f"- Clock ports: {port_info.get('clock_ports')}\n"
                    if port_info.get('reset_ports'):
                        formatted_prompt += f"- Reset ports: {port_info.get('reset_ports')}\n"
                    formatted_prompt += "\n"
                
                formatted_prompt += (
                    f"Please generate test stimulus for this design. The stimulus should:\n"
                    f"1. Cover all input ports with appropriate test vectors\n"
                    f"2. Include edge cases and corner cases\n"
                    f"3. Include random test patterns for comprehensive coverage\n"
                    f"4. For sequential circuits, include reset sequences and clock cycles\n\n"
                    f"Respond with a JSON object containing the test stimulus in this format:\n\n"
                    f"For combinational circuits:\n"
                    f"```json\n"
                    f'{{\n'
                    f'  "type": "combinational",\n'
                    f'  "test_vectors": [\n'
                    f'    {{"port_name": "binary_value", ...}},\n'
                    f'    ...\n'
                    f'  ]\n'
                    f'}}\n'
                    f'```\n\n'
                    f"For sequential circuits:\n"
                    f"```json\n"
                    f'{{\n'
                    f'  "type": "sequential",\n'
                    f'  "test_scenarios": [\n'
                    f'    {{\n'
                    f'      "clock_cycles": 50,\n'
                    f'      "clock_port": ["0", "1", ...],\n'
                    f'      "reset_port": ["1", "0", ...],\n'
                    f'      "input_port": ["binary", ...],\n'
                    f'      ...\n'
                    f'    }},\n'
                    f'    ...\n'
                    f'  ]\n'
                    f'}}\n'
                    f'```\n\n'
                )
            else:
                formatted_prompt = (
                    f"You are a helpful assistant that generates testbench for hardware designs.\n\n"
                    f"Problem Description:\n{question}\n\n"
                    f"Note: Verilog code has not been generated yet. Please wait for the Verilog code to be generated first.\n"
                    f"Once the Verilog code is available, you can generate test stimulus based on the design.\n"
                )
        else:
            # Refinement mode
            formatted_prompt = (
                f"You are a helpful assistant that refines testbench for hardware designs.\n\n"
                f"Problem Description:\n{question}\n\n"
            )
            
            if verilog_code:
                formatted_prompt += (
                    f"Verilog Code:\n```verilog\n{truncatefn(verilog_code, 2000)}\n```\n\n"
                )
            
            if systemc_code:
                formatted_prompt += (
                    f"SystemC Code (for reference):\n```systemc\n{truncatefn(systemc_code, 1000)}\n```\n\n"
                )
            
            formatted_prompt += (
                f"Please refine the test stimulus to ensure comprehensive coverage.\n"
                f"Consider the SystemC implementation when generating test cases.\n\n"
                f"Respond with the refined test stimulus in JSON format as described above.\n"
            )

        self.current_prompt = {"text": formatted_prompt, "image": None}

    def update_from_model(self, response: str):
        # Parse the response and extract test stimulus
        import re
        
        # Try to extract JSON from code blocks
        json_pattern = r'```json\s*(.*?)```'
        json_matches = re.findall(json_pattern, response, re.DOTALL)
        
        if json_matches:
            try:
                stimulus = json.loads(json_matches[-1].strip())
                self.current_action = stimulus
                return stimulus
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from response: {e}")
        
        # Try to extract JSON without code blocks
        json_pattern2 = r'\{[\s\S]*"type"[\s\S]*\}'
        json_matches2 = re.findall(json_pattern2, response, re.DOTALL)
        if json_matches2:
            try:
                stimulus = json.loads(json_matches2[-1].strip())
                self.current_action = stimulus
                return stimulus
            except json.JSONDecodeError:
                pass
        
        # If no valid JSON found, try to auto-generate from Verilog code
        if hasattr(self, 'env_data') and self.env_data:
            state = getattr(self.env_data, "state", None)
            verilog_code = getattr(state, "generated_verilog_code", None) if state else None
            
            if verilog_code:
                try:
                    ports = extract_verilog_ports(verilog_code)
                    stimulus = generate_test_stimulus(ports, num_vectors=100, num_sequences=10)
                    self.current_action = stimulus
                    logger.info("Auto-generated test stimulus from Verilog ports")
                    return stimulus
                except Exception as e:
                    logger.warning(f"Failed to auto-generate stimulus: {e}")
        
        # Fallback: return empty stimulus
        self.current_action = {"type": "combinational", "test_vectors": []}
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        Update the state with generated test stimulus.
        """
        # Parse and update generated test stimulus
        stimulus = self.current_action
        
        if stimulus:
            env_data.state.generated_testbench = json.dumps(stimulus, indent=2)
            
            # Also store in a more accessible format
            if stimulus.get('type') == 'combinational':
                test_vectors = stimulus.get('test_vectors', [])
                if test_vectors:
                    # Convert to test_input/test_output format for compatibility
                    env_data.state.test_input = [json.dumps(vec) for vec in test_vectors]
                    # Note: expected outputs would need to be computed by simulation
                    env_data.state.test_output = []
            elif stimulus.get('type') == 'sequential':
                test_scenarios = stimulus.get('test_scenarios', [])
                if test_scenarios:
                    # Store scenarios as JSON strings
                    env_data.state.test_input = [json.dumps(scenario) for scenario in test_scenarios]
                    env_data.state.test_output = []
        
        # Success is determined by having valid test stimulus
        if stimulus and (
            stimulus.get('test_vectors') or 
            stimulus.get('test_scenarios')
        ):
            self.success = True
            env_data.success = True
        else:
            self.success = False
            env_data.success = False

    def calculate_reward(self, env_data: Env):
        # Reward based on test stimulus quality
        stimulus_quality = 0.0
        
        if env_data.state.generated_testbench:
            try:
                stimulus = json.loads(env_data.state.generated_testbench)
                if stimulus.get('type') == 'combinational':
                    num_vectors = len(stimulus.get('test_vectors', []))
                    stimulus_quality = min(1.0, num_vectors / 100.0)  # Normalize to 1.0
                elif stimulus.get('type') == 'sequential':
                    num_scenarios = len(stimulus.get('test_scenarios', []))
                    stimulus_quality = min(1.0, num_scenarios / 10.0)  # Normalize to 1.0
            except:
                pass
        
        self.agent_reward = stimulus_quality

    def reset(self):
        """
        Reset the agent's internal state for a new episode.
        """
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None

