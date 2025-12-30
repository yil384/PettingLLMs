"""
Worker classes and utilities for code execution.

This module contains the worker implementations for executing code
in isolated environments, including Ray-based distributed workers.

Supports:
- Python code execution
- Verilog simulation (via iverilog + vvp)
- SystemC compilation and execution (via g++ + SystemC library)
- Functional equivalence verification between Verilog and SystemC
"""

import os
import sys
import asyncio
import subprocess
import tempfile
import shutil
import textwrap
import signal
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


async def test_if_eq(x, y):
    """
    Test equality of two outputs ignoring whitespace differences.
    Based on the reference test_if_eq function provided.
    """
    if x is None or y is None:
        return False
    return " ".join(str(x).split()) == " ".join(str(y).split())


def compare_signal_outputs(verilog_outputs: List[Dict], systemc_outputs: List[Dict]) -> Tuple[bool, float, str]:
    """
    Compare outputs from Verilog and SystemC simulations.
    
    Args:
        verilog_outputs: List of output dictionaries from Verilog simulation
        systemc_outputs: List of output dictionaries from SystemC simulation
        
    Returns:
        (match, match_ratio, details_message)
    """
    if not verilog_outputs or not systemc_outputs:
        return False, 0.0, "Empty outputs"
    
    min_len = min(len(verilog_outputs), len(systemc_outputs))
    matches = 0
    mismatches = []
    
    for i in range(min_len):
        v_out = verilog_outputs[i]
        s_out = systemc_outputs[i]
        
        # Compare all output signals
        all_match = True
        for key in v_out.keys():
            if key in s_out:
                if str(v_out[key]).strip() != str(s_out[key]).strip():
                    all_match = False
                    mismatches.append(f"Cycle {i}: {key} - Verilog={v_out[key]}, SystemC={s_out[key]}")
        
        if all_match:
            matches += 1
    
    match_ratio = matches / min_len if min_len > 0 else 0.0
    
    if match_ratio == 1.0:
        details = f"All {min_len} cycles matched"
    else:
        details = f"Match ratio: {matches}/{min_len} ({match_ratio*100:.1f}%)"
        if mismatches:
            details += f"\nFirst mismatches: {'; '.join(mismatches[:5])}"
    
    return match_ratio == 1.0, match_ratio, details


# =============================================================================
# Verilog Simulation (iverilog + vvp)
# =============================================================================

def generate_verilog_testbench(
    module_name: str,
    ports: Dict[str, Any],
    test_stimulus: Dict[str, Any]
) -> str:
    """
    Generate a Verilog testbench from port info and test stimulus.
    
    Args:
        module_name: Name of the DUT module
        ports: Port information dict with 'inputs', 'outputs', 'clock_ports', 'reset_ports'
        test_stimulus: Test stimulus dict with 'type', 'test_vectors' or 'test_scenarios'
        
    Returns:
        Verilog testbench code string
    """
    inputs = ports.get('inputs', [])
    outputs = ports.get('outputs', [])
    clock_ports = ports.get('clock_ports', [])
    reset_ports = ports.get('reset_ports', [])
    
    tb_code = f"`timescale 1ns/1ps\n\n"
    tb_code += f"module tb_{module_name};\n\n"
    
    # Declare signals
    for port_name, width in inputs:
        tb_code += f"    reg [{width-1}:0] {port_name};\n" if width > 1 else f"    reg {port_name};\n"
    for port_name, width in outputs:
        tb_code += f"    wire [{width-1}:0] {port_name};\n" if width > 1 else f"    wire {port_name};\n"
    
    tb_code += f"\n    // DUT instantiation\n"
    tb_code += f"    {module_name} dut (\n"
    all_ports = [(p[0], 'input') for p in inputs] + [(p[0], 'output') for p in outputs]
    port_connections = [f"        .{p[0]}({p[0]})" for p in all_ports]
    tb_code += ",\n".join(port_connections)
    tb_code += "\n    );\n\n"
    
    # Clock generation
    if clock_ports:
        clock_name = clock_ports[0][0]
        tb_code += f"    // Clock generation\n"
        tb_code += f"    initial {clock_name} = 0;\n"
        tb_code += f"    always #5 {clock_name} = ~{clock_name};\n\n"
    
    # Test stimulus
    tb_code += f"    // Test stimulus\n"
    tb_code += f"    initial begin\n"
    tb_code += f'        $dumpfile("waveform.vcd");\n'
    tb_code += f'        $dumpvars(0, tb_{module_name});\n\n'
    
    if test_stimulus.get('type') == 'sequential':
        scenarios = test_stimulus.get('test_scenarios', [])
        if scenarios:
            scenario = scenarios[0]  # Use first scenario
            num_cycles = scenario.get('clock_cycles', 10)
            
            # Apply reset
            for reset_name, polarity, _ in reset_ports:
                tb_code += f"        {reset_name} = {polarity};\n"
            tb_code += f"        #20;\n"
            for reset_name, polarity, _ in reset_ports:
                tb_code += f"        {reset_name} = {1-polarity};\n"
            tb_code += f"        #10;\n\n"
            
            # Apply test vectors
            clock_names = {p[0] for p in clock_ports}
            reset_names = {p[0] for p in reset_ports}
            regular_inputs = [(n, w) for n, w in inputs if n not in clock_names and n not in reset_names]
            
            for cycle in range(min(num_cycles, 50)):  # Limit cycles
                for port_name, _ in regular_inputs:
                    if port_name in scenario:
                        values = scenario[port_name]
                        if cycle < len(values):
                            tb_code += f"        {port_name} = 'b{values[cycle]};\n"
                tb_code += f"        @(posedge {clock_ports[0][0]});\n"
                # Monitor outputs
                output_fmt = " ".join([f"{p[0]}=%b" for p in outputs])
                output_vars = ", ".join([p[0] for p in outputs])
                tb_code += f'        $display("CYCLE {cycle}: {output_fmt}", {output_vars});\n'
    else:
        # Combinational test vectors
        vectors = test_stimulus.get('test_vectors', [])
        clock_names = {p[0] for p in clock_ports}
        reset_names = {p[0] for p in reset_ports}
        regular_inputs = [(n, w) for n, w in inputs if n not in clock_names and n not in reset_names]
        
        for i, vector in enumerate(vectors[:50]):  # Limit vectors
            for port_name, _ in regular_inputs:
                if port_name in vector:
                    tb_code += f"        {port_name} = 'b{vector[port_name]};\n"
            tb_code += f"        #10;\n"
            output_fmt = " ".join([f"{p[0]}=%b" for p in outputs])
            output_vars = ", ".join([p[0] for p in outputs])
            tb_code += f'        $display("VECTOR {i}: {output_fmt}", {output_vars});\n'
    
    tb_code += f"\n        #100;\n"
    tb_code += f"        $finish;\n"
    tb_code += f"    end\n\n"
    tb_code += f"endmodule\n"
    
    return tb_code


async def run_verilog_simulation(
    verilog_code: str,
    testbench_code: str,
    timeout: float = 60.0
) -> Dict[str, Any]:
    """
    Run Verilog simulation using iverilog and vvp.
    
    Args:
        verilog_code: Verilog design code
        testbench_code: Verilog testbench code
        timeout: Simulation timeout in seconds
        
    Returns:
        Dict with 'success', 'outputs', 'error', 'raw_output'
    """
    tmpdir = tempfile.mkdtemp(prefix="verilog_sim_", dir="tmp")
    result = {
        "success": False,
        "outputs": [],
        "error": None,
        "raw_output": ""
    }
    
    try:
        os.makedirs("tmp", exist_ok=True)
        
        # Write files
        design_path = os.path.join(tmpdir, "design.v")
        tb_path = os.path.join(tmpdir, "testbench.v")
        
        with open(design_path, "w") as f:
            f.write(verilog_code)
        with open(tb_path, "w") as f:
            f.write(testbench_code)
        
        # Compile with iverilog
        compile_proc = await asyncio.create_subprocess_exec(
            "iverilog", "-o", "sim.vvp", "design.v", "testbench.v",
            cwd=tmpdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(compile_proc.communicate(), timeout=30.0)
            if compile_proc.returncode != 0:
                result["error"] = f"Compilation failed: {stderr.decode()}"
                return result
        except asyncio.TimeoutError:
            compile_proc.kill()
            result["error"] = "Compilation timeout"
            return result
        
        # Run simulation with vvp
        sim_proc = await asyncio.create_subprocess_exec(
            "vvp", "sim.vvp",
            cwd=tmpdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(sim_proc.communicate(), timeout=timeout)
            result["raw_output"] = stdout.decode()
            
            # Parse outputs from $display statements
            outputs = []
            for line in result["raw_output"].splitlines():
                if line.startswith("CYCLE") or line.startswith("VECTOR"):
                    # Parse: "CYCLE 0: out1=1010 out2=0011"
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        output_dict = {}
                        for item in parts[1].strip().split():
                            if "=" in item:
                                k, v = item.split("=", 1)
                                output_dict[k] = v
                        outputs.append(output_dict)
            
            result["outputs"] = outputs
            result["success"] = True
            
        except asyncio.TimeoutError:
            sim_proc.kill()
            result["error"] = "Simulation timeout"
            
    except FileNotFoundError:
        result["error"] = "iverilog not found. Please install: apt-get install iverilog"
    except Exception as e:
        result["error"] = str(e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    return result


# =============================================================================
# SystemC Simulation (g++ + SystemC library)
# =============================================================================

def generate_systemc_testbench(
    module_name: str,
    ports: Dict[str, Any],
    test_stimulus: Dict[str, Any]
) -> str:
    """
    Generate a SystemC testbench main.cpp from port info and test stimulus.
    
    Args:
        module_name: Name of the DUT module
        ports: Port information dict
        test_stimulus: Test stimulus dict
        
    Returns:
        SystemC testbench code string
    """
    inputs = ports.get('inputs', [])
    outputs = ports.get('outputs', [])
    clock_ports = ports.get('clock_ports', [])
    reset_ports = ports.get('reset_ports', [])
    
    tb_code = '#include <systemc.h>\n'
    tb_code += '#include "design.h"\n\n'
    tb_code += 'int sc_main(int argc, char* argv[]) {\n'
    
    # Declare signals
    for port_name, width in inputs:
        if width > 1:
            tb_code += f'    sc_signal<sc_bv<{width}>> {port_name};\n'
        else:
            tb_code += f'    sc_signal<bool> {port_name};\n'
    for port_name, width in outputs:
        if width > 1:
            tb_code += f'    sc_signal<sc_bv<{width}>> {port_name};\n'
        else:
            tb_code += f'    sc_signal<bool> {port_name};\n'
    
    # Clock generation
    if clock_ports:
        clock_name = clock_ports[0][0]
        tb_code += f'\n    sc_clock {clock_name}_gen("{clock_name}_gen", 10, SC_NS);\n'
        tb_code += f'    sc_signal<bool> {clock_name}_sig;\n'
    
    # DUT instantiation
    tb_code += f'\n    {module_name} dut("dut");\n'
    for port_name, _ in inputs:
        tb_code += f'    dut.{port_name}({port_name});\n'
    for port_name, _ in outputs:
        tb_code += f'    dut.{port_name}({port_name});\n'
    
    # Test stimulus
    tb_code += '\n    // Test stimulus\n'
    
    if test_stimulus.get('type') == 'sequential':
        scenarios = test_stimulus.get('test_scenarios', [])
        if scenarios:
            scenario = scenarios[0]
            num_cycles = min(scenario.get('clock_cycles', 10), 50)
            
            # Reset
            for reset_name, polarity, _ in reset_ports:
                tb_code += f'    {reset_name} = {polarity};\n'
            tb_code += '    sc_start(20, SC_NS);\n'
            for reset_name, polarity, _ in reset_ports:
                tb_code += f'    {reset_name} = {1-polarity};\n'
            
            clock_names = {p[0] for p in clock_ports}
            reset_names = {p[0] for p in reset_ports}
            regular_inputs = [(n, w) for n, w in inputs if n not in clock_names and n not in reset_names]
            
            for cycle in range(num_cycles):
                for port_name, width in regular_inputs:
                    if port_name in scenario:
                        values = scenario[port_name]
                        if cycle < len(values):
                            if width > 1:
                                tb_code += f'    {port_name} = "{values[cycle]}";\n'
                            else:
                                tb_code += f'    {port_name} = {values[cycle]};\n'
                tb_code += '    sc_start(10, SC_NS);\n'
                # Print outputs
                output_prints = []
                for port_name, width in outputs:
                    if width > 1:
                        output_prints.append(f'<< " {port_name}=" << {port_name}.read().to_string(SC_BIN)')
                    else:
                        output_prints.append(f'<< " {port_name}=" << {port_name}.read()')
                tb_code += f'    std::cout << "CYCLE {cycle}:" {" ".join(output_prints)} << std::endl;\n'
    else:
        vectors = test_stimulus.get('test_vectors', [])
        clock_names = {p[0] for p in clock_ports}
        reset_names = {p[0] for p in reset_ports}
        regular_inputs = [(n, w) for n, w in inputs if n not in clock_names and n not in reset_names]
        
        for i, vector in enumerate(vectors[:50]):
            for port_name, width in regular_inputs:
                if port_name in vector:
                    if width > 1:
                        tb_code += f'    {port_name} = "{vector[port_name]}";\n'
                    else:
                        tb_code += f'    {port_name} = {vector[port_name]};\n'
            tb_code += '    sc_start(10, SC_NS);\n'
            output_prints = []
            for port_name, width in outputs:
                if width > 1:
                    output_prints.append(f'<< " {port_name}=" << {port_name}.read().to_string(SC_BIN)')
                else:
                    output_prints.append(f'<< " {port_name}=" << {port_name}.read()')
            tb_code += f'    std::cout << "VECTOR {i}:" {" ".join(output_prints)} << std::endl;\n'
    
    tb_code += '\n    return 0;\n'
    tb_code += '}\n'
    
    return tb_code


async def run_systemc_simulation(
    systemc_code: str,
    testbench_code: str,
    timeout: float = 60.0
) -> Dict[str, Any]:
    """
    Run SystemC simulation by compiling and executing.
    
    Args:
        systemc_code: SystemC design code
        testbench_code: SystemC testbench main.cpp code
        timeout: Simulation timeout in seconds
        
    Returns:
        Dict with 'success', 'outputs', 'error', 'raw_output'
    """
    tmpdir = tempfile.mkdtemp(prefix="systemc_sim_", dir="tmp")
    result = {
        "success": False,
        "outputs": [],
        "error": None,
        "raw_output": ""
    }
    
    try:
        os.makedirs("tmp", exist_ok=True)
        
        # Write files
        design_path = os.path.join(tmpdir, "design.h")
        tb_path = os.path.join(tmpdir, "main.cpp")
        
        with open(design_path, "w") as f:
            f.write(systemc_code)
        with open(tb_path, "w") as f:
            f.write(testbench_code)
        
        # Check for SystemC installation
        systemc_home = os.environ.get("SYSTEMC_HOME", "/usr/local/systemc")
        
        # Compile with g++
        compile_cmd = [
            "g++", "-std=c++17", "-O2",
            f"-I{systemc_home}/include",
            f"-L{systemc_home}/lib-linux64",
            "-o", "sim",
            "main.cpp",
            "-lsystemc", "-lm", "-lpthread"
        ]
        
        compile_proc = await asyncio.create_subprocess_exec(
            *compile_cmd,
            cwd=tmpdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(compile_proc.communicate(), timeout=60.0)
            if compile_proc.returncode != 0:
                result["error"] = f"Compilation failed: {stderr.decode()}"
                return result
        except asyncio.TimeoutError:
            compile_proc.kill()
            result["error"] = "Compilation timeout"
            return result
        
        # Run simulation
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{systemc_home}/lib-linux64:" + env.get("LD_LIBRARY_PATH", "")
        
        sim_proc = await asyncio.create_subprocess_exec(
            "./sim",
            cwd=tmpdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(sim_proc.communicate(), timeout=timeout)
            result["raw_output"] = stdout.decode()
            
            # Parse outputs
            outputs = []
            for line in result["raw_output"].splitlines():
                if line.startswith("CYCLE") or line.startswith("VECTOR"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        output_dict = {}
                        for item in parts[1].strip().split():
                            if "=" in item:
                                k, v = item.split("=", 1)
                                output_dict[k] = v
                        outputs.append(output_dict)
            
            result["outputs"] = outputs
            result["success"] = True
            
        except asyncio.TimeoutError:
            sim_proc.kill()
            result["error"] = "Simulation timeout"
            
    except FileNotFoundError:
        result["error"] = "g++ not found or SystemC not installed"
    except Exception as e:
        result["error"] = str(e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    return result


# =============================================================================
# Functional Equivalence Verification
# =============================================================================

async def verify_functional_equivalence(
    verilog_code: str,
    systemc_code: str,
    ports: Dict[str, Any],
    test_stimulus: Dict[str, Any],
    timeout: float = 120.0
) -> Dict[str, Any]:
    """
    Verify functional equivalence between Verilog and SystemC implementations.
    
    Args:
        verilog_code: Verilog design code
        systemc_code: SystemC design code
        ports: Port information extracted from the design
        test_stimulus: Test stimulus to apply
        timeout: Total timeout for both simulations
        
    Returns:
        Dict with 'equivalent', 'match_ratio', 'verilog_result', 'systemc_result', 'details'
    """
    result = {
        "equivalent": False,
        "match_ratio": 0.0,
        "verilog_result": None,
        "systemc_result": None,
        "details": ""
    }
    
    # Extract module name from Verilog code
    module_match = re.search(r'module\s+(\w+)', verilog_code)
    if not module_match:
        result["details"] = "Could not find module name in Verilog code"
        return result
    
    module_name = module_match.group(1)
    
    # Generate testbenches
    verilog_tb = generate_verilog_testbench(module_name, ports, test_stimulus)
    systemc_tb = generate_systemc_testbench(module_name, ports, test_stimulus)
    
    # Run simulations in parallel
    verilog_task = run_verilog_simulation(verilog_code, verilog_tb, timeout/2)
    systemc_task = run_systemc_simulation(systemc_code, systemc_tb, timeout/2)
    
    verilog_result, systemc_result = await asyncio.gather(verilog_task, systemc_task)
    
    result["verilog_result"] = verilog_result
    result["systemc_result"] = systemc_result
    
    # Check for errors
    if not verilog_result["success"]:
        result["details"] = f"Verilog simulation failed: {verilog_result.get('error', 'Unknown error')}"
        return result
    
    if not systemc_result["success"]:
        result["details"] = f"SystemC simulation failed: {systemc_result.get('error', 'Unknown error')}"
        # If SystemC fails but Verilog succeeds, give partial credit
        result["match_ratio"] = 0.3
        return result
    
    # Compare outputs
    equivalent, match_ratio, details = compare_signal_outputs(
        verilog_result["outputs"],
        systemc_result["outputs"]
    )
    
    result["equivalent"] = equivalent
    result["match_ratio"] = match_ratio
    result["details"] = details
    
    return result


async def _worker_docker(
    script: str,
    input_val: str,
    expected_output: str,
    timeout: float = 40.0,
    image: str = "python:3.11-slim"
):
    # Ensure base tmp directory exists
    try:
        os.makedirs("tmp", exist_ok=True)
    except Exception:
        pass
    tmpdir = tempfile.mkdtemp(prefix="pllm_exec_",dir="tmp")
    script_path = os.path.join(tmpdir, "script.py")
    def cleanup_tmpdir():
        if not os.path.exists(tmpdir):
            return
        try:
            shutil.rmtree(tmpdir, ignore_errors=False)
        except Exception:
            try:
                subprocess.run(["rm", "-rf", tmpdir], timeout=5, capture_output=True)
            except Exception:
                pass
    stdin_file = None
    stdout_file = None
    stderr_file = None
    printed_output = None
    
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        runner_path = os.path.join(tmpdir, "runner.py")
        runner_code = textwrap.dedent(
            """
            import sys, io, typing

            def _main():
                input_data = sys.stdin.read()
                input_lines = iter(input_data.splitlines())

                def fake_input(prompt: str = "") -> str:
                    try:
                        return next(input_lines)
                    except StopIteration:
                        raise EOFError("No more input")

                original_stdin = sys.stdin
                sys.stdin = io.StringIO(input_data)

                context = {
                    "__name__": "__main__",
                    "input": fake_input,
                    "List": typing.List,
                    "Tuple": typing.Tuple,
                    "Optional": typing.Optional,
                }

                try:
                    with open("script.py", "r", encoding="utf-8") as sf:
                        code_text = sf.read()
                    try:
                        exec(code_text, context)
                    except SystemExit:
                        pass
                    except Exception as e:
                        print(f"error: {e}")
                finally:
                    sys.stdin = original_stdin

            if __name__ == "__main__":
                _main()
            """
        )
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(runner_code)

        stdin_text = input_val
        stdin_path = os.path.join(tmpdir, "stdin.txt")
        stdout_path = os.path.join(tmpdir, "stdout.txt")
        stderr_path = os.path.join(tmpdir, "stderr.txt")

        # pre-write stdin content, and redirect stdout/stderr to temporary files to avoid communication through pipes
        with open(stdin_path, "w", encoding="utf-8") as f_in:
            f_in.write(stdin_text)

        stdin_file = open(stdin_path, "rb")
        stdout_file = open(stdout_path, "wb")
        stderr_file = open(stderr_path, "wb")

        try:
            env = dict(os.environ)
            env.update({
                "PYTHONFAULTHANDLER": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONWARNINGS": "default",
                "PYTHONTRACEMALLOC": "5",
                "PYTHONIOENCODING": "utf-8",
            })

            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-X", "dev", "-W", "default", "-u", runner_path,
                stdin=stdin_file,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=tmpdir,
                env=env,
                start_new_session=True,
            )

            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout-2)
                rc = proc.returncode
            except asyncio.TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    with open(stderr_path, "ab") as f_err_append:
                        msg = f"[parent] Timeout after {timeout}s; process killed.\n".encode()
                        f_err_append.write(msg)
                except Exception:
                    pass
                try:
                    await proc.wait()
                except Exception:
                    pass
                rc = None
                printed_output = None
                print("printed_output: None (timeout)")
                try:
                    if proc.returncode is None:
                        os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    await proc.wait()
                except Exception:
                    pass
                rc = proc.returncode
            if printed_output is None and rc is None:
                pass
            elif rc is not None:
                try:
                    with open(stdout_path, "rb") as f_out:
                        out_bytes = f_out.read()
                except Exception:
                    out_bytes = b""
                try:
                    with open(stderr_path, "rb") as f_err:
                        err_bytes = f_err.read()
                except Exception:
                    err_bytes = b""

                if rc == 0:
                    printed_output = out_bytes.decode(errors="replace")
                else:
                    err_text = (err_bytes or b"").decode(errors="replace").strip()
                    out_text = (out_bytes or b"").decode(errors="replace").strip()
                    combined = err_text or out_text
                    if "Traceback (most recent call last):" in combined:
                        last_line = combined.strip().splitlines()[-1]
                        combined = last_line
                    printed_output = f"error: exit {rc}: {combined}"
        finally:
            for file_handle, file_name in [(stdin_file, "stdin"), (stdout_file, "stdout"), (stderr_file, "stderr")]:
                if file_handle is not None:
                    try:
                        if not file_handle.closed:
                            file_handle.close()
                    except Exception as e:
                        print(f"close {file_name} file handle failed: {e}")
                        
    except Exception as e:
        printed_output = f"error: {e}"

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        cleanup_tmpdir()

    if_passed = await test_if_eq(printed_output, str(expected_output)) if printed_output is not None else False
    
    result = {
        "test_input": input_val,
        "code_execution_output": printed_output,
        "test_output": expected_output,
        "passed": if_passed,
    }
    return result


async def _await_ray_object_ref(obj_ref, timeout_seconds: float = 10.0):
    import ray
    import time
    
    start_time = time.time()
    while True:
        ready, _ = ray.wait([obj_ref], timeout=0.1)
        if ready:
            return ray.get(obj_ref)
        
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise asyncio.TimeoutError(f"Ray task timed out after {timeout_seconds}s")
        

        await asyncio.sleep(0.01)


def get_ray_docker_worker_cls(num_workers=180):
    try:
        import ray  # type: ignore
    except Exception as e:
        print(f"Failed to import ray: {e}")
        return None

    cache_key = f"_cls_{num_workers}"
    if hasattr(get_ray_docker_worker_cls, cache_key):
        return getattr(get_ray_docker_worker_cls, cache_key)

    try:
        import multiprocessing
        total_cpus = multiprocessing.cpu_count()
        cpus_per_worker = min(4.0, (total_cpus * 0.6) / num_workers)
        print(f"Ray worker resource allocation: total_cpus={total_cpus}, num_workers={num_workers}, "
              f"cpus_per_worker={cpus_per_worker:.3f} (60% of total)")
    except Exception as e:
        print(f"Failed to calculate CPU allocation, using default: {e}")
        cpus_per_worker = 0.001

    try:
        _max_conc = 20

        @ray.remote(num_cpus=cpus_per_worker, max_concurrency=10000)
        class _RayDockerWorker:
            def __init__(self, idx):
                if not isinstance(idx, (int, float)):
                    print(f"Warning: idx parameter is not numeric: {type(idx)}, converting to int")
                    try:
                        self.idx = int(idx) if idx is not None else 0
                    except (ValueError, TypeError):
                        self.idx = 0
                else:
                    self.idx = int(idx)

            def get_idx(self):
                return self.idx

            async def run(
                self,
                script: str,
                input_val: str,
                expected_output: str,
                timeout: float = 40.0,  
                image: str = "python:3.11-slim",
            ) -> Dict[str, Any]:
                """Run Python code execution (legacy interface)."""
                try:
                    return await _worker_docker(
                        script=script,
                        input_val=input_val,
                        expected_output=expected_output,
                        timeout=timeout,
                        image=image,
                    )
                except Exception as e:
                    print(f"RayDockerWorker.run failed: {e}")
                    return {
                        "code_execution_output": f"error: {e}",
                        "passed": False,
                        "error": str(e)
                    }

            async def run_verilog(
                self,
                verilog_code: str,
                testbench_code: str,
                timeout: float = 60.0,
            ) -> Dict[str, Any]:
                """Run Verilog simulation using iverilog + vvp."""
                try:
                    return await run_verilog_simulation(
                        verilog_code=verilog_code,
                        testbench_code=testbench_code,
                        timeout=timeout,
                    )
                except Exception as e:
                    print(f"RayDockerWorker.run_verilog failed: {e}")
                    return {
                        "success": False,
                        "outputs": [],
                        "error": str(e),
                        "raw_output": ""
                    }

            async def run_systemc(
                self,
                systemc_code: str,
                testbench_code: str,
                timeout: float = 60.0,
            ) -> Dict[str, Any]:
                """Run SystemC simulation by compiling and executing."""
                try:
                    return await run_systemc_simulation(
                        systemc_code=systemc_code,
                        testbench_code=testbench_code,
                        timeout=timeout,
                    )
                except Exception as e:
                    print(f"RayDockerWorker.run_systemc failed: {e}")
                    return {
                        "success": False,
                        "outputs": [],
                        "error": str(e),
                        "raw_output": ""
                    }

            async def verify_equivalence(
                self,
                verilog_code: str,
                systemc_code: str,
                ports: Dict[str, Any],
                test_stimulus: Dict[str, Any],
                timeout: float = 120.0,
            ) -> Dict[str, Any]:
                """Verify functional equivalence between Verilog and SystemC."""
                try:
                    return await verify_functional_equivalence(
                        verilog_code=verilog_code,
                        systemc_code=systemc_code,
                        ports=ports,
                        test_stimulus=test_stimulus,
                        timeout=timeout,
                    )
                except Exception as e:
                    print(f"RayDockerWorker.verify_equivalence failed: {e}")
                    return {
                        "equivalent": False,
                        "match_ratio": 0.0,
                        "verilog_result": None,
                        "systemc_result": None,
                        "details": str(e)
                    }

        RayDockerWorker = _RayDockerWorker
        cache_key = f"_cls_{num_workers}"
        setattr(get_ray_docker_worker_cls, cache_key, RayDockerWorker)
        return RayDockerWorker
        
    except Exception as e:
        print(f"Failed to create RayDockerWorker class: {e}")
        return None

