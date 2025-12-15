"""
Evaluator for Freqtrade Strategy Optimization
"""

import os
import shutil
import subprocess
import ast
import re
from openevolve.evaluation_result import EvaluationResult

def get_strategy_class_name(file_path):
    """Parse the python file to find the class name."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                # We assume the strategy is the first class or we could check base classes
                # In simple files, taking the first class is usually correct.
                return node.name
    except Exception as e:
        print(f"Error parsing class name: {e}")
    return None

def evaluate(program_path):
    """
    Evaluate the strategy by running freqtrade backtesting.
    """
    # 1. Identify Class Name
    class_name = get_strategy_class_name(program_path)
    if not class_name:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": "No class found"},
            artifacts={"error": "Could not find a class definition in the file"}
        )

    # 2. Setup paths
    # Ensure user_data/strategies exists
    # We use the current working directory as the base for freqtrade execution
    cwd = os.getcwd()
    user_data_dir = os.path.join(cwd, "user_data")
    strategies_dir = os.path.join(user_data_dir, "strategies")
    
    # Create directories if they don't exist (freqtrade create-userdir structure)
    os.makedirs(strategies_dir, exist_ok=True)
    
    # Copy the program to the strategies dir
    # We name the file same as class name because Freqtrade often prefers that convention
    target_path = os.path.join(strategies_dir, f"{class_name}.py")
    try:
        shutil.copy(program_path, target_path)
    except Exception as e:
         return EvaluationResult(
            metrics={"combined_score": 0.0, "error": "Copy failed"},
            artifacts={"error": str(e)}
        )

    # 3. Run Freqtrade Backtesting
    # We look for config.json in the current directory
    config_path = os.path.join(cwd, "config.json")
    
    cmd = [
        "freqtrade", "backtesting",
        "--strategy", class_name,
        "--accept-agreement", # In case it asks
    ]
    
    if os.path.exists(config_path):
        cmd.extend(["--config", "config.json"])
    else:
        # If no config, we might fail, but let's try assuming defaults or user has set it up
        print("Warning: config.json not found in current directory.")

    # Optional: Add timerange if you want to speed it up or enforce specific period
    # cmd.extend(["--timerange", "20240101-20240201"])
    
    print(f"Running backtest for {class_name}...")
    
    try:
        # Run with timeout
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600 # 10 minutes timeout
        )
        
        stdout = result.stdout
        stderr = result.stderr
        
        if result.returncode != 0:
             return EvaluationResult(
                metrics={"combined_score": 0.0, "error": "Execution failed"},
                artifacts={"stdout": stdout[-1000:], "stderr": stderr}
            )

        # 4. Parse Output
        # We look for the table line containing the strategy name
        # The separator can be │ or ┃
        
        lines = stdout.splitlines()
        metrics = {}
        found = False
        
        for line in lines:
            # Normalize separators
            clean_line = line.replace('┃', '│')
            
            if class_name in clean_line and "│" in clean_line:
                parts = [p.strip() for p in clean_line.split("│")]
                # Expected format after split by │:
                # 0: ''
                # 1: Strategy Name
                # 2: Trades
                # 3: Avg Profit %
                # 4: Tot Profit USDT
                # 5: Tot Profit %
                # 6: Avg Duration
                # 7: Win Draw Loss Win%
                # 8: Drawdown
                # 9: ''
                
                if len(parts) >= 9:
                    try:
                        trades_str = parts[2]
                        if trades_str == "0":
                            # No trades
                            metrics["trades"] = 0
                            metrics["profit_pct"] = 0.0
                            metrics["win_rate"] = 0.0
                            metrics["drawdown"] = 0.0
                            found = True
                            break
                            
                        metrics["trades"] = int(trades_str)
                        metrics["profit_pct"] = float(parts[5])
                        
                        # Win rate part: "8 0 2 80" -> split by space, take last
                        win_part = parts[7].split()
                        if win_part:
                            metrics["win_rate"] = float(win_part[-1])
                        else:
                            metrics["win_rate"] = 0.0
                        
                        drawdown_str = parts[8].replace('%', '')
                        metrics["drawdown"] = float(drawdown_str)
                        
                        found = True
                        break
                    except ValueError:
                        continue

        if not found:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": "Parsing failed"},
                artifacts={"stdout_tail": stdout[-2000:], "message": "Could not find strategy stats in output"}
            )
            
        # 5. Calculate Score
        # We want high profit, high win rate, low drawdown.
        # Ensure we have trades.
        if metrics["trades"] == 0:
             return EvaluationResult(
                metrics={"combined_score": -100.0, "profit_score": 0.0},
                artifacts={"message": "No trades executed", "metrics": metrics}
            )
            
        # Score calculation
        # Profit %: direct contribution
        # Win Rate: 0-100, let's weight it.
        # Drawdown: negative contribution.
        
        # Example: Profit 20%, Win 60%, Drawdown 5% -> 20 + 60*0.2 - 5*2 = 20 + 12 - 10 = 22
        score = metrics["profit_pct"] + (metrics["win_rate"] * 0.2) - (metrics["drawdown"] * 2.0)
        
        return EvaluationResult(
            metrics={
                "combined_score": score,
                "profit_score": metrics["profit_pct"],
                "win_score": metrics["win_rate"],
                "trades": metrics["trades"],
                "drawdown": metrics["drawdown"]
            },
            artifacts={"metrics": metrics}
        )
        
    except subprocess.TimeoutExpired:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": "Timeout"},
            artifacts={"error": "Backtesting timed out"}
        )
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": "Exception"},
            artifacts={"error": str(e)}
        )

# Stage 2 evaluation (could be more rigorous, e.g., longer timerange)
def evaluate_stage2(program_path):
    return evaluate(program_path)
