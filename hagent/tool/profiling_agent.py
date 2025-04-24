# See LICENSE for details

import os
import json
import tempfile
import subprocess
from typing import Optional, List, Dict, Any, Union
import openai
from hagent.tool.tool import Tool


class ProfilingAgent(Tool):
    """
    ProfilingAgent: A tool that automates the profiling and optimization of compiled binaries.
    
    This tool uses system profiling tools (perf, redspy, zerospy, loadspy) to identify
    performance bottlenecks, converts the profiling data to a structured format,
    and uses OpenAI's API to generate optimization suggestions.
    """

    def __init__(self):
        """
        Initialize the ProfilingAgent tool.
        """
        super().__init__()
        self._profiling_tools = {
            'perf': None,
            'redspy': None,
            'zerospy': None,
            'loadspy': None
        }
        self._openai_key = None
        self._openai_model = 'gpt-4'
        self._temp_dir = None
        
    def setup(self, 
              openai_api_key: Optional[str] = None, 
              openai_model: str = 'gpt-4',
              profiling_tools_path: Optional[str] = None) -> bool:
        """
        Set up the ProfilingAgent tool with the required dependencies.
        
        Args:
            openai_api_key: The OpenAI API key. If None, will try to use OPENAI_API_KEY env var.
            openai_model: The OpenAI model to use (default: 'gpt-4')
            profiling_tools_path: Optional path to profiling tools binaries
            
        Returns:
            True if setup was successful, False otherwise.
        """
        # Reset state
        self._is_ready = False
        self.error_message = ''
        
        # Check OpenAI API key
        self._openai_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        if not self._openai_key:
            self.set_error('OpenAI API key not provided and OPENAI_API_KEY environment variable not set')
            return False
            
        # Set OpenAI model
        self._openai_model = openai_model
        
        # Check profiling tools
        tools_available = True
        
        # Check perf
        if not self.check_executable('perf', profiling_tools_path):
            tools_available = False
            
        # For the other tools, they're optional but we'll note if they're missing
        for tool in ['redspy', 'zerospy', 'loadspy']:
            self._profiling_tools[tool] = self.check_executable(tool, profiling_tools_path, raise_error=False)
        
        if not tools_available:
            return False
            
        # Create a temporary directory for outputs
        self._temp_dir = tempfile.mkdtemp(prefix='profagent_')
        
        self._is_ready = True
        return True
    
    def check_executable(self, executable: str, path: Optional[str] = None, raise_error: bool = True) -> bool:
        """
        Check if an executable exists and is accessible.
        
        Overriding the base method to allow for optional executables.
        
        Args:
            executable: The name of the executable to check
            path: Optional specific path to check, if None uses system PATH
            raise_error: Whether to set error_message on failure
            
        Returns:
            True if the executable is found and accessible, False otherwise.
        """
        if path:
            exec_path = os.path.join(path, executable)
            if not (os.path.exists(exec_path) and os.access(exec_path, os.X_OK)):
                if raise_error:
                    self.set_error(f'{executable} not found or not executable at {exec_path}')
                return False
            return True
        else:
            import shutil
            which_result = shutil.which(executable)
            if not which_result:
                if raise_error:
                    self.set_error(f'{executable} not found in PATH')
                return False
            return True
    
    def profile_binary(self, 
                      binary_path: str, 
                      source_dir: str,
                      args: Optional[List[str]] = None, 
                      tools: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Profile a binary using available profiling tools.
        
        Args:
            binary_path: Path to the compiled binary to profile
            source_dir: Path to the source code directory
            args: Optional list of arguments to pass to the binary
            tools: Optional list of profiling tools to use (default: all available)
            
        Returns:
            A dictionary containing the profiling results
        """
        if not self._is_ready:
            self.set_error('ProfilingAgent tool is not set up. Please run setup first.')
            return {}
            
        if not os.path.exists(binary_path) or not os.access(binary_path, os.X_OK):
            self.set_error(f'Binary not found or not executable: {binary_path}')
            return {}
            
        if not os.path.isdir(source_dir):
            self.set_error(f'Source directory not found: {source_dir}')
            return {}
            
        # If no specific tools requested, use all available
        if tools is None:
            tools = [tool for tool, available in self._profiling_tools.items() if available or tool == 'perf']
            
        # Ensure binary is compiled with debug symbols
        has_debug_symbols = self._check_debug_symbols(binary_path)
        if not has_debug_symbols:
            self.set_error(f'Binary does not have debug symbols: {binary_path}. Please compile with -g flag.')
            return {}
            
        # Run each requested profiling tool
        profile_results = {}
        for tool in tools:
            if tool == 'perf':
                profile_results['perf'] = self._run_perf(binary_path, args)
            elif tool == 'redspy' and self._profiling_tools['redspy']:
                profile_results['redspy'] = self._run_redspy(binary_path, args)
            elif tool == 'zerospy' and self._profiling_tools['zerospy']:
                profile_results['zerospy'] = self._run_zerospy(binary_path, args)
            elif tool == 'loadspy' and self._profiling_tools['loadspy']:
                profile_results['loadspy'] = self._run_loadspy(binary_path, args)
                
        # Extract source code for hotspots
        profile_results = self._extract_source_context(profile_results, source_dir)
                
        return profile_results
    
    def get_optimization_suggestions(self, 
                                    profile_results: Dict[str, Any],
                                    source_context: Dict[str, str]) -> Dict[str, Any]:
        """
        Use OpenAI API to get optimization suggestions based on profiling results.
        
        Args:
            profile_results: Dictionary of profiling results from profile_binary
            source_context: Dictionary mapping file paths to source code
            
        Returns:
            Dictionary containing optimization suggestions
        """
        if not self._is_ready:
            self.set_error('ProfilingAgent tool is not set up. Please run setup first.')
            return {}
            
        # Construct the prompt for OpenAI
        prompt = self._construct_optimization_prompt(profile_results, source_context)
        
        # Call OpenAI API
        try:
            completion = openai.chat.completions.create(
                model=self._openai_model,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert compiler and performance optimization assistant. "
                        "Analyze the profiling data and source code to suggest specific, "
                        "actionable optimizations. Format your response as JSON with both "
                        "natural language explanations and code snippets where appropriate."
                    )},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            response = completion.choices[0].message.content
            suggestions = json.loads(response)
            return suggestions
        except Exception as e:
            self.set_error(f'Error calling OpenAI API: {str(e)}')
            return {}
            
    def optimize_binary(self, 
                       binary_path: str,
                       source_dir: str,
                       output_dir: Optional[str] = None,
                       args: Optional[List[str]] = None,
                       max_iterations: int = 3) -> Dict[str, Any]:
        """
        Full pipeline: profile binary, get suggestions, and generate optimized code.
        
        Args:
            binary_path: Path to the compiled binary
            source_dir: Path to the source code directory
            output_dir: Directory to write optimized source files (if None, creates temp dir)
            args: Optional arguments to pass to the binary during profiling
            max_iterations: Maximum optimization iterations to attempt
            
        Returns:
            Dictionary with optimization results and file paths
        """
        if not self._is_ready:
            self.set_error('ProfilingAgent tool is not set up. Please run setup first.')
            return {}
            
        # Create output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(self._temp_dir, 'optimized')
            os.makedirs(output_dir, exist_ok=True)
        elif not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        results = {
            "iterations": [],
            "final_output_dir": output_dir,
            "success": False
        }
        
        # Copy source files to output directory for first iteration
        self._copy_source_files(source_dir, output_dir)
        current_source_dir = output_dir
        
        for i in range(max_iterations):
            # Profile the binary
            profile_results = self.profile_binary(binary_path, current_source_dir, args)
            if not profile_results:
                results["iterations"].append({
                    "iteration": i + 1,
                    "status": "failed",
                    "error": self.error_message
                })
                break
                
            # Extract source context for hotspots
            source_context = self._get_source_context(profile_results, current_source_dir)
                
            # Get optimization suggestions
            suggestions = self.get_optimization_suggestions(profile_results, source_context)
            if not suggestions:
                results["iterations"].append({
                    "iteration": i + 1,
                    "status": "failed",
                    "error": self.error_message
                })
                break
                
            # Apply optimizations to source files
            optimization_applied = self._apply_optimizations(suggestions, current_source_dir)
            
            results["iterations"].append({
                "iteration": i + 1,
                "status": "success" if optimization_applied else "no_changes",
                "profile_results": profile_results,
                "suggestions": suggestions
            })
            
            if not optimization_applied:
                # No changes made, stop iterations
                break
                
            # TODO: In a full implementation, we would recompile the binary and continue
            # For now, we'll just simulate one iteration
            break
            
        results["success"] = len(results["iterations"]) > 0 and results["iterations"][-1]["status"] == "success"
        return results
    
    def _check_debug_symbols(self, binary_path: str) -> bool:
        """Check if binary has debug symbols."""
        try:
            result = self.run_command(['readelf', '--debug-dump=info', binary_path])
            return result is not None and 'debug_info' in result.stdout
        except Exception:
            # Default to True if we can't check - we'll find out during profiling
            return True
            
    def _run_perf(self, binary_path: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run perf profiling tool and parse results."""
        perf_output = os.path.join(self._temp_dir, 'perf.data')
        cmd = ['perf', 'record', '-o', perf_output, '-g', binary_path]
        if args:
            cmd.extend(args)
            
        try:
            self.run_command(cmd)
            
            # Generate report
            report_output = os.path.join(self._temp_dir, 'perf_report.txt')
            self.run_command(['perf', 'report', '-i', perf_output, '--stdio'], stdout=open(report_output, 'w'))
            
            # Parse the report
            return self._parse_perf_report(report_output)
        except Exception as e:
            self.set_error(f'Error running perf: {str(e)}')
            return {}
            
    def _run_redspy(self, binary_path: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Placeholder for RedSpy tool integration."""
        # In a real implementation, this would run RedSpy tool
        return {"tool": "redspy", "status": "not_implemented"}
            
    def _run_zerospy(self, binary_path: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Placeholder for ZeroSpy tool integration."""
        # In a real implementation, this would run ZeroSpy tool
        return {"tool": "zerospy", "status": "not_implemented"}
            
    def _run_loadspy(self, binary_path: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Placeholder for LoadSpy tool integration."""
        # In a real implementation, this would run LoadSpy tool
        return {"tool": "loadspy", "status": "not_implemented"}
            
    def _parse_perf_report(self, report_path: str) -> Dict[str, Any]:
        """Parse perf report output into structured data."""
        hotspots = []
        
        try:
            with open(report_path, 'r') as f:
                lines = f.readlines()
                
            current_sample = None
            for line in lines:
                line = line.strip()
                
                # Look for percentage lines
                if line and line[0].isdigit() and '%' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        percentage = parts[0].rstrip('%')
                        symbol = parts[-1]
                        
                        # Skip non-user code symbols
                        if '[kernel' in line or symbol.startswith('_'):
                            continue
                            
                        current_sample = {
                            "function": symbol,
                            "cpu_time": percentage + "%"
                        }
                        
                # Look for source file information
                elif current_sample and ':' in line and '/' in line:
                    # This might be a filename:line format
                    parts = line.split(':')
                    if len(parts) >= 2 and os.path.exists(parts[0]):
                        current_sample["file"] = parts[0]
                        current_sample["line"] = int(parts[1].split()[0])
                        hotspots.append(current_sample)
                        current_sample = None
                        
            return {
                "tool": "perf",
                "hotspots": hotspots
            }
        except Exception as e:
            self.set_error(f'Error parsing perf report: {str(e)}')
            return {"tool": "perf", "hotspots": [], "error": str(e)}
            
    def _extract_source_context(self, profile_results: Dict[str, Any], source_dir: str) -> Dict[str, Any]:
        """Extract source code context for each hotspot."""
        for tool_name, tool_results in profile_results.items():
            if tool_name == 'perf' and 'hotspots' in tool_results:
                for hotspot in tool_results['hotspots']:
                    if 'file' in hotspot and 'line' in hotspot:
                        file_path = hotspot['file']
                        line_number = hotspot['line']
                        
                        # If the file path is absolute, use it directly
                        if not os.path.isabs(file_path):
                            file_path = os.path.join(source_dir, file_path)
                            
                        try:
                            if os.path.exists(file_path):
                                with open(file_path, 'r') as f:
                                    lines = f.readlines()
                                    
                                # Extract context (10 lines before and after)
                                start = max(0, line_number - 10)
                                end = min(len(lines), line_number + 10)
                                context = ''.join(lines[start:end])
                                hotspot['source_context'] = context
                                hotspot['source_start_line'] = start + 1
                        except Exception as e:
                            self.set_error(f'Error extracting source context: {str(e)}')
                            
        return profile_results
        
    def _construct_optimization_prompt(self, profile_results: Dict[str, Any], source_context: Dict[str, str]) -> str:
        """Construct prompt for OpenAI API with profiling results and source context."""
        prompt = "Based on the following profiling data:\n"
        prompt += json.dumps(profile_results, indent=2)
        
        prompt += "\n\nAnd the source code context:\n"
        for file_path, source in source_context.items():
            prompt += f"\nFile: {file_path}\n```\n{source}\n```\n"
            
        prompt += "\nPlease analyze the performance bottlenecks and suggest optimizations. Your response should include:\n"
        prompt += "1. An analysis of the main performance issues\n"
        prompt += "2. Specific optimization recommendations with code examples\n"
        prompt += "3. Expected performance improvements\n"
        
        prompt += "\nFormat your response as a JSON object with the following structure:\n"
        prompt += "{\n"
        prompt += '  "analysis": "Overall analysis of the performance issues",\n'
        prompt += '  "optimizations": [\n'
        prompt += '    {\n'
        prompt += '      "file": "path/to/file.cpp",\n'
        prompt += '      "issue": "Description of the issue",\n'
        prompt += '      "recommendation": "Explanation of the optimization",\n'
        prompt += '      "original_code": "The problematic code snippet",\n'
        prompt += '      "optimized_code": "The improved code snippet",\n'
        prompt += '      "expected_improvement": "Estimated performance gain"\n'
        prompt += '    }\n'
        prompt += '  ]\n'
        prompt += '}'
        
        return prompt
        
    def _get_source_context(self, profile_results: Dict[str, Any], source_dir: str) -> Dict[str, str]:
        """Extract source code context from profiling results."""
        source_context = {}
        
        for tool_name, tool_results in profile_results.items():
            if tool_name == 'perf' and 'hotspots' in tool_results:
                for hotspot in tool_results['hotspots']:
                    if 'file' in hotspot:
                        file_path = hotspot['file']
                        
                        # If we haven't loaded this file yet
                        if file_path not in source_context:
                            # If the file path is absolute, use it directly
                            full_path = file_path if os.path.isabs(file_path) else os.path.join(source_dir, file_path)
                            
                            try:
                                if os.path.exists(full_path):
                                    with open(full_path, 'r') as f:
                                        source_context[file_path] = f.read()
                            except Exception as e:
                                self.set_error(f'Error reading source file {full_path}: {str(e)}')
                                
        return source_context
        
    def _copy_source_files(self, source_dir: str, output_dir: str) -> None:
        """Copy source files to output directory."""
        import shutil
        
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith(('.c', '.cpp', '.cc', '.h', '.hpp', '.rs')):
                    source_path = os.path.join(root, file)
                    # Create relative path to maintain directory structure
                    rel_path = os.path.relpath(source_path, source_dir)
                    dest_path = os.path.join(output_dir, rel_path)
                    
                    # Create directories if they don't exist
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    # Copy the file
                    shutil.copy2(source_path, dest_path)
                    
    def _apply_optimizations(self, suggestions: Dict[str, Any], source_dir: str) -> bool:
        """Apply optimization suggestions to source files."""
        changes_made = False
        
        if 'optimizations' not in suggestions:
            return False
            
        for opt in suggestions['optimizations']:
            if all(key in opt for key in ['file', 'original_code', 'optimized_code']):
                file_path = opt['file']
                original_code = opt['original_code']
                optimized_code = opt['optimized_code']
                
                # Find the full path to the file
                full_path = file_path if os.path.isabs(file_path) else os.path.join(source_dir, file_path)
                
                if not os.path.exists(full_path):
                    self.set_error(f'File not found: {full_path}')
                    continue
                    
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                        
                    # Replace the code
                    if original_code in content:
                        new_content = content.replace(original_code, optimized_code)
                        
                        with open(full_path, 'w') as f:
                            f.write(new_content)
                            
                        changes_made = True
                    else:
                        self.set_error(f'Original code not found in {file_path}')
                except Exception as e:
                    self.set_error(f'Error applying optimization to {file_path}: {str(e)}')
                    
        return changes_made 