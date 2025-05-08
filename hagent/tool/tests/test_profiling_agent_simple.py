#!/usr/bin/env python3
"""
Example usage of the ProfilingAgent tool.

This script demonstrates how to use the ProfilingAgent tool to profile
and optimize a compiled binary.
"""

import os
import sys
import tempfile
import shutil
from hagent.tool.profiling_agent import ProfilingAgent


def write_sample_code(directory: str) -> str:
    """Create a sample C++ file with inefficient code."""
    sample_code = """
#include <iostream>
#include <vector>
#include <chrono>

// Inefficient function with redundant computation
void compute(const std::vector<int>& data, std::vector<int>& results) {
    for (size_t i = 0; i < data.size(); ++i) {
        int sum = 0;
        // Redundant computation - recalculating sum for each element
        for (size_t j = 0; j <= i; ++j) {
            sum += data[j];
        }
        results[i] = sum;
    }
}

int main() {
    const int size = 10000;
    std::vector<int> data(size);
    std::vector<int> results(size);
    
    // Fill with test data
    for (int i = 0; i < size; ++i) {
        data[i] = i % 100;
    }
    
    // Measure time
    auto start = std::chrono::high_resolution_clock::now();
    compute(data, results);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time: " << diff.count() << " seconds" << std::endl;
    
    // Print a few results to verify
    for (int i = 0; i < 5; ++i) {
        std::cout << "results[" << i << "] = " << results[i] << std::endl;
    }
    
    return 0;
}
"""
    
    file_path = os.path.join(directory, "sample.cpp")
    with open(file_path, "w") as f:
        f.write(sample_code)
    
    return file_path


def compile_sample(file_path: str) -> str:
    """Compile the sample C++ code."""
    binary_path = file_path.replace(".cpp", "")
    
    import subprocess
    cmd = ["g++", "-g", "-O0", file_path, "-o", binary_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}")
        sys.exit(1)
    
    return binary_path


def main():
    # Create a temporary directory for our sample code
    temp_dir = tempfile.mkdtemp(prefix="profagent_test_")
    
    try:
        # Create and compile sample code
        source_file = write_sample_code(temp_dir)
        binary_path = compile_sample(source_file)
        
        print(f"Created and compiled sample code at: {source_file}")
        
        # Initialize the ProfilingAgent tool
        profiler = ProfilingAgent()
        
        # Setup the tool - will use OPENAI_API_KEY from environment
        if not profiler.setup():
            print(f"Setup failed: {profiler.error_message}")
            sys.exit(1)
        
        print("ProfilingAgent tool setup successfully")
        
        # Profile the binary
        print("Profiling binary...")
        profile_results = profiler.profile_binary(binary_path, temp_dir)
        
        if not profile_results:
            print(f"Profiling failed: {profiler.error_message}")
            sys.exit(1)
        
        print("Profiling completed successfully")
        print(f"Found {len(profile_results.get('perf', {}).get('hotspots', []))} hotspots")
        
        # Directly check if the tool is still ready before continuing
        if not profiler._is_ready:
            print(f"Warning: ProfilingAgent is no longer ready after profiling. Error: {profiler.error_message}")
            # Re-setup if needed
            if not profiler.setup():
                print(f"Re-setup failed: {profiler.error_message}")
                sys.exit(1)
            print("ProfilingAgent was re-setup successfully")
        
        # Get source context
        source_context = profiler._get_source_context(profile_results, temp_dir)
        
        # Get optimization suggestions
        print("Getting optimization suggestions...")
        suggestions = profiler.get_optimization_suggestions(profile_results, source_context)
        
        if not suggestions:
            print(f"Failed to get optimization suggestions: {profiler.error_message}")
            
            # If we don't have enough data from profiling, create a simple mock example
            # for demonstration purposes when no real hotspots are found
            if len(profile_results.get('perf', {}).get('hotspots', [])) == 0:
                print("No hotspots found. Creating a mock example for demonstration purposes.")
                suggestions = {
                    "analysis": "Example optimization for the compute function",
                    "optimizations": [
                        {
                            "file": os.path.join(temp_dir, "sample.cpp"),
                            "issue": "Redundant computation in nested loops",
                            "recommendation": "Use running sum to avoid recalculating for each element",
                            "original_code": "for (size_t j = 0; j <= i; ++j) {\n            sum += data[j];\n        }",
                            "optimized_code": "// Use running sum approach\n        if (i == 0) {\n            sum = data[0];\n        } else {\n            sum = results[i-1] + data[i];\n        }",
                            "expected_improvement": "O(n²) → O(n), significant speedup for large arrays"
                        }
                    ]
                }
                print("Created mock optimization suggestion.")
            else:
                sys.exit(1)
        
        print("Received optimization suggestions:")
        print(f"Analysis: {suggestions.get('analysis', 'No analysis provided')}")
        print(f"Number of optimizations: {len(suggestions.get('optimizations', []))}")
        
        # Apply optimizations
        print("Applying optimizations...")
        output_dir = os.path.join(temp_dir, "optimized")
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy source files
        profiler._copy_source_files(temp_dir, output_dir)
        
        # Apply the optimizations
        changes_made = profiler._apply_optimizations(suggestions, output_dir)
        
        if changes_made:
            print("Successfully applied optimizations")
            print(f"Optimized code is in: {output_dir}")
            
            # Optionally keep the temp directory for inspection
            if not os.environ.get("KEEP_TEMP") == "1":
                print("Set KEEP_TEMP=1 to keep the temporary files for inspection.")
        else:
            print(f"No changes made: {profiler.error_message}")
        
        # In a full implementation, we would recompile and compare performance
        
    finally:
        # Clean up
        if os.environ.get("KEEP_TEMP") != "1":
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")
        else:
            print(f"Keeping temporary directory: {temp_dir}")


if __name__ == "__main__":
    main() 