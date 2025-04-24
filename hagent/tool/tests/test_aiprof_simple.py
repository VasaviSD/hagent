#!/usr/bin/env python3
"""
Example usage of the AIProf tool.

This script demonstrates how to use the AIProf tool to profile
and optimize a compiled binary.
"""

import os
import sys
import tempfile
import shutil
from hagent.tool.aiprof import AIProf


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
    temp_dir = tempfile.mkdtemp(prefix="aiprof_test_")
    
    try:
        # Create and compile sample code
        source_file = write_sample_code(temp_dir)
        binary_path = compile_sample(source_file)
        
        print(f"Created and compiled sample code at: {source_file}")
        
        # Initialize the AIProf tool
        aiprof = AIProf()
        
        # Setup the tool - will use OPENAI_API_KEY from environment
        if not aiprof.setup():
            print(f"Setup failed: {aiprof.error_message}")
            sys.exit(1)
        
        print("AIProf tool setup successfully")
        
        # Profile the binary
        print("Profiling binary...")
        profile_results = aiprof.profile_binary(binary_path, temp_dir)
        
        if not profile_results:
            print(f"Profiling failed: {aiprof.error_message}")
            sys.exit(1)
        
        print("Profiling completed successfully")
        print(f"Found {len(profile_results.get('perf', {}).get('hotspots', []))} hotspots")
        
        # Get source context
        source_context = aiprof._get_source_context(profile_results, temp_dir)
        
        # Get optimization suggestions
        print("Getting optimization suggestions...")
        suggestions = aiprof.get_optimization_suggestions(profile_results, source_context)
        
        if not suggestions:
            print(f"Failed to get optimization suggestions: {aiprof.error_message}")
            sys.exit(1)
        
        print("Received optimization suggestions:")
        print(f"Analysis: {suggestions.get('analysis', 'No analysis provided')}")
        print(f"Number of optimizations: {len(suggestions.get('optimizations', []))}")
        
        # Apply optimizations
        print("Applying optimizations...")
        output_dir = os.path.join(temp_dir, "optimized")
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy source files
        aiprof._copy_source_files(temp_dir, output_dir)
        
        # Apply the optimizations
        changes_made = aiprof._apply_optimizations(suggestions, output_dir)
        
        if changes_made:
            print("Successfully applied optimizations")
            print(f"Optimized code is in: {output_dir}")
        else:
            print(f"No changes made: {aiprof.error_message}")
        
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