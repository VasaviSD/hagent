# See LICENSE for details

import os
import pytest
import tempfile
import json
from unittest.mock import patch, MagicMock

from hagent.tool.aiprof import AIProf


def test_setup_no_api_key():
    """Test setup with no API key."""
    # Ensure environment variable is not set
    with patch.dict('os.environ', {}, clear=True):
        aiprof = AIProf()
        result = aiprof.setup()
        
        assert result is False
        assert "OpenAI API key not provided" in aiprof.error_message


def test_setup_with_api_key():
    """Test setup with API key."""
    aiprof = AIProf()
    
    # Mock check_executable to return True (as if perf is installed)
    with patch.object(aiprof, 'check_executable', return_value=True):
        # Test with direct API key
        result = aiprof.setup(openai_api_key="test_key")
        
        assert result is True
        assert aiprof.error_message == ""
        assert aiprof._openai_key == "test_key"
        assert aiprof._is_ready is True


def test_setup_missing_perf():
    """Test setup with missing perf tool."""
    aiprof = AIProf()
    
    # Mock check_executable to return False for perf
    def mock_check(executable, path=None, raise_error=True):
        if executable == 'perf':
            if raise_error:
                aiprof.set_error("perf not found")
            return False
        return True
    
    with patch.object(aiprof, 'check_executable', side_effect=mock_check):
        result = aiprof.setup(openai_api_key="test_key")
        
        assert result is False
        assert "perf not found" in aiprof.error_message


def test_profile_binary_not_setup():
    """Test profile_binary when tool is not set up."""
    aiprof = AIProf()
    
    result = aiprof.profile_binary("/path/to/binary", "/path/to/source")
    
    assert result == {}
    assert "not set up" in aiprof.error_message


def test_profile_binary_missing_binary():
    """Test profile_binary with missing binary."""
    aiprof = AIProf()
    
    # Setup the tool
    with patch.object(aiprof, 'check_executable', return_value=True):
        aiprof.setup(openai_api_key="test_key")
    
    # Test with non-existent binary
    with patch('os.path.exists', return_value=False):
        result = aiprof.profile_binary("/path/to/binary", "/path/to/source")
        
        assert result == {}
        assert "Binary not found" in aiprof.error_message


def test_profile_binary_missing_source():
    """Test profile_binary with missing source directory."""
    aiprof = AIProf()
    
    # Setup the tool
    with patch.object(aiprof, 'check_executable', return_value=True):
        aiprof.setup(openai_api_key="test_key")
    
    # Mock binary exists but source dir doesn't
    with patch('os.path.exists', side_effect=lambda p: p == "/path/to/binary"):
        with patch('os.access', return_value=True):
            with patch('os.path.isdir', return_value=False):
                result = aiprof.profile_binary("/path/to/binary", "/path/to/source")
                
                assert result == {}
                assert "Source directory not found" in aiprof.error_message


def test_profile_binary_success():
    """Test successful profiling."""
    aiprof = AIProf()
    
    # Setup the tool
    with patch.object(aiprof, 'check_executable', return_value=True):
        aiprof.setup(openai_api_key="test_key")
    
    # Mock necessary functions
    with patch('os.path.exists', return_value=True):
        with patch('os.access', return_value=True):
            with patch('os.path.isdir', return_value=True):
                with patch.object(aiprof, '_check_debug_symbols', return_value=True):
                    with patch.object(aiprof, '_run_perf', return_value={"hotspots": [{"function": "compute", "cpu_time": "72%"}]}):
                        with patch.object(aiprof, '_extract_source_context', return_value={"perf": {"hotspots": [{"function": "compute", "cpu_time": "72%", "source_context": "sample code"}]}}):
                            result = aiprof.profile_binary("/path/to/binary", "/path/to/source")
                            
                            assert result == {"perf": {"hotspots": [{"function": "compute", "cpu_time": "72%", "source_context": "sample code"}]}}


def test_get_optimization_suggestions():
    """Test getting optimization suggestions."""
    aiprof = AIProf()
    
    # Setup the tool
    with patch.object(aiprof, 'check_executable', return_value=True):
        aiprof.setup(openai_api_key="test_key")
    
    # Mock OpenAI API call
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = json.dumps({
        "analysis": "Test analysis",
        "optimizations": [
            {
                "file": "sample.cpp",
                "issue": "Redundant computation",
                "recommendation": "Use running sum",
                "original_code": "for loop",
                "optimized_code": "better for loop",
                "expected_improvement": "50%"
            }
        ]
    })
    
    with patch('openai.chat.completions.create', return_value=mock_completion):
        result = aiprof.get_optimization_suggestions(
            {"perf": {"hotspots": [{"function": "compute", "cpu_time": "72%"}]}},
            {"sample.cpp": "sample code"}
        )
        
        assert result["analysis"] == "Test analysis"
        assert len(result["optimizations"]) == 1
        assert result["optimizations"][0]["file"] == "sample.cpp"


def test_apply_optimizations():
    """Test applying optimization suggestions."""
    aiprof = AIProf()
    
    # Setup the tool
    with patch.object(aiprof, 'check_executable', return_value=True):
        aiprof.setup(openai_api_key="test_key")
    
    # Create temp file for testing
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write("This is the original code with a problem to fix.")
        temp_file_path = temp_file.name
    
    try:
        # Test successful optimization
        suggestions = {
            "optimizations": [
                {
                    "file": temp_file_path,
                    "original_code": "a problem",
                    "optimized_code": "an optimized solution"
                }
            ]
        }
        
        result = aiprof._apply_optimizations(suggestions, "")
        
        assert result is True
        
        # Check file was updated
        with open(temp_file_path, 'r') as f:
            content = f.read()
            assert "This is the original code with an optimized solution to fix." in content
    
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def test_apply_optimizations_no_match():
    """Test applying optimization with no matching code."""
    aiprof = AIProf()
    
    # Setup the tool
    with patch.object(aiprof, 'check_executable', return_value=True):
        aiprof.setup(openai_api_key="test_key")
    
    # Create temp file for testing
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write("This is some code without the target pattern.")
        temp_file_path = temp_file.name
    
    try:
        # Test optimization with no match
        suggestions = {
            "optimizations": [
                {
                    "file": temp_file_path,
                    "original_code": "a pattern that doesn't exist",
                    "optimized_code": "replacement code"
                }
            ]
        }
        
        result = aiprof._apply_optimizations(suggestions, "")
        
        assert result is False
        assert "Original code not found" in aiprof.error_message
    
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 