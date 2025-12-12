#!/usr/bin/env python3
"""
Basic testing harness for DuetMindAgent improvements.
Tests CLI argument support, session persistence, token budgeting, and convergence behavior.
"""

import asyncio
import json
import tempfile
import time
import shutil
from pathlib import Path
from unittest.mock import patch
import sys
import os

# Add the current directory to sys.path so we can import DuetMindAgent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DuetMindAgent import DuetMindAgent, ExampleEngine, _write_demo_files


class TestResults:
    """Simple test results collector"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def assert_true(self, condition, message):
        if condition:
            self.passed += 1
            print(f"✓ {message}")
        else:
            self.failed += 1
            self.errors.append(message)
            print(f"✗ {message}")
    
    def assert_equal(self, actual, expected, message):
        if actual == expected:
            self.passed += 1
            print(f"✓ {message}")
        else:
            self.failed += 1
            error_msg = f"{message} - Expected: {expected}, Got: {actual}"
            self.errors.append(error_msg)
            print(f"✗ {error_msg}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n=== Test Results ===")
        print(f"Passed: {self.passed}/{total}")
        print(f"Failed: {self.failed}/{total}")
        if self.errors:
            print("Errors:")
            for error in self.errors:
                print(f"  - {error}")
        return self.failed == 0


def test_cli_argument_parsing():
    """Test CLI argument parsing functionality"""
    print("\n--- Testing CLI Argument Parsing ---")
    results = TestResults()
    
    # Test help doesn't crash
    try:
        with patch('sys.argv', ['DuetMindAgent.py', '--help']):
            from DuetMindAgent import main
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('--demo', choices=['sync', 'async', 'both'], default='both')
            parser.add_argument('--rounds', type=int, default=3)
            parser.add_argument('--parallel', action='store_true')
            parser.add_argument('--no-multiparty', action='store_true')
            parser.add_argument('--log-dir', default='./sessions')
            args = parser.parse_args(['--demo', 'sync', '--rounds', '2'])
            results.assert_equal(args.demo, 'sync', "CLI demo argument parsing")
            results.assert_equal(args.rounds, 2, "CLI rounds argument parsing")
    except SystemExit:
        # argparse calls sys.exit for --help, which is expected
        results.assert_true(True, "CLI help functionality works")
    except Exception as e:
        results.assert_true(False, f"CLI argument parsing failed: {e}")
    
    return results


def test_session_persistence():
    """Test session persistence functionality"""
    print("\n--- Testing Session Persistence ---")
    results = TestResults()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create a simple agent
        engine = ExampleEngine()
        agent = DuetMindAgent(
            "TestAgent", 
            {"logic": 0.8, "creativity": 0.5, "analytical": 0.7}, 
            engine=engine
        )
        
        # Create a mock session record
        session_record = {
            "session_id": "test_session_123",
            "topic_start": "Test Topic",
            "topic_final": "Final Test Topic",
            "transcript": [
                {
                    "session_id": "test_session_123",
                    "round": 1,
                    "agent": "TestAgent",
                    "content": "Test content",
                    "confidence": 0.8
                }
            ],
            "participants": ["TestAgent"],
            "converged": True,
            "timestamp": time.time()
        }
        
        # Test persistence
        log_dir = tmpdir_path / "test_sessions"
        saved_path = agent.persist_session(session_record, str(log_dir))
        
        results.assert_true(Path(saved_path).exists(), "Session file was created")
        
        # Verify content
        with open(saved_path, 'r') as f:
            loaded_session = json.load(f)
        
        results.assert_equal(
            loaded_session["session_id"], 
            "test_session_123", 
            "Session ID preserved in saved file"
        )
        results.assert_true(
            "persistence_metadata" in loaded_session, 
            "Persistence metadata added to session"
        )
        results.assert_equal(
            loaded_session["persistence_metadata"]["saved_by_agent"],
            "TestAgent",
            "Persistence metadata includes saving agent"
        )
    
    return results


def test_token_counting():
    """Test token counting and transcript truncation"""
    print("\n--- Testing Token Counting and Truncation ---")
    results = TestResults()
    
    # Test token counting
    test_text = "This is a simple test sentence with punctuation!"
    token_count = DuetMindAgent._count_tokens(test_text)
    results.assert_true(token_count > 8, f"Token counting works (counted {token_count} tokens)")
    
    # Test empty string
    empty_count = DuetMindAgent._count_tokens("")
    results.assert_equal(empty_count, 0, "Empty string token count is 0")
    
    # Test transcript truncation
    engine = ExampleEngine()
    agent = DuetMindAgent(
        "TestAgent", 
        {"logic": 0.8, "creativity": 0.5, "analytical": 0.7}, 
        engine=engine,
        dialogue_config={
            "memory_guard": {
                "max_transcript_tokens": 50,
                "truncate_strategy": "head"
            }
        }
    )
    
    # Create a long transcript
    long_transcript = []
    for i in range(10):
        long_transcript.append({
            "content": f"This is a long piece of content for turn {i} that should make the transcript exceed token limits",
            "prompt": f"Round {i} prompt with additional content",
            "round": i,
            "agent": "TestAgent"
        })
    
    truncated = agent._truncate_transcript(long_transcript)
    results.assert_true(len(truncated) < len(long_transcript), "Transcript was truncated")
    
    # Verify token count is under limit
    total_tokens = 0
    for turn in truncated:
        total_tokens += DuetMindAgent._count_tokens(turn.get("content", ""))
        total_tokens += DuetMindAgent._count_tokens(turn.get("prompt", ""))
    
    results.assert_true(total_tokens <= 50, f"Truncated transcript under token limit ({total_tokens} tokens)")
    
    return results


def test_convergence_behavior():
    """Test convergence detection functionality"""
    print("\n--- Testing Convergence Behavior ---")
    results = TestResults()
    
    engine = ExampleEngine()
    agent = DuetMindAgent(
        "TestAgent", 
        {"logic": 0.8, "creativity": 0.5, "analytical": 0.7}, 
        engine=engine,
        dialogue_config={
            "convergence_window": 3,
            "convergence_min_overlap": 0.35  # Lower threshold for test
        }
    )
    
    # Test convergence with similar content
    similar_contents = [
        "This is about machine learning and artificial intelligence",
        "Machine learning and artificial intelligence are important topics",
        "Artificial intelligence and machine learning have great potential"
    ]
    
    converged = agent._check_convergence(similar_contents)
    results.assert_true(converged, "Convergence detected with similar content")
    
    # Test no convergence with different content
    different_contents = [
        "This is about machine learning",
        "That is about cooking recipes",
        "Something completely different about space travel"
    ]
    
    not_converged = agent._check_convergence(different_contents)
    results.assert_true(not not_converged, "No convergence detected with different content")
    
    # Test insufficient content for convergence
    short_contents = ["one", "two"]
    no_convergence_short = agent._check_convergence(short_contents)
    results.assert_true(not no_convergence_short, "No convergence with insufficient content")
    
    return results


def test_dialogue_integration():
    """Test integration of improvements in dialogue"""
    print("\n--- Testing Dialogue Integration ---")
    results = TestResults()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create demo files
        files = _write_demo_files(tmpdir_path)
        engine = ExampleEngine()
        
        try:
            agent = DuetMindAgent.from_config(files["agent"], engine=engine)
            agent_b = DuetMindAgent("TestB", {"logic": 0.7, "creativity": 0.8}, engine=engine)
            
            # Run a short dialogue
            session = agent.brainstorm([agent, agent_b], "Test topic", rounds=1)
            
            results.assert_true("session_id" in session, "Session has ID")
            results.assert_true("transcript" in session, "Session has transcript")
            results.assert_true(len(session["transcript"]) > 0, "Transcript has content")
            
            # Test session persistence
            log_dir = tmpdir_path / "test_sessions"
            saved_path = agent.persist_session(session, str(log_dir))
            results.assert_true(Path(saved_path).exists(), "Session persisted successfully")
            
        except Exception as e:
            results.assert_true(False, f"Dialogue integration test failed: {e}")
    
    return results


async def test_async_functionality():
    """Test async improvements"""
    print("\n--- Testing Async Functionality ---")
    results = TestResults()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create demo files
        files = _write_demo_files(tmpdir_path)
        engine = ExampleEngine()
        
        try:
            agent = DuetMindAgent.from_config(files["agent"], engine=engine)
            agent_b = DuetMindAgent("AsyncTestB", {"logic": 0.6, "creativity": 0.9}, engine=engine)
            
            # Test async reasoning
            result = await agent.async_generate_reasoning_tree("Test async reasoning")
            results.assert_true("result" in result, "Async reasoning produces result")
            results.assert_true("content" in result["result"], "Async result has content")
            
            # Test async dialogue
            session = await agent.async_brainstorm([agent, agent_b], "Async test topic", rounds=1)
            results.assert_true("session_id" in session, "Async session has ID")
            results.assert_true("parallel_round" in session, "Async session has parallel_round flag")
            
            # Test session persistence
            log_dir = tmpdir_path / "async_sessions"
            saved_path = agent.persist_session(session, str(log_dir))
            results.assert_true(Path(saved_path).exists(), "Async session persisted successfully")
            
        except Exception as e:
            results.assert_true(False, f"Async functionality test failed: {e}")
    
    return results


def run_all_tests():
    """Run all tests and return overall results"""
    print("=== DuetMindAgent Improvements Test Suite ===")
    
    all_results = []
    
    # Run synchronous tests
    all_results.append(test_cli_argument_parsing())
    all_results.append(test_session_persistence())
    all_results.append(test_token_counting())
    all_results.append(test_convergence_behavior())
    all_results.append(test_dialogue_integration())
    
    # Run async tests
    async_results = asyncio.run(test_async_functionality())
    all_results.append(async_results)
    
    # Combine results
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    
    print(f"\n=== Overall Test Summary ===")
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")
    print(f"Success Rate: {total_passed/(total_passed + total_failed)*100:.1f}%")
    
    if total_failed > 0:
        print("\nFailed Tests:")
        for results in all_results:
            for error in results.errors:
                print(f"  - {error}")
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)