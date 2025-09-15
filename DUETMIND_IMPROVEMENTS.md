# DuetMindAgent Improvements

This document describes the implemented improvements to the DuetMindAgent system, including CLI argument support, session persistence, token budgeting, and testing harness.

## Overview of Changes

The DuetMindAgent has been enhanced with the following key improvements:

1. **CLI Argument Support** - Flexible command-line interface for demo execution
2. **Session Persistence** - Individual dialogue session logging separate from agent state
3. **Token Budgeting** - Dynamic transcript truncation with configurable limits
4. **Testing Harness** - Comprehensive test suite validating all new functionality

## 1. CLI Argument Support

### Usage

```bash
python DuetMindAgent.py [OPTIONS]

Options:
  --demo {sync,async,both}    Which demo to run (default: both)
  --rounds ROUNDS            Number of rounds for multi-party examples (default: 3)
  --parallel                 Enable parallel_round=True in async multi-party demo
  --no-multiparty           Skip multi-party portion, run only single reasoning tree demo
  --log-dir LOG_DIR         Directory for session logs (default: ./sessions)
```

### Examples

```bash
# Run both sync and async demos (default behavior)
python DuetMindAgent.py

# Run only sync demo with 5 rounds
python DuetMindAgent.py --demo sync --rounds 5

# Run async demo with parallel processing
python DuetMindAgent.py --demo async --parallel

# Run single reasoning only (skip multi-party dialogue)
python DuetMindAgent.py --no-multiparty

# Custom log directory
python DuetMindAgent.py --log-dir /path/to/sessions
```

### Backward Compatibility

Running `python DuetMindAgent.py` without arguments maintains the original behavior (both demos) while utilizing the new argument parsing system.

## 2. Session Persistence

### `persist_session()` Method

New method added to `DuetMindAgent` class:

```python
def persist_session(self, session_record: dict, directory: str = "sessions") -> str:
    """
    Persists a dialogue session record to a JSON file.
    
    Args:
        session_record: Dictionary containing session data (from multi_party_dialogue)
        directory: Directory path where to save session logs (default: "sessions")
        
    Returns:
        str: Full path to the saved session file
    """
```

### Session File Format

Session files are saved as JSON with the following naming convention:
```
{session_id}_{timestamp}.json
```

Each session file contains:
- Original session data (transcript, participants, metrics, etc.)
- Persistence metadata (saved_at, saved_by_agent, filepath)

### Example Session File Structure

```json
{
  "session_id": "dlg_1757917308952_2",
  "topic_start": "Designing resilient edge network",
  "topic_final": "Expand ideas around: Designing resilient edge network",
  "transcript": [...],
  "participants": ["Athena", "Apollo", "Hermes"],
  "converged": true,
  "metrics": {...},
  "persistence_metadata": {
    "saved_at": 1757917309.123,
    "saved_by_agent": "Athena",
    "filepath": "./sessions/dlg_1757917308952_2_1757917309.json"
  }
}
```

## 3. Token Budgeting and Transcript Truncation

### Token Counting

New static method for approximate token counting:

```python
@staticmethod
def _count_tokens(text: str) -> int:
    """
    Simple token counting based on whitespace and punctuation splitting.
    Approximates OpenAI-style tokenization for budget management.
    """
```

### Dynamic Transcript Truncation

New method `_truncate_transcript()` that:
- Monitors token count in dialogue transcripts
- Truncates based on `memory_guard` configuration
- Supports multiple truncation strategies

### Configuration

Token budgeting is configured via the `dialogue_config`:

```python
dialogue_config = {
    "memory_guard": {
        "max_transcript_tokens": 5000,      # Maximum tokens in transcript
        "truncate_strategy": "head"         # "head" or "tail"
    }
}
```

### Truncation Strategies

- **"head"**: Keep newer turns, remove older ones (default)
- **"tail"**: Keep older turns, remove newer ones

### Integration

Transcript truncation is automatically applied during dialogue execution:
- In `multi_party_dialogue()` (sync version)
- In `async_multi_party_dialogue()` (async version)

## 4. Testing Harness

### Test Suite: `test_duetmind_improvements.py`

Comprehensive test suite covering all new functionality:

#### Test Categories

1. **CLI Argument Parsing**
   - Validates argument parsing logic
   - Tests all CLI options

2. **Session Persistence**
   - Tests `persist_session()` method
   - Validates file creation and content
   - Checks persistence metadata

3. **Token Counting and Truncation**
   - Tests token counting accuracy
   - Validates transcript truncation logic
   - Ensures token limits are respected

4. **Convergence Behavior**
   - Tests convergence detection algorithm
   - Validates overlap calculation
   - Tests edge cases (insufficient content, diverse content)

5. **Dialogue Integration**
   - Tests integration of improvements in actual dialogue
   - Validates session persistence in real scenarios

6. **Async Functionality**
   - Tests async reasoning and dialogue
   - Validates parallel processing options
   - Tests async session persistence

### Running Tests

```bash
python test_duetmind_improvements.py
```

### Test Results

Current test suite achieves **100% success rate** (22/22 tests passing):

```
=== Overall Test Summary ===
Total Passed: 22
Total Failed: 0
Success Rate: 100.0%
```

## 5. Implementation Details

### Key Changes to `DuetMindAgent.py`

1. **Imports**: Added `argparse` for CLI support
2. **New Methods**:
   - `persist_session()` - Session persistence
   - `_count_tokens()` - Token counting
   - `_truncate_transcript()` - Transcript truncation
   - `main()` - CLI entry point

3. **Enhanced Demo Functions**:
   - `_demo_sync()` - Enhanced with parameters
   - `_demo_async()` - Enhanced with parameters
   - `_demo_both()` - New function for both demos

4. **Updated Dialogue Methods**:
   - Integrated transcript truncation
   - Added session persistence calls

### Configuration Changes

The `dialogue_config` now supports `memory_guard` settings:

```python
dialogue_config = {
    "default_rounds": 3,
    "max_rounds": 20,
    "convergence_window": 3,
    "convergence_min_overlap": 0.55,
    "convergence_min_conf": 0.75,
    "memory_guard": {
        "max_transcript_tokens": 5000,
        "truncate_strategy": "head"
    }
}
```

## 6. Usage Examples

### Basic Usage

```bash
# Default behavior - both demos with session persistence
python DuetMindAgent.py

# Only sync demo, 2 rounds, custom log directory
python DuetMindAgent.py --demo sync --rounds 2 --log-dir my_sessions

# Async with parallel processing
python DuetMindAgent.py --demo async --parallel --rounds 4
```

### Session Files

Sessions are automatically saved to JSON files:

```bash
$ ls sessions/
dlg_1757917308952_2_1757917309.json
adlg_1757917407150_1_1757917407.json
```

### Loading Sessions

Session files can be loaded and analyzed:

```python
import json
with open('sessions/dlg_1757917308952_2_1757917309.json', 'r') as f:
    session = json.load(f)
    print(f"Topic: {session['topic_final']}")
    print(f"Participants: {session['participants']}")
    print(f"Converged: {session['converged']}")
```

## 7. Performance Considerations

### Token Budgeting Benefits

- Prevents memory bloat in long dialogues
- Maintains consistent performance
- Configurable limits for different use cases

### Session Persistence Benefits

- Individual session analysis
- Historical dialogue tracking
- Debugging and research capabilities
- Separate from agent state persistence

### Async Improvements

- Parallel processing option for faster multi-agent dialogues
- Maintains backward compatibility with sequential processing
- Async session persistence for non-blocking I/O

## 8. Future Enhancements

Potential areas for further improvement:

1. **Advanced Token Counting**: Integration with actual tokenizers (tiktoken, transformers)
2. **Session Analytics**: Built-in analysis tools for session files
3. **Distributed Persistence**: Support for remote storage (databases, cloud storage)
4. **Real-time Monitoring**: Live dashboard for ongoing dialogues
5. **Session Resumption**: Ability to resume interrupted dialogues

## 9. Testing and Validation

All improvements have been thoroughly tested:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end dialogue testing
- **CLI Tests**: Command-line interface validation
- **Async Tests**: Asynchronous functionality testing
- **Persistence Tests**: File I/O and data integrity testing

The test suite provides comprehensive coverage and serves as documentation for expected behavior.