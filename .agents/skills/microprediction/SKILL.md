```markdown
# microprediction Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill teaches you how to contribute to the `microprediction` Python codebase, which focuses on creating and extending forecasting platform starter kits. You'll learn the project's coding conventions, common workflows for developing and enhancing platform integrations, and how to structure and test your contributions.

## Coding Conventions

- **File Naming:**  
  Use `camelCase` for Python files.  
  _Example:_  
  ```
  streamingAttacker.py
  crossValidation.py
  ```

- **Import Style:**  
  Use **relative imports** within modules.  
  _Example:_  
  ```python
  from .helpers import some_function
  ```

- **Export Style:**  
  Use **named exports** (explicit function and class definitions).  
  _Example:_  
  ```python
  def my_forecaster(...):
      ...
  ```

- **Directory Structure:**  
  Platform starter kits are organized under `explorations/build_on_both/<platform>/`.

## Workflows

### Platform Starter Kit Development
**Trigger:** When you want to add a new platform integration or example pipeline.  
**Command:** `/new-platform-starter`

1. **Create a new subdirectory** under `explorations/build_on_both/` named after your platform.
2. **Add main implementation file(s)** such as `main.py` or `submit.py`.
3. **Add a `requirements.txt`** listing all dependencies for your platform kit.
4. **Update or create a `README.md`** in the platform directory. Include:
    - Platform comparison
    - Setup and usage instructions
    - Methodology and rationale
5. **Verify the code runs end-to-end** or compiles without errors.

_Example structure:_
```
explorations/build_on_both/myPlatform/
    main.py
    submit.py
    requirements.txt
    README.md
```

### Platform Starter Kit Extension
**Trigger:** When you want to add new features, models, or validation utilities to an existing platform starter kit.  
**Command:** `/extend-platform-starter`

1. **Edit or upgrade main implementation file(s)** (e.g., `main.py`, `submit.py`) in the relevant platform directory.
2. **Add new utility/helper files** as needed (e.g., `streamingAttacker.py`, `cv.py`).
3. **Update `requirements.txt`** if new dependencies are introduced.
4. **Update the platform's `README.md`** to document new features or changes.
5. **Verify all files run or compile** as expected.

_Example addition:_
```python
# explorations/build_on_both/myPlatform/streamingAttacker.py
def streaming_attack(...):
    ...
```

## Testing Patterns

- **Framework:** Unknown (not explicitly detected).
- **Test File Pattern:** Test files are named with the pattern `*.test.*`.
- **Location:** Test files are typically placed alongside implementation files.
- **Example:**
  ```
  main.test.py
  streamingAttacker.test.py
  ```

## Commands

| Command                  | Purpose                                                      |
|--------------------------|--------------------------------------------------------------|
| /new-platform-starter    | Scaffold a new platform starter kit                          |
| /extend-platform-starter | Add features or utilities to an existing platform starter kit |
```
