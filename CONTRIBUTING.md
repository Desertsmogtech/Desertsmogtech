# Contributing to Desertsmogtech

Welcome, builder of seamed systems! This repository welcomes contributors who value humility, clarity, and legacy-grade infrastructure. Whether you're debugging, curating datasets, or riffing on metaphors—your work here will be part of a teachable, auditable legacy.

## Our Philosophy

Every commit is a legacy artifact. Every friction point is a teachable moment.

We build infrastructure with:
- **Visible joins** — code that shows how it's stitched together
- **Teachable welds** — commits that explain *why*, not just *what*
- **Auditable clarity** — provenance and contributor trails preserved

---

## Getting Started

### 1. Fork & Clone
```bash
git clone https://github.com/YOUR-USERNAME/desertsmogtech.git
cd desertsmogtech
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names: `feature/signalscope-module`, `fix/recovery-kit-docs`, `lore/contributor-stories`.

### 3. Set Up Your Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 4. Make Your Changes
Follow the coding standards below. Keep commits small and focused.

### 5. Run Tests & Linting
```bash
pytest tests/ -v --cov=src
black src tests
isort src tests
flake8 src tests
```

All checks must pass before opening a PR.

### 6. Submit a Pull Request
- Link your PR to relevant issues or project goals
- Write a clear, teachable commit message (see below)
- Expect collaborative feedback—this is a learning space

---

## Coding Standards

### Python Style
- **Version**: Python 3.10+
- **Format**: [Black](https://github.com/psf/black) (enforced via CI)
- **Import sorting**: [isort](https://pycqa.github.io/isort/) (enforced via CI)
- **Linting**: [flake8](https://flake8.pycqa.org/) (enforced via CI)

### Code Quality
- **Type hints** — required for all function signatures
  ```python
  def modular_function(input_data: dict, retries: int = 3) -> str:
      """Transform input with auditable clarity."""
  ```

- **Docstrings** — Google-style for all public functions/classes
  ```python
  def process_module(config: dict) -> bool:
      """Process a module configuration with validation.
      
      Args:
          config: Module configuration dictionary.
      
      Returns:
          bool: True if successful, False otherwise.
      
      Raises:
          ValueError: If config is missing required keys.
      """
  ```

- **Tests** — pytest required for all new code
  - Minimum 80% coverage for new modules
  - Test file naming: `tests/test_<module_name>.py`

### Commit Messages

Every commit is a legacy artifact. Write teachable messages:

```
[category] Brief, clear summary (50 chars max)

Explain what changed and *why*. Reference issues (#123).
Show the joins. Help the next contributor understand the weld.

- Use bullet points for multiple changes
- Keep lines under 72 characters
- Sign off with your name (git commit -s)
```

**Categories**:
- `[feat]` — new feature or module
- `[fix]` — bug fix or recovery
- `[docs]` — documentation, lore, or scaffolding
- `[test]` — test additions or improvements
- `[refactor]` — code restructuring without behavior change
- `[chore]` — build config, dependencies, or CI updates
- `[lore]` — narrative commits, evolution tracking, contributor stories

**Example**:
```
[feat] Add modular observability toolkit (SignalScope)

Implement core signal processing with pluggable backends.
Includes audit trails for provider selection and metric validation.

Resolves #42. Addresses short-term goal: "Modularize scattered work"

- signal_processor.py: Core signal handling
- backends/: Pluggable observability providers
- tests/: Full pytest coverage with fixture mocks
```

---

## Modules & Architecture

### PMA (Personal Memory Architecture)
The repo is organized around modular clarity:

- **`src/core/`** — Core transformers, utilities, and foundational modules
- **`src/modules/`** — Domain-specific packages (SignalScope, SecureListenerKit, etc.)
- **`PMA/`** — Framework documentation
  - `Recovery-Kit/` — Troubleshooting, overflow catchers, fallback logic
  - `Audit-Trail/` — Timestamped decisions and pivots
  - `Onboarding/` — Contributor guides and tone expectations
  - `Assistant-Memory/` — Session logs and evolution tracking
  - `Lore/` — Cosmic riffs, image prompts, mythologies

### Contributing to Specific Areas

**New Module?** Create a folder under `src/modules/<module_name>/` with:
```
src/modules/my_module/
├── __init__.py
├── core.py
├── utils.py
└── tests/
    └── test_core.py
```

**Documentation Improvement?** Update the relevant `PMA/` folder or add to root README.

**Recovery Logic?** Add to `PMA/Recovery-Kit/` with clear troubleshooting steps and lore.

**Evolution Tracking?** Update `PMA/Audit-Trail/` with timestamped decisions.

---

## Testing

We use [pytest](https://pytest.org/) for all testing.

### Running Tests
```bash
pytest tests/ -v                          # Run all tests with verbose output
pytest tests/ --cov=src                   # Run with coverage report
pytest tests/test_specific.py -k "test_name"  # Run specific test
```

### Writing Tests
```python
import pytest
from src.core.transformers import workflow_transformer

def test_workflow_transformation():
    """Test that workflow transforms correctly with valid input."""
    input_data = {"name": "test", "steps": []}
    result = workflow_transformer.transform(input_data)
    assert result["name"] == "test"
    assert isinstance(result["steps"], list)

@pytest.fixture
def sample_config():
    """Fixture: provide sample configuration."""
    return {"retry": 3, "timeout": 30}

def test_with_fixture(sample_config):
    """Test using fixture."""
    assert sample_config["retry"] == 3
```

**Coverage Requirements**:
- New code should have ≥80% coverage
- Aim for high coverage to catch regressions early

---

## Reporting Issues

Found a bug? Open an **Issue** with:
- Clear title and description
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (Python version, OS, etc.)

Label it appropriately:
- `bug` — something is broken
- `enhancement` — feature request
- `documentation` — docs need updating
- `recovery` — friction point, recovery opportunity

---

## Code Review & Feedback

**Be kind, be clear, be teachable.**

- Reviews are conversations, not judgments
- Ask questions to understand intent
- Suggest improvements respectfully
- Every friction point is a chance to document better

We aim for:
- ✅ Passes all CI checks
- ✅ Tests included and passing
- ✅ Docstrings and type hints
- ✅ Commit messages explain the *why*
- ✅ No unrelated changes in one PR

---

## Onboarding Resources

New to the repo? Start here:

1. **PMA/Onboarding/** — Read contributor guides
2. **PMA/Assistant-Memory/** — Understand tone and evolution expectations
3. **PMA/Recovery-Kit/** — Troubleshoot common friction points
4. **PMA/Audit-Trail/** — See how decisions are tracked
5. **PMA/Lore/** — Understand the mythologies and vision

---

## Questions?

- Email: **smogmasterjv@gmail.com**
- Open an issue with `[question]` tag
- Check existing issues for similar questions

---

## Code of Conduct

Be respectful, humble, and honest. We welcome all backgrounds and experience levels. Harassment, discrimination, or bad faith are not tolerated.

---

**Every commit is a legacy artifact. Let's build infrastructure that endures—with seams that show, and stories that teach.**

Welcome aboard! 🌌
