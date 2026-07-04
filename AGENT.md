# Python Rule

## Description
This rule establishes the primary coding conventions for Python code within this project to ensure consistency, readability, and maintainability. All Python code should adhere to these principles, which are largely based on the PEP 8 style guide.

## !!!IMPORTANT
- Use `uv` as Python package management tool
- Use `git` as Version management tool

## 1. Naming Conventions
- **Variables, Functions, and Methods**: Use `snake_case` (e.g., `my_variable`, `calculate_sum()`).
- **Classes**: Use `PascalCase` (e.g., `MyClass`).
- **Constants**: Use `UPPER_SNAKE_CASE` (e.g., `MAX_CONNECTIONS`).
- **Modules**: Use short, `snake_case` names.

## 2. Code Layout
- **Indentation**: Use 4 spaces per indentation level. Do not use tabs.
- **Line Length**: Maximum line length is 79 characters for code and 72 for docstrings.
- **Blank Lines**:
    - Use two blank lines to separate top-level functions and class definitions.
    - Use one blank line to separate method definitions inside a class.
    - Use blank lines sparingly inside functions to show logical sections.

## 3. Imports
- Imports should be at the top of the file, just after any module comments and docstrings.
- Group imports in the following order, with a blank line between each group:
    1. Standard library imports (e.g., `os`, `sys`).
    2. Third-party library imports (e.g., `requests`, `numpy`).
    3. Local application/library specific imports.
- Avoid wildcard imports (`from module import *`).

## 4. Comments and Docstrings
- **Docstrings**: All public modules, functions, classes, and methods must have docstrings. Use Google-style docstrings.
- **Comments**: Use comments to explain non-obvious code. Keep them concise and up-to-date. Inline comments should be used sparingly.

## When to apply
- When writing or modifying any Python code (`.py` files).
- During code reviews to enforce style consistency.

## When not to apply
- When working with third-party libraries that have their own established conventions.
- In specific cases where deviating from these conventions improves readability or is necessary for compatibility, with justification provided in comments.