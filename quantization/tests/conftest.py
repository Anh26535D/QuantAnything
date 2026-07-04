import sys, os
# Ensure the project root is on sys.path for tests
project_root = os.path.abspath(os.path.join(__file__, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
