"""Build student and solution notebooks from the percent-format sources in src/.

Source conventions (src/dayN.py):
  # %% [markdown]          a markdown cell; following '# ' comment lines are its text
  # %% [markdown] solution a markdown cell that appears ONLY in the solution notebook
  # %%                     a code cell
  # BEGIN SOLUTION ... # END SOLUTION
                           inside a code cell: replaced in the student notebook by a TODO stub

Usage: python build_notebooks.py [--execute]   (from the labs/ directory)
"""
import re
import subprocess
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
CELL = re.compile(r"^# %%(?: \[markdown\])?(?: solution)?\s*$")


def parse(path):
    cells, kind, solution_only, lines = [], None, False, []

    def flush():
        if kind is not None:
            text = "\n".join(lines).strip("\n")
            if text:
                cells.append((kind, solution_only, text))

    for line in path.read_text().splitlines():
        if CELL.match(line):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            solution_only, lines = line.rstrip().endswith("solution"), []
        elif kind == "markdown":
            lines.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
        elif kind is not None:
            lines.append(line)
    flush()
    return cells


def student_code(text):
    out, skipping = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "# BEGIN SOLUTION":
            indent = line[: len(line) - len(line.lstrip())]
            out += [f"{indent}# TODO: your code here", f"{indent}..."]
            skipping = True
        elif stripped == "# END SOLUTION":
            skipping = False
        elif not skipping:
            out.append(line)
    return "\n".join(out)


def solution_code(text):
    return "\n".join(l for l in text.splitlines() if l.strip() not in ("# BEGIN SOLUTION", "# END SOLUTION"))


def build(src, execute):
    cells = parse(src)
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
    student = new_notebook(metadata=meta)
    solution = new_notebook(metadata=meta)
    for kind, sol_only, text in cells:
        if kind == "markdown":
            solution.cells.append(new_markdown_cell(text))
            if not sol_only:
                student.cells.append(new_markdown_cell(text))
        else:
            solution.cells.append(new_code_cell(solution_code(text)))
            student.cells.append(new_code_cell(student_code(text)))
    out_student = HERE / f"{src.stem}.ipynb"
    out_solution = HERE / "solutions" / f"{src.stem}_solution.ipynb"
    out_solution.parent.mkdir(exist_ok=True)
    nbformat.write(student, out_student)
    nbformat.write(solution, out_solution)
    print(f"{src.name}: {len(student.cells)} student cells, {len(solution.cells)} solution cells")
    if execute:
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--inplace", "--ExecutePreprocessor.timeout=300", str(out_solution)],
                       check=True, cwd=HERE)
        print(f"  executed {out_solution.relative_to(HERE)}")


if __name__ == "__main__":
    for src in sorted((HERE / "src").glob("day*.py")):
        build(src, "--execute" in sys.argv)
