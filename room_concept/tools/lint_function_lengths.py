#!/usr/bin/env python3
"""Report C++ function bodies longer than a limit (refactor guardrail, docs/refactor_proposal_2026-10-13.md §5).

Usage:  python3 -I tools/lint_function_lengths.py MAX_LINES FILE...
Prints one line per offender:  <lines> <file>:<function>
A heuristic brace-depth scan, not a parser: comments and string/char literals are blanked, then every
`{` that opens while NOT already inside a function body, and whose header (the text since the previous
`;`, `{` or `}`) contains `(` and does not start with a control keyword, is taken as a function body.
Lambdas and blocks nested inside a function are counted as part of that function. Overloads share a
key (file:name); the longest one is reported.
"""
import re
import sys

CONTROL = re.compile(r'^\s*(if|for|while|switch|catch|do|else|try|return|case|default)\b')
IDENT = re.compile(r'([A-Za-z_~][A-Za-z_0-9:<>~]*)\s*$')


def blank_comments_and_strings(text: str) -> str:
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i)); i = j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r'[^\n]', ' ', text[i:j])); i = j
        elif text.startswith('R"', i):                       # raw string R"delim( ... )delim"
            m = re.match(r'R"([^(\s]*)\(', text[i:])
            if m:
                end = text.find(')' + m.group(1) + '"', i)
                j = n if end < 0 else end + len(m.group(1)) + 2
                out.append(re.sub(r'[^\n]', ' ', text[i:j])); i = j
            else:
                out.append(c); i += 1
        elif c in '"\'':
            if c == "'" and i > 0 and text[i - 1].isalnum():   # digit separator 1'000'000
                out.append(' '); i += 1; continue
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == '\\' else 1
            j = min(j + 1, n)
            out.append(re.sub(r'[^\n]', ' ', text[i:j])); i = j
        else:
            out.append(c); i += 1
    return ''.join(out)


def function_lengths(path: str):
    with open(path, encoding='utf-8', errors='replace') as f:
        text = blank_comments_and_strings(f.read())
    # drop preprocessor lines (keep the newline so line numbers hold)
    text = re.sub(r'(?m)^[ \t]*#.*$', '', text)
    result = {}
    depth, func_depth, func_name, func_start = 0, None, None, 0
    header_start = 0
    line = 1
    for i, c in enumerate(text):
        if c == '\n':
            line += 1
        elif c == '{':
            if func_depth is None:
                header = text[header_start:i]
                if '(' in header and not CONTROL.match(header) and '=' not in header.split('(')[0]:
                    name_m = IDENT.search(header.split('(')[0])
                    if name_m:
                        func_depth, func_name = depth, name_m.group(1)
                        sig_idx = header_start + len(header) - len(header.lstrip())   # first char of the signature
                        func_start = line - text.count('\n', sig_idx, i)
            depth += 1
            header_start = i + 1
        elif c == '}':
            depth -= 1
            if func_depth is not None and depth == func_depth:
                n = line - func_start + 1
                key = func_name
                result[key] = max(result.get(key, 0), n)
                func_depth = None
            header_start = i + 1
        elif c == ';':
            header_start = i + 1
    return result


def main() -> int:
    limit = int(sys.argv[1])
    for path in sys.argv[2:]:
        for name, n in sorted(function_lengths(path).items(), key=lambda kv: -kv[1]):
            if n > limit:
                print(f'{n} {path}:{name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
