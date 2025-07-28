import ast
import inspect
import re
import textwrap
from typing import Type

import factors
from factors import FactorBase  # 按你项目的真实路径来

def _escape_triple_quotes(s: str) -> str:
    return s.replace('"""', '\\"\\"\\"')

def _build_docstring(desc: str, indent: str) -> str:
    """
    生成你想要的样式：
        4格缩进的 \"\"\" 开头
        每一行内容同样 4格缩进
        4格缩进的 \"\"\" 结尾
    """
    desc = textwrap.dedent(desc.rstrip("\n"))
    desc = _escape_triple_quotes(desc)
    lines = desc.splitlines()

    if not lines:
        return f'{indent}"""\n{indent}"""\n'

    # 第一行放在三引号之后，同一行展示
    
    first = lines[0]
    buf = [f'{indent}"""']
    buf.append(f'{indent}{first}')
    for line in lines[1:]:
        buf.append(f'{indent}{line}')
    buf.append(f'{indent}"""')
    return "\n".join(buf) + "\n"

def update_docstrings(replace_existing: bool = False, indent_size: int = 4):
    for _, cls in inspect.getmembers(factors, inspect.isclass):
        if not issubclass(cls, FactorBase) or cls is FactorBase:
            continue

        filename = inspect.getfile(cls)
        with open(filename, "r", encoding="utf-8") as f:
            code = f.read()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # 如果当前文件就已经坏了，先跳过
            continue

        # 找到当前类定义（同名的只处理第一个）
        target = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
                target = node
                break
        if target is None:
            continue

        # 已有 docstring 且不想覆盖 -> 跳过
        if ast.get_docstring(target, clean=False) and not replace_existing:
            continue

        # 计算缩进
        lines = code.splitlines(True)  # 保留行尾
        header_line_idx = target.lineno - 1
        header_line = lines[header_line_idx]
        m = re.match(r"(\s*)class\s", header_line)
        head_indent = m.group(1) if m else ""
        body_indent = head_indent + " " * indent_size

        # 组装 docstring
        desc = getattr(cls, "description", "") or ""
        doc = _build_docstring(desc, body_indent)

        # 如果类开头本来有空行，直接插在 header 下一行
        insert_at = header_line_idx + 1

        # 如果类前面带装饰器（@xxx），ast.lineno 会指向 decorator 的第一行；
        # 但我们是按 header_line_idx 来插入，OK 的（因为 header_line_idx 指的是真正的 class 那行）

        # 再检查一下：如果我们“以为没有 docstring”，但其实文件里 class 后第一条就是 docstring，
        # 说明 ast 里拿不到是因为某种格式问题；这里简单粗暴删掉它再写入
        # （也可以不写这段，遇到再加）
        first_body_line = lines[insert_at] if insert_at < len(lines) else ""
        if first_body_line.lstrip().startswith(("'''", '"""', 'r"""', "r'''", 'u"""', "u'''", 'b"""', "b'''")) and replace_existing:
            # 删除原 docstring（直到配对的三引号结束）
            i = insert_at
            quote = first_body_line.lstrip()[:3]
            triple = quote if quote in ("'''", '"""') else first_body_line.lstrip()[1:4]
            # 简单向下找结束（不处理嵌套 triple quotes 的极端情况）
            while i < len(lines):
                if triple in lines[i]:
                    i += 1
                    break
                i += 1
            del lines[insert_at:i]

        # 插入新的
        lines.insert(insert_at, doc)

        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"Updated {cls.__name__} in {filename}")

if __name__ == "__main__":
    # 默认：已有 docstring 的类跳过
    update_docstrings(replace_existing=False, indent_size=4)
