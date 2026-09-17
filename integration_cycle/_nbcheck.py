import json, ast, builtins, sys

def collect(tree):
    """(definidos_en_la_celda, leidos_en_top_level, leidos_dentro_de_funciones)"""
    d=set(); top=set(); infn=set()
    for n in ast.walk(tree):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): d.add(n.name)
        elif isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store): d.add(n.id)
        elif isinstance(n,(ast.Import,ast.ImportFrom)):
            for a in n.names: d.add((a.asname or a.name).split('.')[0])
        elif isinstance(n,ast.arg): d.add(n.arg)
        elif isinstance(n,ast.ExceptHandler) and n.name: d.add(n.name)
    def walk(node, inside):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): inside=True
        if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load):
            (infn if inside else top).add(node.id)
        for ch in ast.iter_child_nodes(node): walk(ch, inside)
    walk(tree, False)
    return d, top, infn

for NB in sys.argv[1:]:
    nb=json.load(open(f'integration_cycle/{NB}.ipynb',encoding='utf-8'))
    cells=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    known=set(dir(builtins)); errs=[]; warns=[]
    for ci,src in enumerate(cells):
        try: tree=ast.parse(src)
        except SyntaxError as e: errs.append((ci,f'SyntaxError: {e.msg}')); continue
        d,top,infn=collect(tree)
        known|=d                                  # lo de esta celda ya cuenta
        for n in sorted(top-known): errs.append((ci,n))
        # Decorador huerfano: insertar codigo justo encima de un `def` reasigna su
        # decorador a otra funcion, en silencio. Paso @torch.no_grad() de
        # encoder_parity_r2 a vendi_score y el resultado fue un RuntimeError a mitad
        # de corrida. Se marca cualquier decorador conocido sobre una funcion que
        # "no deberia" llevarlo por su nombre.
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                for dec in node.decorator_list:
                    src_dec=ast.unparse(dec) if hasattr(ast,'unparse') else ''
                    if 'no_grad' in src_dec and ('score' in node.name or 'vendi' in node.name):
                        warns.append((ci,f'decorador {src_dec} sobre {node.name} (?)'))
        for n in sorted(infn-known): warns.append((ci,n))
    print(f'{NB[:2]}:')
    print(f'   ERRORES (top-level, nombre inexistente): {sorted(set(n for _,n in errs)) or "ninguno"}')
    print(f'   AVISOS  (en cuerpo de función, definido más tarde): {sorted(set(n for _,n in warns)) or "ninguno"}')
