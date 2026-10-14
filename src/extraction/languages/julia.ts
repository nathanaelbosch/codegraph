import type { Node as SyntaxNode } from 'web-tree-sitter';
import { getNodeText, getChildByField, getPrecedingDocstring } from '../tree-sitter-helpers';
import type { LanguageExtractor, ExtractorContext } from '../tree-sitter-types';
import type { NodeKind } from '../../types';

/**
 * Julia extractor.
 *
 * The tree-sitter-julia grammar names almost no fields (a function's name sits
 * inside `signature > call_expression > identifier`, a struct's inside
 * `type_head`), and half of Julia's definitions are assignments
 * (`area(p) = …`), so every symbol arrives through the visitNode hook rather
 * than the node-type lists — the same approach as R:
 *
 *   - modules:    `module M … end` / `baremodule` → `module`, scoping what it holds.
 *   - functions:  `function f(…) … end`, the short form `f(x) = …`,
 *                 `function f end`, `name = (x) -> …`, and `macro m(…) … end`
 *                 (named `@m`, as its call sites spell it). Each method of a
 *                 generic function is its own `function` node. A method added
 *                 to another module's function (`Base.show(io, x) = …`) is
 *                 named by the function (`show`); its signature keeps the
 *                 owner. A functor `(p::Poly)(x) = …` is named `(::Poly)`.
 *   - types:      `struct` / `mutable struct` → `struct` with `field`s and
 *                 inner constructors as methods; `abstract type` → `interface`;
 *                 `primitive type` → `struct`. `A <: B` records the supertype:
 *                 `implements` from a concrete type (Julia's concrete types
 *                 cannot be subtyped, so their supertype is always abstract)
 *                 and `extends` from an abstract one.
 *   - enums:      `@enum Color red green blue` (and the `begin … end` form).
 *   - variables:  module-level assignments; `const` ones are constants, or
 *                 type aliases when the value is a type (`const V = Vector{T}`).
 *   - imports:    `using` / `import` (one import per module path), and
 *                 `include("file.jl")`, which pulls another file into the
 *                 module and is how Julia packages are split across files.
 *   - calls:      `f(x)`, `M.f(x)`, broadcast `f.(x)`, constructor calls
 *                 `T{P}(x)`, pipes `x |> f`, and macro calls `@m …`.
 *   - docstrings: the string literal written right before a definition, or
 *                 `@doc "…" def`; `#` comments above it otherwise.
 *
 * Bodies are walked with ctx.visitNode (not visitFunctionBody), because the
 * body walker doesn't consult this hook and nested definitions, calls and
 * imports all need it.
 */

/** Module-level containers whose assignments define globals. */
const MODULE_LEVEL_PARENTS = new Set(['source_file']);
/** Statement wrappers an assignment may sit in and still be module-level. */
const DECLARATION_WRAPPERS = new Set(['const_statement', 'global_statement']);
/**
 * Where an assignment is a statement (so `f(x) = …` defines a method) rather
 * than a keyword argument or named-tuple entry (`f(x; k = v)`).
 */
const STATEMENT_PARENTS = new Set([
  'source_file', 'block', 'compound_statement', 'const_statement', 'global_statement',
  'local_statement', 'macro_argument_list', 'if_statement', 'elseif_clause', 'else_clause',
  'let_statement', 'for_statement', 'while_statement', 'try_statement', 'catch_clause',
  'finally_clause', 'quote_statement',
]);
/** Calls that pull another file into the enclosing module. */
const INCLUDE_FNS = new Set(['include']);
/** Pipe operators whose right side is a function applied to the left side. */
const PIPE_OPERATORS = new Set(['|>', '.|>']);
/** Value nodes that make a `const X = …` a type alias (`Vector{Float64}`, `Union{A, B}`). */
const TYPE_VALUE_TYPES = new Set(['parametrized_type_expression', 'curly_expression']);
/**
 * Macros of Base and Core that annotate code rather than call anything a
 * project defines — not worth a reference at every use.
 */
const ANNOTATION_MACROS = /^@(?:doc|inline|noinline|inbounds|propagate_inbounds|views|view|assert|show|info|warn|error|debug|time|elapsed|kwdef|generated|eval|static|simd|fastmath|nospecialize|specialize|__dot__|\.)$/;

/** Named children of a node. */
function named(node: SyntaxNode): SyntaxNode[] {
  const out: SyntaxNode[] = [];
  for (let i = 0; i < node.namedChildCount; i++) {
    const c = node.namedChild(i);
    if (c) out.push(c);
  }
  return out;
}

function firstNamed(node: SyntaxNode, type?: string): SyntaxNode | null {
  for (let i = 0; i < node.namedChildCount; i++) {
    const c = node.namedChild(i);
    if (c && (!type || c.type === type)) return c;
  }
  return null;
}

/** Whether an assignment / declaration sits directly at module level. */
function isModuleLevel(node: SyntaxNode): boolean {
  let p = node.parent;
  while (p && DECLARATION_WRAPPERS.has(p.type)) p = p.parent;
  if (!p) return false;
  if (MODULE_LEVEL_PARENTS.has(p.type)) return true;
  return p.type === 'block' && p.parent?.type === 'module_definition';
}

/** Whether the node is inside a `const` declaration. */
function isConstDeclaration(node: SyntaxNode): boolean {
  let p = node.parent;
  while (p && DECLARATION_WRAPPERS.has(p.type)) {
    if (p.type === 'const_statement') return true;
    p = p.parent;
  }
  return false;
}

/** Text of a quoted operator name — `:+` / `:(==)` → `+` / `==`. */
function quotedName(node: SyntaxNode, source: string): string {
  return getNodeText(node, source).replace(/^:\(?/, '').replace(/\)$/, '').trim();
}

/**
 * The callee written at the head of a call, as a reference name: `f`, `M.f`,
 * `A.B.f`, `T` for `T{P}(…)`. Null when the callee has no static name
 * (`f(x)(y)`, `xs[i](y)`, `(x -> x)(1)`).
 */
function calleeName(callee: SyntaxNode, source: string): string | null {
  switch (callee.type) {
    case 'identifier':
      return getNodeText(callee, source);
    case 'operator':
      return getNodeText(callee, source);
    case 'parametrized_type_expression': {
      const head = firstNamed(callee);
      return head ? calleeName(head, source) : null;
    }
    case 'field_expression': {
      const value = getChildByField(callee, 'value');
      const parts = named(callee);
      const member = parts[parts.length - 1];
      if (!value || !member || member === value) return null;
      const owner = calleeName(value, source);
      const memberName =
        member.type === 'identifier' ? getNodeText(member, source)
          : member.type === 'quote_expression' ? quotedName(member, source)
            : null;
      if (!owner || !memberName) return null;
      return `${owner}.${memberName}`;
    }
    default:
      return null;
  }
}

/** A method's head, read off its signature. */
interface JuliaSignature {
  /** The function's name, without its owner. */
  name: string;
  /** The module the method extends (`Base` for `Base.show(…)`). */
  owner?: string;
  /** The call node holding the parameter list (absent for `function f end`). */
  call?: SyntaxNode;
  /** The declared return type, when written (`f(x)::Int`). */
  returnType?: string;
}

/**
 * Read a method head: `f(x)`, `f(x)::T`, `f(x) where T`, `M.f(x)`, `M.:+(a, b)`,
 * `+(a, b)`, `T{P}(x)`, `(p::Poly)(x)`, or a bare `f` (`function f end`).
 */
function readSignature(head: SyntaxNode, source: string): JuliaSignature | null {
  let node: SyntaxNode | null = head;
  let returnType: string | undefined;
  // Peel `where` clauses and the return-type annotation, in any order.
  for (let guard = 0; node && guard < 8; guard++) {
    if (node.type === 'where_expression') {
      node = firstNamed(node);
    } else if (node.type === 'typed_expression' && firstNamed(node)?.type === 'call_expression') {
      const parts = named(node);
      returnType = parts[1] ? getNodeText(parts[1], source) : undefined;
      node = parts[0] ?? null;
    } else {
      break;
    }
  }
  if (!node) return null;
  if (node.type === 'identifier') return { name: getNodeText(node, source) };
  if (node.type === 'field_expression' && !returnType) {
    // `function Base.foo end`
    const full = calleeName(node, source);
    if (!full) return null;
    const dot = full.lastIndexOf('.');
    return { name: full.slice(dot + 1), owner: full.slice(0, dot) };
  }
  if (node.type !== 'call_expression') return null;
  const callee = firstNamed(node);
  if (!callee) return null;
  if (callee.type === 'parenthesized_expression') {
    // Functor: `(p::Poly)(x)` / `(::Poly)(x)` — the type is the callable.
    const inner = firstNamed(callee);
    const typeNode = inner?.type === 'typed_expression' ? named(inner).pop() : inner?.type === 'unary_typed_expression' ? firstNamed(inner) : null;
    const typeName = typeNode ? calleeName(typeNode, source) : null;
    return typeName ? { name: `(::${typeName})`, call: node, returnType } : null;
  }
  const full = calleeName(callee, source);
  if (!full) return null;
  const dot = callee.type === 'field_expression' ? full.lastIndexOf('.') : -1;
  return dot > 0
    ? { name: full.slice(dot + 1), owner: full.slice(0, dot), call: node, returnType }
    : { name: full, call: node, returnType };
}

/** Whether an assignment's left side is a method head (`f(x) = …`). */
function isMethodHead(lhs: SyntaxNode): boolean {
  let node: SyntaxNode | null = lhs;
  for (let guard = 0; node && guard < 8; guard++) {
    if (node.type === 'call_expression') return true;
    if (node.type === 'where_expression' || node.type === 'typed_expression') node = firstNamed(node);
    else return false;
  }
  return false;
}

/** Prose of a string literal written as a docstring. */
function docstringText(lit: SyntaxNode, source: string): string | undefined {
  const raw = getNodeText(lit, source);
  const body = raw.startsWith('"""') ? raw.slice(3, -3) : raw.startsWith('"') ? raw.slice(1, -1) : raw;
  const lines = body.replace(/^\n/, '').split('\n');
  const indents = lines.filter((l) => l.trim()).map((l) => /^\s*/.exec(l)![0].length);
  const cut = indents.length ? Math.min(...indents) : 0;
  const text = lines.map((l) => l.slice(cut)).join('\n').trim();
  return text || undefined;
}

/** Whether a string-ish node can be a docstring (`"…"`, `"""…"""`, `raw"…"`, `doc"…"`). */
function isDocLiteral(node: SyntaxNode | null): node is SyntaxNode {
  return !!node && (node.type === 'string_literal' || node.type === 'prefixed_string_literal');
}

/**
 * The statement a definition stands as among its siblings: a macro-wrapped
 * definition (`@inline function …`) is documented before the macro call.
 */
function statementOf(node: SyntaxNode): SyntaxNode {
  let s = node;
  while (s.parent?.type === 'macro_argument_list' && s.parent.parent?.type === 'macrocall_expression') {
    s = s.parent.parent;
  }
  while (s.parent && DECLARATION_WRAPPERS.has(s.parent.type)) s = s.parent;
  return s;
}

/** The docstring of a definition: a string literal right before it, `@doc`, or `#` comments. */
function docstringOf(node: SyntaxNode, source: string): string | undefined {
  // `@doc "…" def` — the string is the definition's sibling in the macro's arguments.
  const prevInArgs = node.parent?.type === 'macro_argument_list' ? node.previousNamedSibling : null;
  if (isDocLiteral(prevInArgs)) return docstringText(prevInArgs, source);
  const stmt = statementOf(node);
  const prev = stmt.previousNamedSibling;
  // A docstring sits on the line(s) right above the definition.
  if (isDocLiteral(prev) && stmt.startPosition.row - prev.endPosition.row <= 1) {
    return docstringText(prev, source);
  }
  return getPrecedingDocstring(stmt, source);
}

/** Text of a signature, whitespace-collapsed and capped. */
function signatureText(node: SyntaxNode, source: string): string {
  const text = getNodeText(node, source).replace(/\s+/g, ' ').trim();
  return text.length > 300 ? `${text.slice(0, 297)}...` : text;
}

/** Read a type head: name, and the supertype after `<:` when written. */
function readTypeHead(head: SyntaxNode | null, source: string): { name: string; supertype?: SyntaxNode } | null {
  if (!head) return null;
  let node: SyntaxNode | null = head.type === 'type_head' ? firstNamed(head) : head;
  // `struct S{T} <: A where {T}` — the where clause wraps the whole head.
  while (node?.type === 'where_expression') node = firstNamed(node);
  let supertype: SyntaxNode | undefined;
  if (node?.type === 'binary_expression') {
    const parts = named(node);
    const op = parts.find((p) => p.type === 'operator');
    if (op && getNodeText(op, source) === '<:') {
      node = parts[0] ?? null;
      supertype = parts[parts.length - 1];
    }
  }
  if (node?.type === 'parametrized_type_expression') node = firstNamed(node);
  if (node?.type !== 'identifier') return null;
  return { name: getNodeText(node, source), supertype };
}

/** The name of a type reference node (`AbstractShape`, `Polygon{T}` → `Polygon`, `M.T`). */
function typeRefName(node: SyntaxNode, source: string): string | null {
  return calleeName(node, source);
}

/** Per parse tree: the names each module (by start offset) exports or declares `public`. */
const MODULE_EXPORTS = new WeakMap<object, Map<number, Set<string>>>();

/** Whether the module a definition is written in exports (or declares `public`) its name. */
function isExportedName(def: SyntaxNode, name: string, source: string): boolean {
  let mod = def.parent;
  while (mod && mod.type !== 'module_definition') mod = mod.parent;
  if (!mod) return false;
  let perTree = MODULE_EXPORTS.get(def.tree);
  if (!perTree) MODULE_EXPORTS.set(def.tree, (perTree = new Map()));
  let names = perTree.get(mod.startIndex);
  if (!names) {
    names = new Set();
    const body = firstNamed(mod, 'block');
    for (const s of body ? named(body) : []) {
      if (s.type !== 'export_statement' && s.type !== 'public_statement') continue;
      for (const e of named(s)) names.add(getNodeText(e, source).replace(/\s+/g, ''));
    }
    perTree.set(mod.startIndex, names);
  }
  return names.has(name);
}

function topScope(ctx: ExtractorContext): string | undefined {
  return ctx.nodeStack[ctx.nodeStack.length - 1];
}

function currentScopeKind(ctx: ExtractorContext): NodeKind | undefined {
  const id = topScope(ctx);
  if (!id) return undefined;
  for (let i = ctx.nodes.length - 1; i >= 0; i--) {
    if (ctx.nodes[i]!.id === id) return ctx.nodes[i]!.kind;
  }
  return id.startsWith('file:') ? 'file' : undefined;
}

function addRef(ctx: ExtractorContext, name: string, kind: 'calls' | 'imports' | 'extends' | 'implements' | 'references', at: SyntaxNode, from?: string): void {
  const fromNodeId = from ?? topScope(ctx);
  if (!fromNodeId) return;
  ctx.addUnresolvedReference({
    fromNodeId,
    referenceName: name,
    referenceKind: kind,
    line: at.startPosition.row + 1,
    column: at.startPosition.column,
  });
}

/** Visit every named child of a node through the hook-aware walker. */
function visitChildren(node: SyntaxNode, ctx: ExtractorContext, skip?: SyntaxNode | null): void {
  for (const c of named(node)) {
    if (c !== skip) ctx.visitNode(c);
  }
}

/** Default-value expressions of a parameter list (their calls run in the method). */
function visitParameterDefaults(call: SyntaxNode | undefined, ctx: ExtractorContext): void {
  const args = call ? firstNamed(call, 'argument_list') : null;
  if (!args) return;
  for (const p of named(args)) {
    if (p.type === 'assignment') {
      const value = named(p).pop();
      if (value && value.type !== 'operator') ctx.visitNode(value);
    }
  }
}

/**
 * Types too common to be worth a reference from every method that takes one —
 * Base's and Core's, which no project declares.
 */
const BUILTIN_TYPES = new Set([
  'Any', 'Nothing', 'Missing', 'Bool', 'Char', 'String', 'Symbol', 'Number', 'Real', 'Integer', 'Signed',
  'Unsigned', 'AbstractFloat', 'Int', 'Int8', 'Int16', 'Int32', 'Int64', 'Int128', 'UInt', 'UInt8', 'UInt16',
  'UInt32', 'UInt64', 'UInt128', 'Float16', 'Float32', 'Float64', 'BigInt', 'BigFloat', 'Complex', 'ComplexF64',
  'ComplexF32', 'Rational', 'AbstractString', 'SubString', 'AbstractChar', 'Function', 'Type', 'DataType',
  'Union', 'UnionAll', 'Tuple', 'NTuple', 'NamedTuple', 'Vararg', 'Val', 'Array', 'Vector', 'Matrix',
  'AbstractArray', 'AbstractVector', 'AbstractMatrix', 'AbstractVecOrMat', 'DenseArray', 'DenseVector',
  'DenseMatrix', 'AbstractRange', 'UnitRange', 'StepRange', 'Dict', 'AbstractDict', 'Set', 'AbstractSet',
  'Pair', 'IO', 'IOBuffer', 'Module', 'Expr', 'Exception', 'Ref', 'Ptr', 'Cvoid', 'Cint', 'Cstring',
]);

/** The type names an annotation mentions: `Vector{Point{T}}` → Vector, Point, T. */
function annotationTypeNames(node: SyntaxNode, out: SyntaxNode[] = []): SyntaxNode[] {
  if (node.type === 'identifier' || node.type === 'field_expression') out.push(node);
  else for (const c of named(node)) annotationTypeNames(c, out);
  return out;
}

/** Type variables a method head declares: `where {T, S<:Real}` → T, S. */
function whereVariables(head: SyntaxNode | null, source: string): Set<string> {
  const vars = new Set<string>();
  for (let w = head; w?.type === 'where_expression'; w = firstNamed(w)) {
    for (const clause of named(w).slice(1)) {
      const items = clause.type === 'curly_expression' ? named(clause) : [clause];
      for (const item of items) {
        const v = item.type === 'binary_expression' ? firstNamed(item) : item;
        if (v?.type === 'identifier') vars.add(getNodeText(v, source));
      }
    }
  }
  return vars;
}

/**
 * `references` from a method to the project types its parameters are
 * annotated with — in Julia, the methods that take a type are its interface,
 * so `area(c::Circle)` is part of what Circle's references show. Type
 * variables (`where T`) and Base's types are left out.
 */
function referenceParameterTypes(sig: JuliaSignature, sigNode: SyntaxNode, fromId: string, ctx: ExtractorContext): void {
  const args = sig.call ? firstNamed(sig.call, 'argument_list') : null;
  if (!args) return;
  const source = ctx.source;
  let head: SyntaxNode | null = sigNode.type === 'signature' ? firstNamed(sigNode) : sigNode;
  while (head?.type === 'typed_expression' && firstNamed(head)?.type !== 'call_expression') head = firstNamed(head);
  if (head?.type === 'typed_expression') head = firstNamed(head);
  const typeVars = whereVariables(head, source);
  const seen = new Set<string>();
  for (let p of named(args)) {
    if (p.type === 'assignment') p = firstNamed(p) ?? p;
    if (p.type === 'splat_expression') p = firstNamed(p) ?? p;
    const annotation = p.type === 'typed_expression' ? named(p)[1] : p.type === 'unary_typed_expression' ? firstNamed(p) : null;
    if (!annotation) continue;
    for (const t of annotationTypeNames(annotation)) {
      const name = typeRefName(t, source);
      if (!name || seen.has(name) || typeVars.has(name) || BUILTIN_TYPES.has(name.slice(name.lastIndexOf('.') + 1))) continue;
      seen.add(name);
      addRef(ctx, name, 'references', t, fromId);
    }
  }
}

/**
 * Create one method node and walk its body under it. `body` is the block of a
 * long-form definition or the right-hand side of a short-form one.
 */
function emitMethod(
  defNode: SyntaxNode,
  sig: JuliaSignature,
  sigNode: SyntaxNode,
  body: SyntaxNode | null,
  ctx: ExtractorContext,
  nameOverride?: string,
): void {
  const source = ctx.source;
  const scopeKind = currentScopeKind(ctx);
  // A definition inside a struct body is an inner constructor.
  const kind: NodeKind = scopeKind === 'struct' ? 'method' : 'function';
  const name = nameOverride ?? sig.name;
  const fn = ctx.createNode(kind, name, defNode, {
    signature: signatureText(sigNode, source),
    docstring: docstringOf(defNode, source),
    returnType: sig.returnType,
    isExported: isExportedName(defNode, name, source),
  });
  if (!fn) {
    if (body) ctx.visitNode(body);
    return;
  }
  referenceParameterTypes(sig, sigNode, fn.id, ctx);
  ctx.pushScope(fn.id);
  visitParameterDefaults(sig.call, ctx);
  if (body) ctx.visitNode(body);
  ctx.popScope();
}

/** `@enum Color red green blue` / `@enum Color::UInt8 begin red = 1 … end`. */
function emitEnum(macro: SyntaxNode, args: SyntaxNode, ctx: ExtractorContext): boolean {
  const source = ctx.source;
  const parts = named(args);
  const head = parts[0];
  if (!head) return false;
  const nameNode = head.type === 'typed_expression' ? firstNamed(head) : head;
  if (nameNode?.type !== 'identifier') return false;
  const name = getNodeText(nameNode, source);
  const en = ctx.createNode('enum', name, macro, {
    signature: signatureText(macro, source).slice(0, 120),
    docstring: docstringOf(macro, source),
    isExported: isExportedName(macro, name, source),
  });
  if (!en) return true;
  ctx.pushScope(en.id);
  const members: SyntaxNode[] = [];
  for (const p of parts.slice(1)) {
    if (p.type === 'compound_statement' || p.type === 'block') members.push(...named(p));
    else members.push(p);
  }
  for (const m of members) {
    const id = m.type === 'assignment' ? firstNamed(m) : m;
    if (id?.type === 'identifier') ctx.createNode('enum_member', getNodeText(id, source), m, {});
  }
  ctx.popScope();
  return true;
}

/** Create the `import` node(s) of a `using` / `import` statement. */
function emitImports(stmt: SyntaxNode, ctx: ExtractorContext): void {
  const source = ctx.source;
  const keyword = stmt.type === 'using_statement' ? 'using' : 'import';
  for (const clause of named(stmt)) {
    let pathNode: SyntaxNode | null = clause;
    let items = '';
    if (clause.type === 'selected_import') {
      const parts = named(clause);
      pathNode = parts[0] ?? null;
      items = parts.slice(1).map((p) => getNodeText(p, source)).join(', ');
    } else if (clause.type === 'import_alias') {
      pathNode = firstNamed(clause);
    }
    if (!pathNode) continue;
    // `..Parent.Sub` keeps its leading dots; the module path is the import's name.
    const modulePath = getNodeText(pathNode, source).replace(/\s+/g, '');
    if (!modulePath) continue;
    const imp = ctx.createNode('import', modulePath, clause, {
      signature: items ? `${keyword} ${modulePath}: ${items}` : `${keyword} ${getNodeText(clause, source).trim()}`,
    });
    if (imp) addRef(ctx, modulePath, 'imports', clause);
  }
}

/** A plain string literal's text, or null (interpolated strings have no static value). */
function plainString(node: SyntaxNode | null, source: string): string | null {
  if (node?.type !== 'string_literal' || named(node).some((c) => c.type !== 'content')) return null;
  return getNodeText(node, source).replace(/^"+|"+$/g, '');
}

/**
 * The path an `include` names, relative to the including file:
 * `include("file.jl")`, or `include(joinpath(@__DIR__, "sub", "file.jl"))`.
 */
function includedPath(call: SyntaxNode, source: string): string | null {
  const args = firstNamed(call, 'argument_list');
  const first = args ? firstNamed(args) : null;
  if (!first) return null;
  const literal = plainString(first, source);
  if (literal !== null) return literal;
  const callee = first.type === 'call_expression' ? firstNamed(first) : null;
  if (!callee || getNodeText(callee, source) !== 'joinpath') return null;
  const parts = named(firstNamed(first, 'argument_list') ?? first);
  if (parts[0] && getNodeText(parts[0], source).replace(/\s+/g, '') === '@__DIR__') parts.shift();
  const segments = parts.map((p) => plainString(p, source));
  return segments.length > 0 && segments.every((x) => x !== null) ? segments.join('/') : null;
}

export const juliaExtractor: LanguageExtractor = {
  // Handled in visitNode; listed so the function-as-value scan stops at a
  // nested definition instead of attributing its body to the enclosing scope.
  functionTypes: ['function_definition', 'macro_definition'],
  classTypes: [],
  methodTypes: [],
  interfaceTypes: [],
  structTypes: [],
  enumTypes: [],
  typeAliasTypes: [],
  importTypes: [],
  callTypes: [], // calls are recorded in visitNode, where the callee shapes are known
  variableTypes: [],
  nameField: 'name',
  bodyField: 'body',
  paramsField: 'parameters',

  visitNode: (node, ctx) => {
    // Everything this hook handles it walks itself, through ctx.visitNode
    // with the right scope pushed.
    const handled = visitJuliaNode(node, ctx);
    if (handled) ctx.markWalked(node);
    return handled;
  },
};

/** The hook proper: create what a Julia node defines and walk it. */
function visitJuliaNode(node: SyntaxNode, ctx: ExtractorContext): boolean {
  const source = ctx.source;

  switch (node.type) {
    case 'module_definition': {
      const nameNode = getChildByField(node, 'name') ?? firstNamed(node, 'identifier');
      const body = firstNamed(node, 'block');
      if (!nameNode) return false;
      const name = getNodeText(nameNode, source);
      const mod = ctx.createNode('module', name, node, {
        docstring: docstringOf(node, source),
        signature: `${source.slice(node.startIndex, nameNode.startIndex).trim()} ${name}`,
        isExported: true,
      });
      if (mod) ctx.pushScope(mod.id);
      if (body) ctx.visitNode(body);
      if (mod) ctx.popScope();
      return true;
    }

    case 'function_definition':
    case 'macro_definition': {
      const sigNode = firstNamed(node, 'signature');
      const head = sigNode ? firstNamed(sigNode) : null;
      const sig = head ? readSignature(head, source) : null;
      const body = firstNamed(node, 'block');
      if (!sig || !sigNode) {
        // Anonymous `function (x) … end` — its calls belong to the enclosing scope.
        if (body) ctx.visitNode(body);
        return true;
      }
      const name = node.type === 'macro_definition' ? `@${sig.name}` : sig.name;
      emitMethod(node, sig, sigNode, body, ctx, name);
      return true;
    }

    case 'assignment': {
      const parts = named(node);
      const lhs = parts[0];
      const rhs = parts.length >= 3 ? parts[parts.length - 1] : null;
      const op = parts.find((p) => p.type === 'operator');
      if (!lhs || !rhs || !op || getNodeText(op, source) !== '=') return false;
      const isStatement = !!node.parent && STATEMENT_PARENTS.has(node.parent.type);
      if (!isStatement) return false;

      // Short-form method: `f(x) = …`, `f(x)::T = …`, `f(x::T) where T = …`.
      if (isMethodHead(lhs)) {
        const sig = readSignature(lhs, source);
        if (!sig) return false;
        emitMethod(node, sig, lhs, rhs, ctx);
        return true;
      }

      // `name = (x) -> …` — a named anonymous function.
      if (lhs.type === 'identifier' && rhs.type === 'arrow_function_expression') {
        emitMethod(node, { name: getNodeText(lhs, source) }, node, rhs, ctx);
        return true;
      }

      // Module-level globals. Locals are skipped (graph bloat); their
      // right-hand sides still walk so calls attribute to the scope.
      if (!isModuleLevel(node)) return false;
      const targets = lhs.type === 'open_tuple' || lhs.type === 'tuple_expression' ? named(lhs) : [lhs];
      const isConst = isConstDeclaration(node);
      for (const t of targets) {
        const id = t.type === 'typed_expression' ? firstNamed(t) : t;
        if (id?.type !== 'identifier') continue;
        const name = getNodeText(id, source);
        const isTypeValue =
          isConst &&
          targets.length === 1 &&
          (TYPE_VALUE_TYPES.has(rhs.type) ||
            (rhs.type === 'identifier' && /^[A-Z]/.test(getNodeText(rhs, source)) && /^[A-Z]/.test(name) && !/^[A-Z0-9_]+$/.test(name)));
        const kind: NodeKind = isTypeValue ? 'type_alias' : isConst ? 'constant' : 'variable';
        ctx.createNode(kind, name, node, {
          signature: signatureText(statementOf(node), source).slice(0, 200),
          docstring: docstringOf(node, source),
          isExported: isExportedName(node, name, source),
        });
      }
      ctx.visitNode(rhs);
      return true;
    }

    case 'struct_definition':
    case 'abstract_definition':
    case 'primitive_definition': {
      const head = readTypeHead(firstNamed(node, 'type_head'), source);
      if (!head) return false;
      const isAbstract = node.type === 'abstract_definition';
      const kind: NodeKind = isAbstract ? 'interface' : 'struct';
      const typeNode = ctx.createNode(kind, head.name, node, {
        signature: signatureText(firstNamed(node, 'type_head') ?? node, source).slice(0, 200),
        docstring: docstringOf(node, source),
        isExported: isExportedName(node, head.name, source),
        ...(isAbstract ? { isAbstract: true } : {}),
      });
      if (!typeNode) return true;
      if (head.supertype) {
        const superName = typeRefName(head.supertype, source);
        if (superName) addRef(ctx, superName, isAbstract ? 'extends' : 'implements', head.supertype, typeNode.id);
      }
      const body = node.type === 'struct_definition' ? firstNamed(node, 'block') : null;
      if (body) {
        ctx.pushScope(typeNode.id);
        for (const member of named(body)) {
          // `x::T`, `x`, or with a `@kwdef` default `x::T = v`.
          const decl = member.type === 'assignment' && !isMethodHead(firstNamed(member) ?? member) ? firstNamed(member) : member;
          const id = decl?.type === 'typed_expression' ? firstNamed(decl) : decl;
          if (decl && id?.type === 'identifier' && (decl === member || member.type === 'assignment')) {
            const fieldType = decl.type === 'typed_expression' ? named(decl)[1] : undefined;
            ctx.createNode('field', getNodeText(id, source), member, {
              signature: signatureText(member, source).slice(0, 200),
            });
            if (fieldType) {
              const typeName = typeRefName(fieldType, source);
              if (typeName && !BUILTIN_TYPES.has(typeName) && !/^[A-Z]$/.test(typeName)) addRef(ctx, typeName, 'references', fieldType);
            }
            if (member.type === 'assignment') {
              const value = named(member).pop();
              if (value) ctx.visitNode(value);
            }
            continue;
          }
          ctx.visitNode(member);
        }
        ctx.popScope();
      }
      return true;
    }

    case 'macrocall_expression': {
      const macroId = firstNamed(node, 'macro_identifier');
      const args = firstNamed(node, 'macro_argument_list');
      const macroName = macroId ? getNodeText(macroId, source).replace(/\s+/g, '') : '';
      if (macroName === '@enum' && args && emitEnum(node, args, ctx)) return true;
      // Record the macro use (`@m` resolves to `macro m`).
      if (macroName && !ANNOTATION_MACROS.test(macroName)) addRef(ctx, macroName, 'calls', node);
      if (args) visitChildren(args, ctx);
      return true;
    }

    case 'call_expression':
    case 'broadcast_call_expression': {
      const callee = firstNamed(node);
      if (!callee) return false;
      const name = calleeName(callee, source);
      if (name && INCLUDE_FNS.has(name) && node.type === 'call_expression') {
        const file = includedPath(node, source);
        if (file) {
          const imp = ctx.createNode('import', file, node, { signature: signatureText(node, source) });
          if (imp) addRef(ctx, file, 'imports', node);
          return true;
        }
      }
      if (name) addRef(ctx, name, 'calls', node);
      // A callee with no static name may itself hold calls (`f(x)(y)`).
      if (!name) ctx.visitNode(callee);
      else if (callee.type === 'parametrized_type_expression') visitChildren(callee, ctx, firstNamed(callee));
      for (const c of named(node)) {
        if (c !== callee) ctx.visitNode(c);
      }
      return true;
    }

    case 'binary_expression': {
      // `x |> f` / `xs .|> f` apply `f`.
      const parts = named(node);
      const op = parts.find((p) => p.type === 'operator');
      const rhs = parts[parts.length - 1];
      if (op && rhs && PIPE_OPERATORS.has(getNodeText(op, source))) {
        const name = calleeName(rhs, source);
        if (name) addRef(ctx, name, 'calls', rhs);
      }
      return false;
    }

    case 'using_statement':
    case 'import_statement':
      emitImports(node, ctx);
      return true;

    case 'export_statement':
    case 'public_statement':
      return true;

    default:
      return false;
  }
}
