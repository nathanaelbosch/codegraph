/**
 * Julia call resolution: multiple dispatch, approximated statically.
 *
 * A Julia function is a family of methods, and a call runs the one whose
 * parameter types its arguments fit — across files and across packages. A
 * project commonly adds methods to another module's function
 * (`Base.show(io::IO, p::Point)`, or `import Base: copy` then `copy(df::DataFrame)`),
 * so a bare `copy(x)` anywhere in the project is usually Base's own `copy`, and
 * only sometimes the project's method. Matching such a call by name alone
 * linked every `copy(xs)` on a vector to DataFrames' `copy(df::DataFrame)`.
 *
 * So a call is matched against the methods of its name by what the call site
 * shows:
 *   - arity: positional arguments against required / optional / vararg
 *     parameters (keyword arguments never take part in dispatch);
 *   - argument types, where the call site gives them: a literal, a constructor
 *     call `T(…)`, or a name the calling method annotates (`x::T`) or assigns
 *     from a constructor (`x = T(…)`). A type fits a parameter when it is the
 *     parameter's type or reaches it through declared supertypes — the
 *     project's own `struct S <: A` and a table of Base's common types.
 *
 * A method whose arity or types contradict the call is never linked. Among the
 * rest, the best-fitting ones are linked (all of them when several tie, up to a
 * few — dispatch could reach any of them). A function the project only extends
 * (Base's, another package's) needs a positive type match before anything is
 * linked: without one, the call is most likely to the owner's own methods, and
 * silent beats wrong.
 */

import * as path from 'path';
import type { Node } from '../types';
import type { ResolutionContext, ResolvedRef, UnresolvedRef } from './types';

/** Most methods one call links when its arguments can't tell them apart. */
const MAX_TIED_TARGETS = 6;
/**
 * Most methods one call links when they are one interface's implementations:
 * each takes, at the same position, its own subtype of one abstract type
 * (`ene_ops(term::TermKinetic, …)`, `ene_ops(term::TermHartree, …)` under
 * `Term`) — dispatch on that argument, as a virtual call is.
 */
const MAX_FAMILY_TARGETS = 64;

/**
 * Base / standard-library functions that packages extend with methods for
 * their own types. A project method of one of these names is an extension, so
 * a call needs type evidence before it is linked to it.
 */
const BASE_GENERIC_FUNCTIONS: ReadonlySet<string> = new Set([
  'show', 'print', 'println', 'display', 'summary', 'string', 'repr', 'showerror', 'write', 'read', 'open', 'close',
  'flush', 'length', 'size', 'axes', 'ndims', 'eltype', 'keytype', 'valtype', 'getindex', 'setindex!', 'view',
  'firstindex', 'lastindex', 'eachindex', 'keys', 'values', 'pairs', 'haskey', 'get', 'get!', 'iterate', 'first',
  'last', 'isempty', 'empty', 'empty!', 'push!', 'pop!', 'pushfirst!', 'popfirst!', 'append!', 'prepend!',
  'insert!', 'delete!', 'deleteat!', 'resize!', 'fill!', 'copy', 'copy!', 'copyto!', 'deepcopy', 'similar',
  'convert', 'promote_rule', 'promote_type', 'zero', 'zeros', 'one', 'ones', 'oneunit', 'iszero', 'isone', '==',
  '!=', '===', 'isequal', 'isless', 'isapprox', 'hash', '<', '<=', '>', '>=', '+', '-', '*', '/', '\\', '^', '%',
  '÷', '&', '|', '⊗', 'abs', 'abs2', 'sign', 'sqrt', 'cbrt', 'exp', 'log', 'sin', 'cos', 'tan', 'inv', 'real',
  'imag', 'conj', 'adjoint', 'transpose', 'sum', 'prod', 'maximum', 'minimum', 'extrema', 'map', 'map!', 'filter',
  'filter!', 'reduce', 'mapreduce', 'foldl', 'foldr', 'broadcast', 'broadcast!', 'materialize', 'collect', 'parent',
  'names', 'propertynames', 'getproperty', 'setproperty!', 'hasproperty', 'merge', 'merge!', 'in', 'sort', 'sort!',
  'sortperm', 'reverse', 'reverse!', 'unique', 'union', 'intersect', 'setdiff', 'issubset', 'vcat', 'hcat', 'cat',
  'reshape', 'vec', 'permutedims', 'nameof', 'min', 'max', 'clamp', 'floor', 'ceil', 'round', 'trunc', 'float',
  'parse', 'tryparse', 'rand', 'randn', 'run', 'wait', 'notify', 'lock', 'unlock', 'isready', 'take!', 'put!',
  'dot', 'norm', 'det', 'tr', 'mul!', 'ldiv!', 'rdiv!', 'lmul!', 'rmul!', 'axpy!', 'axpby!', 'kron', 'diag',
  'eigen', 'svd', 'qr', 'lu', 'cholesky', 'factorize', 'issymmetric', 'ishermitian', 'isposdef', 'checkbounds',
  'IndexStyle', 'IteratorSize', 'IteratorEltype', 'BroadcastStyle', 'broadcastable', 'axes1', 'strides', 'stride',
  'unsafe_convert', 'cconvert', 'pointer', 'precision', 'eps', 'typemin', 'typemax', 'big', 'widen', 'step',
  'range', 'count', 'any', 'all', 'findfirst', 'findall', 'findnext', 'findprev', 'replace', 'split', 'join',
  'occursin', 'startswith', 'endswith', 'lowercase', 'uppercase', 'strip', 'getfield', 'fieldnames', 'Tuple',
  'NamedTuple', 'Dict', 'Set', 'Array', 'Vector', 'Matrix',
]);

/**
 * Supertypes of Base's common types — enough to tell an `Int` argument fits
 * `x::Real` and does not fit `x::AbstractString`.
 */
const BUILTIN_SUPERTYPE: Readonly<Record<string, string>> = {
  Int: 'Signed', Int8: 'Signed', Int16: 'Signed', Int32: 'Signed', Int64: 'Signed', Int128: 'Signed',
  UInt: 'Unsigned', UInt8: 'Unsigned', UInt16: 'Unsigned', UInt32: 'Unsigned', UInt64: 'Unsigned', UInt128: 'Unsigned',
  Signed: 'Integer', Unsigned: 'Integer', Bool: 'Integer', BigInt: 'Signed', Integer: 'Real',
  Float16: 'AbstractFloat', Float32: 'AbstractFloat', Float64: 'AbstractFloat', BigFloat: 'AbstractFloat',
  AbstractFloat: 'Real', Rational: 'Real', AbstractIrrational: 'Real', Irrational: 'AbstractIrrational', Real: 'Number',
  Complex: 'Number', ComplexF64: 'Number', ComplexF32: 'Number', Number: 'Any',
  String: 'AbstractString', SubString: 'AbstractString', AbstractString: 'Any',
  Char: 'AbstractChar', AbstractChar: 'Any', Symbol: 'Any', Nothing: 'Any', Missing: 'Any',
  Vector: 'DenseVector', DenseVector: 'AbstractVector', Matrix: 'DenseMatrix', DenseMatrix: 'AbstractMatrix',
  Array: 'DenseArray', DenseArray: 'AbstractArray', AbstractVector: 'AbstractArray', AbstractMatrix: 'AbstractArray',
  UnitRange: 'AbstractUnitRange', AbstractUnitRange: 'OrdinalRange', StepRange: 'OrdinalRange',
  OrdinalRange: 'AbstractRange', StepRangeLen: 'AbstractRange', LinRange: 'AbstractRange', AbstractRange: 'AbstractVector',
  AbstractArray: 'Any', Tuple: 'Any', NamedTuple: 'Any', Dict: 'AbstractDict', IdDict: 'AbstractDict',
  AbstractDict: 'Any', Set: 'AbstractSet', AbstractSet: 'Any', Function: 'Any', Type: 'Any', DataType: 'Type',
  IO: 'Any', IOBuffer: 'IO', IOStream: 'IO', Regex: 'Any', Exception: 'Any', Pair: 'Any', Module: 'Any',
};

/** Type names that mean "anything" in a parameter annotation. */
const ANY_TYPES: ReadonlySet<string> = new Set(['Any', '']);

/** A method's positional parameter, read off its signature. */
interface JlParam {
  type: string | null;
  optional: boolean;
  vararg: boolean;
}

/** A method's positional parameters, or null when they could not be read. */
interface JlParams {
  params: JlParam[];
  /** Upper bounds of the method's `where` type variables (`T<:Real` → Real). */
  bounds: Map<string, string>;
}

/**
 * What a call site shows about an argument's type. `exact` when the value's
 * own type is written (a literal, a constructor call); otherwise `name` is a
 * declared upper bound (`x::AbstractTerm`), and the value may be any subtype.
 */
interface JlArgType {
  name: string;
  exact: boolean;
}

/** A call site's positional arguments. */
interface JlArgs {
  /** Inferred type per positional argument (null when unknown). */
  types: (JlArgType | null)[];
  /** A splatted argument (`f(xs...)`) makes the count a lower bound. */
  splat: boolean;
}

/** Per resolution context: what the project's Julia declarations say. */
interface JuliaIndex {
  /** Declared supertype of each project type name. */
  supertypes: Map<string, Set<string>>;
  /** Project type names (struct / abstract / alias). */
  typeNames: Set<string>;
  /** Names a project file imports from another module to extend (`import Base: show`). */
  importedForExtension: Set<string>;
  /** Project module names. */
  modules: Set<string>;
  /** Parsed parameters per method node id. */
  params: Map<string, JlParams | null>;
  /** Module each file belongs to, through `include` chains. */
  fileModule: Map<string, string>;
}

const INDEX = new WeakMap<ResolutionContext, JuliaIndex>();

/** Drop the per-context index (see ReferenceResolver.clearCaches). */
export function clearJuliaDispatchMemos(context: ResolutionContext): void {
  INDEX.delete(context);
}

function juliaNodesOfKind(kind: Node['kind'], context: ResolutionContext): Node[] {
  const out: Node[] = [];
  const it = context.iterateNodesByKind?.(kind);
  if (it) {
    for (const n of it) if (n.language === 'julia') out.push(n);
    return out;
  }
  return context.getNodesByKind(kind).filter((n) => n.language === 'julia');
}

function indexOf(context: ResolutionContext): JuliaIndex {
  let idx = INDEX.get(context);
  if (idx) return idx;
  idx = {
    supertypes: new Map(),
    typeNames: new Set(),
    importedForExtension: new Set(),
    modules: new Set(),
    params: new Map(),
    fileModule: new Map(),
  };
  for (const kind of ['struct', 'interface', 'type_alias'] as const) {
    for (const n of juliaNodesOfKind(kind, context)) {
      idx.typeNames.add(n.name);
      // `Point{T<:Real} <: AbstractShape{T}` — the supertype follows the last top-level `<:`.
      const sig = n.signature ?? '';
      const at = topLevelSubtypeOperator(sig);
      if (at >= 0) {
        const sup = bareTypeName(sig.slice(at + 2));
        if (sup) {
          let set = idx.supertypes.get(n.name);
          if (!set) idx.supertypes.set(n.name, (set = new Set()));
          set.add(sup);
        }
      }
    }
  }
  for (const m of juliaNodesOfKind('module', context)) idx.modules.add(m.name);
  const includes: { from: string; module: string | null; target: string }[] = [];
  for (const imp of juliaNodesOfKind('import', context)) {
    const sig = imp.signature ?? '';
    // `import Base: show, copy` / `import Base.show` extend; `using` does not.
    const selected = /^import\s+[\w.]+\s*:\s*(.+)$/.exec(sig);
    if (selected) {
      for (const part of selected[1]!.split(',')) {
        const name = part.trim();
        if (name) idx.importedForExtension.add(name);
      }
    } else if (/^import\s/.test(sig) && imp.name.includes('.') && !imp.name.startsWith('.')) {
      idx.importedForExtension.add(imp.name.slice(imp.name.lastIndexOf('.') + 1));
    }
    if (/^include\s*\(/.test(sig) && imp.name.endsWith('.jl')) {
      // The include's scope is its module, unless it sits in a function (or
      // a file with no module of its own: then the file's module, below).
      const q = imp.qualifiedName.split('::');
      const scope = q.length > 1 ? q[q.length - 2]! : null;
      includes.push({
        from: imp.filePath,
        module: scope && idx.modules.has(scope) ? scope : null,
        target: path.posix.normalize(path.posix.join(path.posix.dirname(imp.filePath), imp.name)),
      });
    }
  }
  // A file included from inside `module M` belongs to M, as does whatever it
  // includes in turn. Resolve chains to a fixed point (bounded).
  for (let pass = 0; pass < 8; pass++) {
    let changed = false;
    for (const inc of includes) {
      const owner = inc.module ?? idx.fileModule.get(inc.from);
      if (owner && idx.fileModule.get(inc.target) !== owner) {
        idx.fileModule.set(inc.target, owner);
        changed = true;
      }
    }
    if (!changed) break;
  }
  INDEX.set(context, idx);
  return idx;
}

/** Index of the last `<:` outside brackets, or -1. */
function topLevelSubtypeOperator(text: string): number {
  let depth = 0;
  let found = -1;
  for (let i = 0; i < text.length - 1; i++) {
    const c = text[i]!;
    if (c === '{' || c === '(' || c === '[') depth++;
    else if (c === '}' || c === ')' || c === ']') depth--;
    else if (depth === 0 && c === '<' && text[i + 1] === ':') found = i;
  }
  return found;
}

/**
 * The bare name of a type expression: `Vector{Float64}` → `Vector`,
 * `Base.RefValue{T}` → `RefValue`, `::T` → `T`. Null for anything else.
 */
function bareTypeName(text: string): string | null {
  const t = text.trim().replace(/^::/, '').trim();
  const m = /^((?:[A-Za-z_][\w!]*\.)*)([A-Za-z_][\w!]*)/.exec(t);
  return m ? m[2]! : null;
}

/** Split a list at top-level separators, outside brackets and strings. */
function splitTopLevel(text: string, seps: string): { parts: string[]; sepsSeen: string[] } {
  const parts: string[] = [];
  const sepsSeen: string[] = [];
  let depth = 0;
  let cur = '';
  for (let i = 0; i < text.length; i++) {
    const c = text[i]!;
    if (c === '"') {
      const triple = text.startsWith('"""', i);
      const close = triple ? text.indexOf('"""', i + 3) : findStringEnd(text, i + 1);
      const end = close < 0 ? text.length : close + (triple ? 3 : 1);
      cur += text.slice(i, end);
      i = end - 1;
      continue;
    }
    if (c === '(' || c === '[' || c === '{') depth++;
    else if (c === ')' || c === ']' || c === '}') depth--;
    if (depth === 0 && seps.includes(c)) {
      parts.push(cur);
      sepsSeen.push(c);
      cur = '';
      continue;
    }
    cur += c;
  }
  parts.push(cur);
  return { parts, sepsSeen };
}

function findStringEnd(text: string, from: number): number {
  for (let i = from; i < text.length; i++) {
    if (text[i] === '\\') i++;
    else if (text[i] === '"') return i;
  }
  return -1;
}

/**
 * The balanced `( … )` group starting at `open` (an index of `(`): its inner
 * text and the index just past it. Null when it never closes.
 */
function parenGroup(text: string, open: number): { inner: string; end: number } | null {
  let depth = 0;
  for (let i = open; i < text.length; i++) {
    const c = text[i]!;
    if (c === '"') {
      const triple = text.startsWith('"""', i);
      const close = triple ? text.indexOf('"""', i + 3) : findStringEnd(text, i + 1);
      if (close < 0) return null;
      i = close + (triple ? 2 : 0);
      continue;
    }
    if (c === '#' && text[i + 1] !== '=') {
      const nl = text.indexOf('\n', i);
      if (nl < 0) return null;
      i = nl;
      continue;
    }
    if (c === '(' || c === '[' || c === '{') depth++;
    else if (c === ')' || c === ']' || c === '}') {
      depth--;
      if (depth === 0) return { inner: text.slice(open + 1, i), end: i + 1 };
    }
  }
  return null;
}

/**
 * Where a method head's parameter list opens: the first top-level `(` that is
 * not a quoted operator name (`Base.:(==)(a, b)`) or a functor's receiver
 * (`(p::Poly)(x)`), and not inside the head's type parameters (`T{S}(x)`).
 */
function paramListOpen(sig: string): number {
  let depth = 0;
  for (let i = 0; i < sig.length; i++) {
    const c = sig[i]!;
    if (c === '{' || c === '[') depth++;
    else if (c === '}' || c === ']') depth--;
    else if (c === '(' && depth === 0) {
      const quoted = i > 0 && sig[i - 1] === ':';
      if (quoted || i === 0) {
        const g = parenGroup(sig, i);
        if (!g) return -1;
        if (i === 0 && sig[g.end] !== '(') return -1;
        i = g.end - 1;
        continue;
      }
      return i;
    }
  }
  return -1;
}

/** A parameter written through a macro — `@nospecialize(args...)` — as the parameter itself. */
function unwrapParamMacro(raw: string): string {
  const m = /^@\w+\s*\(([^]*)\)$/.exec(raw.trim());
  return m ? m[1]!.trim() : raw.trim();
}

/** Read a method's positional parameters and `where` bounds from its signature. */
function readParams(n: Node, context: ResolutionContext): JlParams | null {
  const idx = indexOf(context);
  if (idx.params.has(n.id)) return idx.params.get(n.id)!;
  let sig = n.signature ?? '';
  if (sig.endsWith('...') && sig.length >= 300) {
    // The stored signature was capped; read the head from source.
    const lines = context.getFileLines?.(n.filePath) ?? context.readFile(n.filePath)?.split(/\r?\n/) ?? null;
    sig = lines ? lines.slice(n.startLine - 1, Math.min(lines.length, n.startLine + 30)).join('\n').replace(/^\s*(?:function|macro)\s+/, '') : '';
  }
  let result: JlParams | null = null;
  const open = paramListOpen(sig);
  const group = open >= 0 ? parenGroup(sig, open) : null;
  if (group) {
    const bounds = new Map<string, string>();
    const tail = sig.slice(group.end);
    const where = /\bwhere\b\s*(\{([^]*)\}|([^=\n]+))/.exec(tail);
    if (where) {
      for (const v of splitTopLevel(where[2] ?? where[3] ?? '', ',').parts) {
        const m = /^\s*([A-Za-z_]\w*)\s*(?:<:\s*(.+))?$/.exec(v.trim());
        if (m) bounds.set(m[1]!, m[2] ? bareTypeName(m[2]) ?? 'Any' : 'Any');
      }
    }
    const { parts, sepsSeen } = splitTopLevel(group.inner, ',;');
    const params: JlParam[] = [];
    for (let i = 0; i < parts.length; i++) {
      if (i > 0 && sepsSeen[i - 1] === ';') break; // keyword parameters follow
      const raw = parts[i]!.trim();
      if (!raw) continue;
      const eq = splitTopLevel(unwrapParamMacro(raw), '=').parts;
      const decl = eq[0]!.trim();
      const optional = eq.length > 1;
      const vararg = decl.endsWith('...') || /::\s*Vararg\b/.test(decl);
      const typeAt = decl.indexOf('::');
      let type = typeAt >= 0 ? decl.slice(typeAt + 2).replace(/\.\.\.$/, '').trim() : null;
      if (type && /^Vararg\b/.test(type)) type = /^Vararg\{([^,}]+)/.exec(type)?.[1]?.trim() ?? null;
      params.push({ type, optional, vararg });
      if (sepsSeen[i] === ';') break;
    }
    result = { params, bounds };
  }
  idx.params.set(n.id, result);
  return result;
}

/** Text of the call written at the ref's position, from its line onward. */
function callTextAt(ref: UnresolvedRef, context: ResolutionContext): string | null {
  const lines = context.getFileLines?.(ref.filePath) ?? context.readFile(ref.filePath)?.split(/\r?\n/) ?? null;
  if (!lines || ref.line < 1 || ref.line > lines.length) return null;
  const first = lines[ref.line - 1]!.slice(Math.max(0, ref.column));
  return [first, ...lines.slice(ref.line, Math.min(lines.length, ref.line + 60))].join('\n');
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&');
}

/** Whether `op` occurs in `text` outside brackets and strings. */
function hasTopLevel(text: string, op: string): boolean {
  let depth = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text[i]!;
    if (c === '"') {
      const close = findStringEnd(text, i + 1);
      if (close < 0) return false;
      i = close;
      continue;
    }
    if (c === '(' || c === '[' || c === '{') depth++;
    else if (c === ')' || c === ']' || c === '}') depth--;
    else if (depth === 0 && text.startsWith(op, i)) return true;
  }
  return false;
}

/** Literal / constructor type of an argument expression, when its text shows one. */
function literalType(arg: string, idx: JuliaIndex): string | null {
  const a = arg.trim();
  if (/^-?\d[\d_]*$/.test(a) || /^0x[\da-fA-F_]+$/.test(a)) return 'Int';
  if (/^-?(?:\d[\d_]*\.\d*|\.\d+|\d[\d_]*)(?:[eEf][-+]?\d+)?$/.test(a)) return 'Float64';
  if (a.startsWith('"')) return 'String';
  if (/^'(?:\\.|[^'\\])'$/.test(a)) return 'Char';
  if (/^:[A-Za-z_]\w*$/.test(a)) return 'Symbol';
  if (a === 'true' || a === 'false') return 'Bool';
  if (a === 'nothing') return 'Nothing';
  if (a === 'missing') return 'Missing';
  if (hasTopLevel(a, '=>')) return 'Pair';
  if (hasTopLevel(a, '->') || /^function\s*\(/.test(a)) return 'Function';
  if (a.startsWith('[') && a.endsWith(']') && parenGroup(a, 0)?.end === a.length) {
    const inner = a.slice(1, -1);
    // `[a b; c d]` is a matrix, `[a, b]` / `[f(x) for x in xs]` a vector.
    return hasTopLevel(inner, ';') || (/\S\s+\S/.test(inner) && !hasTopLevel(inner, ',') && !/\bfor\b/.test(inner)) ? 'Matrix' : 'Vector';
  }
  if (a.startsWith('(') && a.endsWith(')') && parenGroup(a, 0)?.end === a.length && hasTopLevel(a.slice(1, -1), ',')) {
    return /^\(\s*[A-Za-z_]\w*\s*=(?!=)/.test(a) ? 'NamedTuple' : 'Tuple';
  }
  if (/^\d[\d_]*\s*:\s*[\w\d]/.test(a)) return 'UnitRange';
  // Constructor call `T(…)` / `T{…}(…)` / `M.T(…)` of a known type.
  const ctor = /^((?:[A-Za-z_]\w*\.)*)([A-Z][\w!]*)\s*(?:\{[^]*?\})?\s*\(/.exec(a);
  if (ctor && (idx.typeNames.has(ctor[2]!) || ctor[2]! in BUILTIN_SUPERTYPE)) {
    const g = parenGroup(a, a.indexOf('(', ctor[0].length - 1));
    if (g && g.end === a.length) return ctor[2]!;
  }
  return null;
}

/** Type a name has in the calling scope: an annotated parameter, or an assignment above the call. */
function nameType(name: string, ref: UnresolvedRef, caller: Node | null, idx: JuliaIndex, context: ResolutionContext): JlArgType | null {
  if (caller && (caller.kind === 'function' || caller.kind === 'method')) {
    const params = readCallerParams(caller, name);
    if (params) {
      const p = params.type;
      const bare = p ? bareTypeName(p) : null;
      // An unannotated parameter is Any: nothing to read further up.
      if (!bare) return null;
      const t = params.bounds.get(bare) ?? bare;
      return ANY_TYPES.has(t) ? null : { name: t, exact: false };
    }
  }
  const lines = context.getFileLines?.(ref.filePath) ?? null;
  if (!lines) return null;
  const from = caller && caller.kind !== 'file' ? Math.max(caller.startLine, ref.line - 400) : Math.max(1, ref.line - 400);
  const esc = escapeRegExp(name);
  const assign = new RegExp(`(?:^|[\\s;(,])${esc}\\s*(?:::\\s*([A-Z][\\w!.]*)\\s*)?=(?!=)\\s*(.*)$`);
  for (let line = ref.line - 1; line >= from; line--) {
    const text = lines[line - 1];
    if (!text || !text.includes(name)) continue;
    const m = assign.exec(text);
    if (!m) continue;
    if (m[1]) {
      const declared = bareTypeName(m[1]);
      return declared ? { name: declared, exact: false } : null;
    }
    const t = literalType(m[2]!.replace(/\s+#.*$/, '').trim(), idx);
    return t ? { name: t, exact: true } : null;
  }
  return null;
}

/** The annotated type of a parameter named `name` in a method's signature. */
function readCallerParams(caller: Node, name: string): { type: string | null; bounds: Map<string, string> } | null {
  const sig = caller.signature ?? '';
  const open = paramListOpen(sig);
  const group = open >= 0 ? parenGroup(sig, open) : null;
  if (!group) return null;
  const bounds = new Map<string, string>();
  const where = /\bwhere\b\s*(\{([^]*)\}|([^=\n]+))/.exec(sig.slice(group.end));
  if (where) {
    for (const v of splitTopLevel(where[2] ?? where[3] ?? '', ',').parts) {
      const m = /^\s*([A-Za-z_]\w*)\s*(?:<:\s*(.+))?$/.exec(v.trim());
      if (m) bounds.set(m[1]!, m[2] ? bareTypeName(m[2]) ?? 'Any' : 'Any');
    }
  }
  for (const raw of splitTopLevel(group.inner, ',;').parts) {
    const decl = splitTopLevel(unwrapParamMacro(raw), '=').parts[0]!.trim();
    const m = /^([A-Za-z_][\w!]*)\s*(?:::\s*(.+))?$/.exec(decl);
    if (m && m[1] === name) return { type: m[2] ? m[2].replace(/\.\.\.$/, '').trim() : null, bounds };
  }
  return null;
}

/**
 * Whether a name is bound in the calling method: one of its parameters or
 * type variables, or a local assigned (or iterated) above the call.
 */
function isLocallyBound(name: string, ref: UnresolvedRef, caller: Node | null, context: ResolutionContext): boolean {
  if (!caller) return false;
  if (!/^[A-Za-z_\u00a0-\uffff][\w!\u00a0-\uffff]*$/.test(name)) return false; // operators are never bound locally
  const inMethod = caller.kind === 'function' || caller.kind === 'method';
  // Script-level code (tests, examples) binds loop variables too: `for T in (Float32, Float64)`.
  const fromLine = inMethod ? caller.startLine + 1 : 1;
  if (inMethod && readCallerParams(caller, name)) return true;
  const sig = inMethod ? caller.signature ?? '' : '';
  const open = paramListOpen(sig);
  const group = open >= 0 ? parenGroup(sig, open) : null;
  if (group) {
    const where = /\bwhere\b\s*(\{([^]*)\}|([^=\n]+))/.exec(sig.slice(group.end));
    const vars = where ? splitTopLevel(where[2] ?? where[3] ?? '', ',').parts.map((v) => /^\s*([A-Za-z_]\w*)/.exec(v)?.[1]) : [];
    if (vars.includes(name)) return true;
  }
  const lines = context.getFileLines?.(ref.filePath) ?? null;
  if (!lines) return false;
  const esc = escapeRegExp(name);
  const binding = new RegExp(`(?:^|[\\s;(,]|\\blocal\\s)${esc}\\s*(?:::[^=]+)?=(?![=>])|\\bfor\\s+\\(?\\s*${esc}\\s+(?:in|=|∈)`);
  const lambda = new RegExp(`(?:^|[\\s;])${esc}\\s*=\\s*(?:function\\b|\\(?[\\w\\s,:{}]*\\)?\\s*->)`);
  for (let line = Math.max(fromLine, ref.line - 400); line < ref.line; line++) {
    const text = lines[line - 1];
    // `g = x -> …` names a function the index holds — not a value to skip.
    if (text && text.includes(name) && binding.test(text) && !lambda.test(text)) return true;
  }
  return false;
}

/** Positional arguments of the call written at the ref, or null when there is no argument list. */
function readCallArgs(ref: UnresolvedRef, name: string, caller: Node | null, idx: JuliaIndex, context: ResolutionContext): JlArgs | null {
  const text = callTextAt(ref, context);
  if (!text) return null;
  // Skip the callee as written (`f`, `M.f`, `T{P}`, `f.`) to its `(`.
  let i = 0;
  while (i < text.length && /[\w!.@:\u00a0-\uffff]/.test(text[i]!)) i++;
  if (text[i] === '{') {
    const g = parenGroup(text, i);
    if (!g) return null;
    i = g.end;
  }
  if (text[i] === '.') i++;
  if (text[i] !== '(') {
    // `x |> f` — one argument, whose type the pipe's left side would give.
    return /\|>\s*$/.test(context.getFileLines?.(ref.filePath)?.[ref.line - 1]?.slice(0, ref.column) ?? '')
      ? { types: [null], splat: false }
      : null;
  }
  const g = parenGroup(text, i);
  if (!g) return null;
  const { parts, sepsSeen } = splitTopLevel(g.inner, ',;');
  const types: (JlArgType | null)[] = [];
  let splat = false;
  for (let k = 0; k < parts.length; k++) {
    if (k > 0 && sepsSeen[k - 1] === ';') break;
    const raw = parts[k]!.trim();
    if (!raw) continue;
    // `k = v` / `k => v` in a call: a keyword argument / pair... a Pair is positional.
    if (/^[A-Za-z_][\w!]*\s*=(?!=)/.test(raw)) continue;
    if (raw.endsWith('...')) {
      splat = true;
      continue;
    }
    const lit = literalType(raw, idx);
    types.push(lit ? { name: lit, exact: true } : /^[A-Za-z_][\w!]*$/.test(raw) ? nameType(raw, ref, caller, idx, context) : null);
    if (sepsSeen[k] === ';') break;
  }
  // `f(x) do y … end` passes the do-block as the first argument.
  if (/^\s*do\b/.test(text.slice(g.end))) types.unshift({ name: 'Function', exact: false });
  void name;
  return { types, splat };
}

/** Whether `sub` is `sup` or reaches it through declared supertypes. */
function isSubtype(sub: string, sup: string, idx: JuliaIndex): boolean {
  if (sub === sup || ANY_TYPES.has(sup)) return true;
  const seen = new Set<string>();
  const queue = [sub];
  while (queue.length) {
    const t = queue.shift()!;
    if (t === sup) return true;
    if (seen.has(t) || seen.size > 32) continue;
    seen.add(t);
    const project = idx.supertypes.get(t);
    if (project) queue.push(...project);
    const builtin = BUILTIN_SUPERTYPE[t];
    if (builtin) queue.push(builtin);
  }
  return false;
}

/** Whether a type name is one this resolver knows the place of (so a non-fit is a real mismatch). */
function isKnownType(t: string, idx: JuliaIndex): boolean {
  return idx.typeNames.has(t) || t in BUILTIN_SUPERTYPE || t === 'Any';
}

/**
 * How a parameter's annotation admits an argument: a score (exact type 3,
 * subtype 2; 0 for unannotated, unknown, or a subtype the declared argument
 * may turn out to be) and whether the fit is evidence for this method — a match
 * involving one of the project's own types. Null when the annotation rules
 * the argument out.
 */
function paramFit(arg: JlArgType | null, param: JlParam, bounds: Map<string, string>, idx: JuliaIndex): { score: number; evidence: boolean } | null {
  if (!param.type || !arg) return { score: 0, evidence: false };
  const options = /^Union\s*\{/.test(param.type)
    ? splitTopLevel(param.type.slice(param.type.indexOf('{') + 1, param.type.lastIndexOf('}')), ',').parts
    : [param.type];
  let best: { score: number; evidence: boolean } | null = null;
  const take = (score: number, evidence: boolean): void => {
    if (!best || score > best.score) best = { score, evidence };
  };
  let anyUnknown = false;
  for (const opt of options) {
    let t = bareTypeName(opt);
    if (!t) {
      anyUnknown = true;
      continue;
    }
    t = bounds.get(t) ?? t;
    if (t === 'Type') {
      // `::Type{T}` takes a type — the argument is a type name, not a value.
      anyUnknown = true;
      continue;
    }
    if (ANY_TYPES.has(t)) {
      take(0, false);
      continue;
    }
    const projectType = idx.typeNames.has(t) || idx.typeNames.has(arg.name);
    if (arg.name === t) {
      take(3, projectType);
      continue;
    }
    if (isSubtype(arg.name, t, idx)) {
      take(2, projectType);
      continue;
    }
    // A declared abstract argument may hold any of its subtypes at run time.
    // It ranks no higher than an unknown: any of those subtypes may be the one.
    if (!arg.exact && isSubtype(t, arg.name, idx)) {
      take(0, false);
      continue;
    }
    // A project type reaches external supertypes we cannot follow past its
    // first; a builtin value against an external abstract type is unknown.
    // An annotation naming a type we don't know may be a subtype of a declared
    // argument, or a supertype outside what the argument's declarations show.
    if (!isKnownType(t, idx) && (!arg.exact || !idx.typeNames.has(arg.name) || [...ancestors(arg.name, idx)].some((a) => !isKnownType(a, idx)))) anyUnknown = true;
    else if (!arg.exact && !isKnownType(arg.name, idx)) anyUnknown = true;
  }
  if (best) return best;
  return anyUnknown ? { score: 0, evidence: false } : null;
}

/** Score a method against the call's arguments: null when arity or a type rules it out. */
function methodFit(n: Node, args: JlArgs, context: ResolutionContext): { score: number; evidence: boolean } | null {
  const idx = indexOf(context);
  let params: JlParams | null;
  if (n.kind === 'struct') {
    params = structParams(n, context);
  } else {
    params = readParams(n, context);
  }
  if (!params) return { score: 0, evidence: false };
  const ps = params.params;
  const vararg = ps.length > 0 && ps[ps.length - 1]!.vararg;
  const required = ps.filter((p) => !p.optional && !p.vararg).length;
  const max = vararg ? Infinity : ps.length;
  const count = args.types.length;
  // A splatted argument supplies any number of positions, so the count it
  // leaves is only a lower bound.
  if (count > max || (!args.splat && count < required)) return null;
  // With no positional argument, a method that takes none (`T(; kwargs...)`)
  // is more specific than one whose varargs happen to be empty.
  let score = count === 0 && !args.splat && ps.length === 0 ? 1 : 0;
  let evidence = false;
  for (let i = 0; i < count; i++) {
    const p = i < ps.length ? ps[i]! : vararg ? ps[ps.length - 1]! : null;
    if (!p) break;
    const fit = paramFit(args.types[i]!, p, params.bounds, idx);
    if (fit === null) return null;
    if (fit.evidence) evidence = true;
    score += fit.score;
  }
  return { score, evidence };
}

/** A struct's default constructor: one parameter per field, typed as declared. */
function structParams(n: Node, context: ResolutionContext): JlParams | null {
  const idx = indexOf(context);
  const cached = idx.params.get(n.id);
  if (cached !== undefined) return cached;
  const prefix = `${n.qualifiedName}::`;
  const fields = context
    .getNodesInFile(n.filePath)
    .filter((f) => f.kind === 'field' && f.qualifiedName === `${prefix}${f.name}`)
    .sort((a, b) => a.startLine - b.startLine);
  const params: JlParam[] = fields.map((f) => {
    const m = /::\s*([^=]+?)\s*(?:=|$)/.exec(f.signature ?? '');
    return { type: m ? m[1]!.trim() : null, optional: /=/.test(f.signature ?? ''), vararg: false };
  });
  // The struct's own type variables are unbounded here.
  const result: JlParams = { params, bounds: new Map() };
  idx.params.set(n.id, result);
  return result;
}

/** The module a node belongs to: its own `module` scope, else the module its file is included into. */
function moduleOf(n: Node, idx: JuliaIndex): string | null {
  const parts = n.qualifiedName.split('::');
  for (let i = parts.length - 2; i >= 0; i--) {
    if (idx.modules.has(parts[i]!)) return parts[i]!;
  }
  return idx.fileModule.get(n.filePath) ?? null;
}

/**
 * Whether a method extends a function another module owns (`Base.show(…)`) —
 * not one of the project's own modules (`DFTK.ene_ops(…)` in an example).
 */
function extendsOtherModule(n: Node, idx: JuliaIndex): boolean {
  const sig = n.signature ?? '';
  const owner = /^((?:[A-Za-z_]\w*\.)+):?\(?[^\s(]/.exec(sig);
  if (!owner || sig.indexOf('.') >= paramListOpen(sig)) return false;
  const path = owner[1]!.slice(0, -1);
  return !idx.modules.has(path.slice(path.lastIndexOf('.') + 1));
}

/** Every type a type reaches through declared supertypes, itself included, `Any` excluded. */
function ancestors(t: string, idx: JuliaIndex): Set<string> {
  const out = new Set<string>();
  const queue = [t];
  while (queue.length && out.size < 32) {
    const cur = queue.shift()!;
    if (out.has(cur) || ANY_TYPES.has(cur)) continue;
    out.add(cur);
    const project = idx.supertypes.get(cur);
    if (project) queue.push(...project);
    const builtin = BUILTIN_SUPERTYPE[cur];
    if (builtin) queue.push(builtin);
  }
  return out;
}

/**
 * Whether methods are one interface's implementations: at some position each
 * annotates a different project type, and those types share a project
 * supertype.
 */
function isDispatchFamily(methods: Node[], context: ResolutionContext): boolean {
  const idx = indexOf(context);
  const lists = methods.map((m) => (m.kind === 'struct' ? null : readParams(m, context)));
  if (lists.some((l) => !l)) return false;
  const width = Math.min(...lists.map((l) => l!.params.length));
  for (let i = 0; i < width; i++) {
    const types = lists.map((l) => {
      const p = l!.params[i]!;
      const bare = p.type ? bareTypeName(p.type) : null;
      return bare ? l!.bounds.get(bare) ?? bare : null;
    });
    // Every method annotates the position; nearly all with a project type (one
    // the index lacks — a declaration the grammar could not parse — is let be).
    if (types.some((t) => !t)) continue;
    const known = types.filter((t) => idx.typeNames.has(t!));
    if (known.length < 2 || known.length < types.length * 0.9) continue;
    if (new Set(known).size < 2) continue;
    let shared = ancestors(known[0]!, idx);
    for (const t of known.slice(1)) {
      const a = ancestors(t!, idx);
      shared = new Set([...shared].filter((x) => a.has(x)));
    }
    if ([...shared].some((x) => idx.typeNames.has(x))) return true;
  }
  return false;
}

/** Path distance between two files, for the final tie-break. */
function pathDistance(a: string, b: string): number {
  const pa = a.split('/');
  const pb = b.split('/');
  let common = 0;
  while (common < pa.length - 1 && common < pb.length - 1 && pa[common] === pb[common]) common++;
  return pa.length + pb.length - 2 * common;
}

function nearest(cands: Node[], ref: UnresolvedRef): Node[] {
  return [...cands].sort((x, y) =>
    (x.filePath === ref.filePath ? 0 : 1) - (y.filePath === ref.filePath ? 0 : 1) ||
    pathDistance(x.filePath, ref.filePath) - pathDistance(y.filePath, ref.filePath) ||
    x.startLine - y.startLine);
}

function resolved(ref: UnresolvedRef, targets: Node[], confidence: number): ResolvedRef | null {
  if (targets.length === 0) return null;
  return {
    original: ref,
    targetNodeId: targets[0]!.id,
    confidence,
    resolvedBy: 'exact-match',
    ...(targets.length > 1 ? { alsoTargets: targets.slice(1).map((t) => ({ targetNodeId: t.id })) } : {}),
  };
}

const CALLABLE_KINDS: ReadonlySet<string> = new Set(['function', 'method', 'struct']);

/**
 * Resolve a Julia `calls` reference — `f`, `M.f`, `@m` — to the method(s) of
 * that name the call can dispatch to, or to nothing.
 */
export function matchJuliaCall(ref: UnresolvedRef, context: ResolutionContext): ResolvedRef | null {
  const idx = indexOf(context);
  const name = ref.referenceName;

  // Macro calls: `@m` → `macro m`, nearest first.
  if (name.startsWith('@')) {
    const macros = context.getNodesByName(name).filter((n) => n.language === 'julia' && n.kind === 'function');
    if (macros.length === 0) return null;
    const caller = context.getNodeById?.(ref.fromNodeId) ?? null;
    const mod = caller ? moduleOf(caller, idx) : idx.fileModule.get(ref.filePath) ?? null;
    const sameModule = mod ? macros.filter((m) => moduleOf(m, idx) === mod) : [];
    return resolved(ref, nearest(sameModule.length ? sameModule : macros, ref).slice(0, 1), 0.85);
  }

  const dot = name.lastIndexOf('.');
  const qualifier = dot > 0 ? name.slice(0, dot) : null;
  const member = dot > 0 ? name.slice(dot + 1) : name;
  let cands = context.getNodesByName(member).filter((n) => n.language === 'julia' && CALLABLE_KINDS.has(n.kind));
  if (cands.length === 0) return null;

  const caller = context.getNodeById?.(ref.fromNodeId) ?? null;
  // `f(x)` through a parameter, a type variable (`T(0)` under `where T`) or
  // a local binding calls a value, not the function of that name.
  if (!qualifier && isLocallyBound(member, ref, caller, context)) return null;
  let externallyOwned = BASE_GENERIC_FUNCTIONS.has(member) || idx.importedForExtension.has(member) ||
    cands.some((c) => c.kind !== 'struct' && extendsOtherModule(c, idx));

  if (qualifier) {
    const qualLast = qualifier.slice(qualifier.lastIndexOf('.') + 1);
    if (idx.modules.has(qualLast)) {
      // `M.f(…)` into a project module: only M's methods.
      cands = cands.filter((c) => moduleOf(c, idx) === qualLast || (c.kind !== 'struct' && (c.signature ?? '').startsWith(`${qualifier}.`)));
      if (cands.length === 0) return null;
    } else if (/^[A-Z]/.test(qualLast)) {
      // `Base.f(…)` / `LinearAlgebra.f(…)`: the owner's function; the project's
      // methods take part only as extensions of it.
      cands = cands.filter((c) => c.kind !== 'struct' && ((c.signature ?? '').startsWith(`${qualifier}.`) || idx.importedForExtension.has(member)));
      if (cands.length === 0) return null;
      externallyOwned = true;
    } else {
      // `obj.f(…)`: a function stored in a field — no static target.
      return null;
    }
  }

  const args = readCallArgs(ref, member, caller, idx, context);
  type Fit = { node: Node; score: number; evidence: boolean };
  let fits: Fit[];
  if (!args) {
    fits = cands.map((node) => ({ node, score: 0, evidence: false }));
  } else {
    fits = [];
    for (const node of cands) {
      const fit = methodFit(node, args, context);
      if (fit) fits.push({ node, ...fit });
    }
  }
  if (fits.length === 0) return null;

  const best = Math.max(...fits.map((f) => f.score));
  let top = fits.filter((f) => f.score === best);
  const evidence = top.some((f) => f.evidence);
  if (externallyOwned && !evidence) return null;
  // A struct ties with its constructor methods only when nothing else decides:
  // prefer the methods, whose bodies say what construction does.
  if (top.length > 1 && top.some((f) => f.node.kind !== 'struct')) top = top.filter((f) => f.node.kind !== 'struct');

  let targets = top.map((f) => f.node);
  if (targets.length > MAX_TIED_TARGETS && !(targets.length <= MAX_FAMILY_TARGETS && isDispatchFamily(targets, context))) {
    const callerModule = caller ? moduleOf(caller, idx) : idx.fileModule.get(ref.filePath) ?? null;
    const sameFile = targets.filter((t) => t.filePath === ref.filePath);
    const sameModule = callerModule ? targets.filter((t) => moduleOf(t, idx) === callerModule) : [];
    if (sameFile.length > 0 && sameFile.length <= MAX_TIED_TARGETS) targets = sameFile;
    else if (sameModule.length > 0 && sameModule.length <= MAX_TIED_TARGETS) targets = sameModule;
    else return null;
  }
  const confidence = evidence ? 0.9 : targets.length === 1 ? 0.8 : 0.6;
  return resolved(ref, nearest(targets, ref), confidence);
}
