/**
 * Julia calls resolve by multiple dispatch, approximated statically: among
 * the methods of a name, the ones the call's arity and argument types fit.
 * A method added to Base's function (`Base.copy(df::Frame)`) is linked only
 * when the call shows a Frame — a `copy(xs)` on anything else is Base's own.
 * An interface's implementations (one method per subtype of an abstract
 * type) are all linked from a call whose argument is only known abstractly.
 */
import { describe, it, expect, afterAll, beforeAll } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { CodeGraph } from '../src';
import type { Node } from '../src/types';

let root = '';
let cg: CodeGraph;

const TERMS = ['Kinetic', 'Hartree', 'Ewald', 'Local', 'Nonlocal', 'Xc', 'Entropy', 'Magnetic'];

beforeAll(async () => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'cg-julia-dispatch-'));
  const files: Record<string, string> = {
    'src/Geo.jl': `module Geo

import Base: copy

include("shapes.jl")
include("terms.jl")

struct Frame
    cols::Vector{Any}
end
Frame(; kwargs...) = Frame(Any[])
Frame(cols::Vector) = Frame(cols)

copy(f::Frame) = Frame(Base.copy(f.cols))
Base.size(f::Frame, i::Integer) = length(f.cols)

macro twice(ex)
    esc(ex)
end

end
`,
    'src/shapes.jl': `abstract type Shape end
struct Circle <: Shape
    r::Float64
end
struct Square <: Shape
    side::Float64
end

area(c::Circle) = pi * c.r^2
area(s::Square) = s.side^2

scale(x::Number) = 2x
scale(x::Number, factor::Number) = factor * x
scale(s::AbstractString) = s * s
`,
    'src/terms.jl': `abstract type Term end
${TERMS.map((t) => `struct Term${t} <: Term end\nene_ops(term::Term${t}, basis) = 0`).join('\n')}

function energy(term::Term, basis)
    ene_ops(term, basis)
end
`,
    'src/use.jl': `function circle_area()
    c = Circle(1.0)
    area(c)
end

square_area(s::Square) = area(s)

scale_one() = scale(3)
scale_two() = scale(3, 2)
scale_text() = scale("ab")

function copies(f::Frame, xs)
    a = copy(f)
    b = copy(xs)
    n = size(xs, 1)
    m = size(f, 1)
    (a, b, n, m)
end

make_frame() = Frame(a = 1)

function apply(f, x)
    f(x)
end

function generic_scale(T)
    for S in (Float32, Float64)
        S(1)
    end
end

twice_area(c::Circle) = @twice area(c)

module_qualified(c::Circle) = Geo.area(c)
`,
  };
  for (const [rel, content] of Object.entries(files)) {
    fs.mkdirSync(path.dirname(path.join(root, rel)), { recursive: true });
    fs.writeFileSync(path.join(root, rel), content);
  }
  cg = await CodeGraph.init(root, { index: true });
});

afterAll(() => {
  cg?.close();
  if (root) fs.rmSync(root, { recursive: true, force: true });
});

/** The targets of a caller's call edges to methods named `name`, as `file:line signature`. */
const callsFrom = (caller: string, name: string): string[] => {
  const node = cg.getNodesInFile('src/use.jl').find((n) => n.name === caller)
    ?? cg.searchNodes(caller).map((r) => r.node).find((n) => n.name === caller)!;
  return cg.getOutgoingEdges(node.id)
    .filter((e) => e.kind === 'calls' || e.kind === 'instantiates')
    .map((e) => cg.getNode(e.target)!)
    .filter((t: Node) => t.name === name)
    .map((t: Node) => t.signature ?? t.name)
    .sort();
};

describe('Julia multiple dispatch', () => {
  it('picks the method for the argument type a constructor assignment gives', () => {
    expect(callsFrom('circle_area', 'area')).toEqual(['area(c::Circle)']);
  });

  it('picks the method for an annotated parameter', () => {
    expect(callsFrom('square_area', 'area')).toEqual(['area(s::Square)']);
  });

  it('tells methods apart by arity and literal types', () => {
    expect(callsFrom('scale_one', 'scale')).toEqual(['scale(x::Number)']);
    expect(callsFrom('scale_two', 'scale')).toEqual(['scale(x::Number, factor::Number)']);
    expect(callsFrom('scale_text', 'scale')).toEqual(['scale(s::AbstractString)']);
  });

  it('links an extension of a Base function only when the argument is the project type', () => {
    expect(callsFrom('copies', 'copy')).toEqual(['copy(f::Frame)']);
    // `size(xs, 1)`: the literal fits `i::Integer`, but nothing shows xs is a Frame.
    const sizes = cg.getOutgoingEdges(cg.getNodesInFile('src/use.jl').find((n) => n.name === 'copies')!.id)
      .filter((e) => e.kind === 'calls' && cg.getNode(e.target)?.name === 'size');
    expect(sizes).toHaveLength(1);
    expect(sizes[0]!.line).toBe(cg.getNodesInFile('src/use.jl').find((n) => n.name === 'copies')!.startLine + 4);
  });

  it('links a keyword-only construction to the keyword-only constructor', () => {
    expect(callsFrom('make_frame', 'Frame')).toEqual(['Frame(; kwargs...)']);
  });

  it('links every implementation of an interface method from an abstract argument', () => {
    const targets = callsFrom('energy', 'ene_ops');
    expect(targets).toHaveLength(TERMS.length);
    expect(targets).toContain('ene_ops(term::TermKinetic, basis)');
  });

  it('never links a call through a parameter or a loop variable', () => {
    const apply = cg.getNodesInFile('src/use.jl').find((n) => n.name === 'apply')!;
    expect(cg.getOutgoingEdges(apply.id).filter((e) => e.kind === 'calls')).toEqual([]);
    const generic = cg.getNodesInFile('src/use.jl').find((n) => n.name === 'generic_scale')!;
    expect(cg.getOutgoingEdges(generic.id).filter((e) => e.kind === 'calls' || e.kind === 'instantiates')).toEqual([]);
  });

  it('resolves macro calls and module-qualified calls', () => {
    expect(callsFrom('twice_area', '@twice')).toEqual(['twice(ex)']);
    expect(callsFrom('twice_area', 'area')).toEqual(['area(c::Circle)']);
    expect(callsFrom('module_qualified', 'area')).toEqual(['area(c::Circle)']);
  });

  it('links include() to the included file', () => {
    const mod = cg.getNodesInFile('src/Geo.jl').find((n) => n.kind === 'module')!;
    const included = cg.getOutgoingEdges(mod.id).filter((e) => e.kind === 'imports').map((e) => cg.getNode(e.target)!.filePath);
    expect(included.sort()).toEqual(['src/shapes.jl', 'src/terms.jl']);
  });
});
