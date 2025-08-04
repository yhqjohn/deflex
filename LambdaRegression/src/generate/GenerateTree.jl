"""
    GenerateTree

Tree generation module for symbolic regression using lambda expressions.

This module provides a recursive tree generation system based on constructor patterns
and chooser strategies. It uses a context-based composition pattern with NamedTuple
to manage generation state, including depth/try budgets, type constraints, and constructor
caches. The generation strategy balances success rate and diversity through controlled
recursion with fallback mechanisms.

Key components:
- Abstract constructor system (`AbstractCtor`) for different node types
- Chooser strategies (`AbstractChooser`) for constructor selection  
- Context management through flattened namespace pattern
- Special handling for application nodes with const-first strategy
- Error types for generation failure cases
"""
module GenerateTree

using Random
using TestItems

using ..TypeSystem
using ..TypeSystem: unify!, instantiate
using ..ASTs.NodeASTs
using ..ASTs.Contexts
using ..Utils

"""
    AbstractCtor

Abstract base type for all AST node constructors.

Constructors implement the generation logic for specific node types (constants, 
variables, abstractions, applications). Each constructor defines how to check 
type qualification and construct trees of its node type.
"""
abstract type AbstractCtor end

"""
    initialize_ctx(ctor::AbstractCtor, ctx::NamedTuple) -> NamedTuple

Initialize constructor-specific context fields. Returns additional context
fields needed by the constructor during generation.
"""
initialize_ctx(::AbstractCtor, ::NamedTuple) = (;)

"""
    isqualified(ty::TyLike, ctor::AbstractCtor, ctx::NamedTuple) -> Bool

Check if a constructor can generate trees of the given type in the current context.
"""
isqualified(::Ty, ::AbstractCtor, ctx::NamedTuple) = false
isqualified(::TyScheme, ::AbstractCtor, ctx::NamedTuple) = isqualified(instantiate(ty, 0), ctor, ctx)

"""
    construct_tree(ctor::AbstractCtor, ty::Ty, ctx::NamedTuple) -> Union{AstNode, Err}

Construct an AST tree of the given type using the specified constructor.
Returns either a valid AST node or an error.
"""
function construct_tree end

"""
    AbstractChooser

Abstract base type for constructor selection strategies.

Choosers implement the logic for selecting which constructor to use when
multiple constructors are qualified for a given type.
"""
abstract type AbstractChooser end

"""
    initialize_ctx(chooser::AbstractChooser, ctx::NamedTuple) -> NamedTuple

Initialize chooser-specific context fields.
"""
initialize_ctx(::AbstractChooser, ::NamedTuple) = (;)

"""
    choose_ctor(chooser::AbstractChooser, ty::TyLike, ctx::NamedTuple) -> Union{AbstractCtor, Err}

Select a qualified constructor for the given type using the chooser's strategy.
"""
function choose_ctor end

struct MaxDepthReachedErr end
struct MaxTryReachedErr end
struct NoQualifiedCtorErr end
struct SingleTypeResolveErr end


"""
    UniformChooser <: AbstractChooser

Chooser that selects constructors uniformly at random from qualified candidates.

Uses shuffled iteration through constructors to provide uniform selection
among those qualified for the requested type.
"""
struct UniformChooser <: AbstractChooser end
function choose_ctor(chooser::UniformChooser, ty::TyLike, ctx::NamedTuple)
    ctors = ctx.ctors
    ctors = shuffle(ctx.rng, ctors)
    for ctor in ctors
        if isqualified(ty, ctor, ctx)
            return ctor
        end
    end
    return Err(NoQualifiedCtorErr())
end

"""
    generate_tree(ty::Ty; kwargs...) -> Union{AstTree, Err}

Generate an AST tree of the specified type using recursive constructor selection.

# Arguments
- `ty::Ty`: Target type for the generated tree
- `ctors::Vector{<:AbstractCtor}`: Available constructors for tree generation
- `chooser::AbstractChooser=UniformChooser()`: Strategy for constructor selection
- `max_depth::Int=10`: Maximum recursion depth budget
- `max_try::Int=10`: Maximum attempts per generation step
- `tyfactory=newtyvar`: Type variable factory function
- `ctx...`: Additional context fields

Uses a recursive generation strategy with depth and attempt budgets to construct
valid AST trees. Combines constructor qualification checking with chooser strategies
to balance success rate and diversity.
"""
function generate_tree(ty::Ty;
    ctors::Vector{<:AbstractCtor},
    chooser::AbstractChooser=UniformChooser(),
    max_depth::Int=10,
    max_try::Int=10,
    tyfactory = newtyvar,
    symfactory = SymbolGenerator(),
    ctx...
)
    initial_ctx = merge(ctx, (;ctors=ctors, chooser=chooser, max_depth=max_depth, max_try=max_try, tyfactory=tyfactory, symfactory=symfactory))
    for ctor in ctors
        ctor_ctx = initialize_ctx(ctor, initial_ctx)
        initial_ctx = merge(initial_ctx, ctor_ctx)
    end
    initial_ctx = merge(initial_ctx, initialize_ctx(chooser, initial_ctx))
    _generate_tree(ty, initial_ctx)
end

function _generate_tree(ty::Ty, ctx::NamedTuple)
    max_try = ctx.max_try
    for _ in 1:max_try
        ctor = @✓ choose_ctor(ctx.chooser, ty, ctx)
        result = construct_tree(ctor, ty, ctx)
        if result isa AstTree
            return result
        elseif result isa Err
            continue
        end
    end
    return Err(MaxTryReachedErr())
end

abstract type AbstractLeafCtor <: AbstractCtor end

include("constantctors.jl")

"""
    AbstractAppCtor <: AbstractCtor

Abstract base type for application node constructors.

Application constructors generate ApplyNode instances representing function
applications in the lambda calculus AST.
"""
abstract type AbstractAppCtor <: AbstractCtor end

"""
    AbstractAbsCtor <: AbstractCtor

Abstract base type for abstraction node constructors.

Abstraction constructors generate AbsNode instances representing lambda
abstractions in the AST.
"""
abstract type AbstractAbsCtor <: AbstractCtor end

"""
    AbstractVarCtor <: AbstractCtor

Abstract base type for variable node constructors.

Variable constructors generate variable references, typically as De Bruijn
indices for bound variables.
"""
abstract type AbstractVarCtor <: AbstractLeafCtor end

"""
    IndexCtor <: AbstractVarCtor

Constructor for De Bruijn index variable nodes.

Generates IndexNode instances that reference variables by their De Bruijn
indices within the lambda abstraction scope stack.
"""
struct IndexCtor <: AbstractVarCtor end
function initialize_ctx(::IndexCtor, ctx::NamedTuple)
    var_ty_ctx = haskey(ctx, :var_ty) ? (;) : (;var_ty=Dict{Symbol,Ty}())
    scope_stack_ctx = haskey(ctx, :scope_stack) ? (;) : (;scope_stack=Vector{Symbol}())
    return merge(var_ty_ctx, scope_stack_ctx)
end

"""
    ApplyLeafCtor <: AbstractAppCtor

Application constructor with constant function strategy.

Generates application nodes where the function part is constructed using
constant constructors first, then the argument is generated to match the
inferred function type. Uses caching for qualified constant constructors
to improve efficiency.
"""
struct ApplyLeafCtor <: AbstractAppCtor end

function initialize_ctx(::ApplyLeafCtor, ctx::NamedTuple)
    return (;quanlified_const_ctor_cache=Dict{TyLike,Vector{AbstractCtor}}())
end
function isqualified(ty::TyLike, ::ApplyLeafCtor, ctx::NamedTuple)
    arg_ty = ctx.tyfactory(1)
    ret_ty = instantiate(ty, 1)
    func_ty = generalize(arg_ty → ret_ty)

    if haskey(ctx.quanlified_const_ctor_cache, func_ty)
        return length(ctx.quanlified_const_ctor_cache[func_ty]) > 0
    else
        qualified_ctors = AbstractCtor[]
        for ctor in ctx.ctors
            if ctor isa AbstractLeafCtor && isqualified(func_ty, ctor, ctx)
                push!(qualified_ctors, ctor)
            end
        end
        ctx.quanlified_const_ctor_cache[func_ty] = qualified_ctors
        return length(qualified_ctors) > 0
    end
end
function construct_tree(::ApplyLeafCtor, ty::Ty, ctx::NamedTuple)
    arg_ty = ctx.tyfactory(1)
    ret_ty = instantiate(generalize(ty), 1)
    func_ty = generalize(arg_ty → ret_ty)
    func = @✓ _generate_tree(instantiate(func_ty, 0), merge(ctx, 
    (;max_depth=ctx.max_depth-1, 
    ctors=ctx.quanlified_const_ctor_cache[func_ty])))
    inst_func_ty = @✓ infer(AstTree(func), MultipleMapView(ctx.global_consts_ty, ctx.local_consts_ty))
    inst_arg_ty = inst_func_ty.arg
    for _ in 1:ctx.max_try
        arg_result = _generate_tree(inst_arg_ty, merge(ctx, (;max_depth=ctx.max_depth-1)))
        if arg_result isa AstNode
            arg = arg_result
            app = ApplyNode(func, arg)
            inst_app_ty_result = infer(AstTree(app), MultipleMapView(ctx.global_consts_ty, ctx.local_consts_ty))
            if inst_app_ty_result isa Err
                continue
            end
            inst_app_ty = inst_app_ty_result
            if iscompatible(inst_app_ty, ty)
                return app
            end
        end
    end
    return Err(MaxTryReachedErr())
end

"""
    ApplyToLeafCtor <: AbstractAppCtor

Application constructor with constant argument strategy.

Generates application nodes where the argument part is generated first,
then the function is constructed to match the required function type based
on the inferred argument type and target result type.
"""
struct ApplyToLeafCtor <: AbstractAppCtor end
function construct_tree(::ApplyToLeafCtor, ty::Ty, ctx::NamedTuple)
    arg_ty = ctx.tyfactory(1)
    arg = @✓ _generate_tree(arg_ty, merge(ctx, (;max_depth=ctx.max_depth-1)))
    inst_arg_ty = @✓ infer(AstTree(arg), MultipleMapView(ctx.global_consts_ty, ctx.local_consts_ty))
    func_ty = generalize(inst_arg_ty → ty)
    func = @✓ _generate_tree(instantiate(func_ty, 0), merge(ctx, (;max_depth=ctx.max_depth-1)))
    app = ApplyNode(func, arg)
    inst_app_ty = @✓ infer(AstTree(app), MultipleMapView(ctx.global_consts_ty, ctx.local_consts_ty))
    if iscompatible(inst_app_ty, ty)
        return app
    else
        return Err(SingleTypeResolveErr())
    end
end

struct AbsCtor <: AbstractAbsCtor end

isqualified(ty::TyFun, ::AbsCtor, ctx::NamedTuple) = true # only for function type
function construct_tree(::AbsCtor, ty::Ty, ctx::NamedTuple)
    arg_ty = ty.arg
    name = ctx.symfactory()
    

end






end # module GenerateTree