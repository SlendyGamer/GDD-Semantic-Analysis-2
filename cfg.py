"""Estruturas auxiliares do CFG da MicroC. Não substituem a AST."""

from __future__ import annotations

import load_previous  # Disponibiliza os módulos das etapas anteriores.

from dataclasses import dataclass, field

from ast_nodes import Expr, FunctionDecl, ReturnStmt, Stmt
from symbols import Scope


BlockId = int


@dataclass(frozen=True, slots=True)
class Jump:
    target: BlockId


@dataclass(frozen=True, slots=True)
class Branch:
    condition: Expr
    true_target: BlockId
    false_target: BlockId


@dataclass(frozen=True, slots=True)
class Return:
    statement: ReturnStmt


@dataclass(frozen=True, slots=True)
class Stop:
    """Terminador das saídas artificiais, que não possuem sucessores."""


Terminator = Jump | Branch | Return | Stop


@dataclass(frozen=True, slots=True)
class ScopeExit:
    """Marca auxiliar, em ordem, para o fim da vida dos símbolos locais.

    Não é um novo nó da AST. As transferências e o verificador fornecidos
    removem os símbolos locais ao processar esta marca.
    """

    scope: Scope


Operation = Stmt | ScopeExit


@dataclass(slots=True)
class BasicBlock:
    id: BlockId
    operations: list[Operation] = field(default_factory=list)
    terminator: Terminator | None = None
    predecessors: set[BlockId] = field(default_factory=set)
    successors: set[BlockId] = field(default_factory=set)

    # None significa que o bloco ainda está aberto durante a construção.
    # Condições e expressões de retorno estão nos respectivos terminadores.


class CFG:
    def __init__(self, function: FunctionDecl) -> None:
        self.function = function
        self.blocks: dict[BlockId, BasicBlock] = {}
        self._next_id = 0

        self.entry = self.new_block().id
        self.return_exit = self.new_block().id
        self.fallthrough_exit = self.new_block().id

        self.terminate(self.return_exit, Stop())
        self.terminate(self.fallthrough_exit, Stop())

    def new_block(self) -> BasicBlock:
        block = BasicBlock(id=self._next_id)
        self.blocks[block.id] = block
        self._next_id += 1
        return block

    def _connect(self, source: BlockId, target: BlockId) -> None:
        """Atualize as duas pontas de uma aresta. Use terminate por fora."""

        self.blocks[source].successors.add(target)
        self.blocks[target].predecessors.add(source)

    def terminate(self, block_id: BlockId, terminator: Terminator) -> None:
        """Feche um bloco e registre as arestas descritas pelo terminador.

        Todos os destinos precisam existir. A construção da forma do grafo,
        incluindo a escolha desses destinos, permanece no CFGBuilder.
        """

        block = self.blocks[block_id]
        if block.terminator is not None:
            raise ValueError(f"o bloco {block_id} já possui terminador")

        if isinstance(terminator, Jump):
            targets = (terminator.target,)
        elif isinstance(terminator, Branch):
            targets = (terminator.true_target, terminator.false_target)
        elif isinstance(terminator, Return):
            targets = (self.return_exit,)
        elif isinstance(terminator, Stop):
            targets = ()
        else:
            raise TypeError(f"terminador desconhecido: {type(terminator).__name__}")

        for target in targets:
            if target not in self.blocks:
                raise ValueError(f"o bloco de destino {target} não existe")

        block.terminator = terminator
        for target in targets:
            self._connect(block_id, target)


def format_cfg(cfg: CFG) -> str:
    """Inspeção textual dos blocos existentes, inclusive os ainda abertos."""

    lines = [
        f"função {cfg.function.name}",
        f"entry=B{cfg.entry} return_exit=B{cfg.return_exit} "
        f"fallthrough_exit=B{cfg.fallthrough_exit}",
    ]
    for block_id, block in sorted(cfg.blocks.items()):
        lines.append(
            f"B{block_id}: pred={sorted(block.predecessors)} "
            f"succ={sorted(block.successors)}"
        )
        for operation in block.operations:
            if isinstance(operation, ScopeExit):
                lines.append("  ScopeExit")
            else:
                span = operation.span
                lines.append(
                    f"  {type(operation).__name__} "
                    f"em {span.start_line}:{span.start_column}"
                )
        # Mostra a referência à AST sem expandir expressões recursivamente
        terminator = block.terminator
        if isinstance(terminator, Branch):
            span = terminator.condition.span
            description = (
                f"Branch(condition={type(terminator.condition).__name__} "
                f"em {span.start_line}:{span.start_column}, "
                f"true_target={terminator.true_target}, "
                f"false_target={terminator.false_target})"
            )
        elif isinstance(terminator, Return):
            span = terminator.statement.span
            description = (
                f"Return(statement=ReturnStmt "
                f"em {span.start_line}:{span.start_column})"
            )
        else:
            description = repr(terminator)
        lines.append(f"  terminador: {description}")
    return "\n".join(lines)
