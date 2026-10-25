"""Construa um CFG auxiliar por função, preservando os nós da AST."""

from __future__ import annotations

import load_previous  # Disponibiliza os módulos das etapas anteriores.

from ast_nodes import (
    Assignment, Block, BoolLiteral, CallStmt, FunctionDecl, IfStmt, PrintStmt,
    ReturnStmt, Stmt, VarDecl, WhileStmt,
)
from cfg import BlockId, Branch, CFG, Jump, Return, ScopeExit


class CFGBuilder:
    def build(self, function: FunctionDecl) -> CFG:
        self.cfg = CFG(function)
        body_entry = self.cfg.new_block()
        self.cfg.terminate(self.cfg.entry, Jump(body_entry.id))
        continuation = self._build_block(function.body, body_entry.id)
        if continuation is not None:
            self.cfg.terminate(continuation, Jump(self.cfg.fallthrough_exit))
        return self.cfg

    def _build_block(self, block: Block, current: BlockId) -> BlockId | None:
        """Percorra um bloco da AST e devolva sua continuação normal.

        `current` identifica um bloco básico aberto. Comandos sequenciais
        permanecem nele. Comandos de controle podem mudar a continuação.
        None indica que os caminhos deste bloco não alcançam o próximo comando.
        """

        for statement in block.statements:
            next_block = self._build_statement(statement, current)
            if next_block is None:
                return None
            current = next_block
        self.cfg.blocks[current].operations.append(ScopeExit(block.metadata["scope"]))
        return current

    def _build_statement(self, statement: Stmt, current: BlockId) -> BlockId | None:
        """Incorpore um comando ao CFG e devolva sua continuação normal."""

        if isinstance(statement, Block):
            return self._build_block(statement, current)
        if isinstance(statement, (VarDecl, Assignment, PrintStmt, CallStmt)):
            self.cfg.blocks[current].operations.append(statement)
            return current
        if isinstance(statement, ReturnStmt):
            self.cfg.terminate(current, Return(statement))
            return None
        if isinstance(statement, IfStmt):
            return self._build_if(statement, current)
        if isinstance(statement, WhileStmt):
            return self._build_while(statement, current)
        raise TypeError(f"comando desconhecido: {type(statement).__name__}")

    def _build_if(self, statement: IfStmt, current: BlockId) -> BlockId | None:
        """Construa o fluxo da condicional e devolva sua continuação."""

        then_entry = self.cfg.new_block().id
        else_entry = (
            self.cfg.new_block().id if statement.else_block is not None else None
        )
        join = self.cfg.new_block().id
        self.cfg.terminate(current, Branch(
            statement.condition,
            then_entry,
            join if else_entry is None else else_entry,
        ))

        reaches_join = else_entry is None
        branches = [(statement.then_block, then_entry)]
        if statement.else_block is not None:
            branches.append((statement.else_block, else_entry))
        for body, entry in branches:
            end = self._build_block(body, entry)
            if end is not None:
                self.cfg.terminate(end, Jump(join))
                reaches_join = True

        return join if reaches_join else None

    def _build_while(self, statement: WhileStmt, current: BlockId) -> BlockId | None:
        """Construa o fluxo do laço e devolva sua continuação."""

        header = self.cfg.new_block().id
        body_entry = self.cfg.new_block().id
        self.cfg.terminate(current, Jump(header))

        condition = statement.condition
        infinite = isinstance(condition, BoolLiteral) and condition.value is True
        after = None
        if infinite:
            self.cfg.terminate(header, Jump(body_entry))
        else:
            after = self.cfg.new_block().id
            self.cfg.terminate(header, Branch(condition, body_entry, after))

        end = self._build_block(statement.body, body_entry)
        if end is not None:
            self.cfg.terminate(end, Jump(header))
        return after
