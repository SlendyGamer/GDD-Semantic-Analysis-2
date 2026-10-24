from __future__ import annotations

from ast_nodes import (
    Assignment,
    BinaryExpr,
    BinaryOperator,
    Block,
    BoolLiteral,
    CallExpr,
    CallStmt,
    Expr,
    FunctionDecl,
    IdentifierExpr,
    IfStmt,
    IntLiteral,
    Node,
    PrintStmt,
    Program,
    ReturnStmt,
    Stmt,
    StringLiteral,
    TypeName,
    UnaryExpr,
    UnaryOperator,
    VarDecl,
    WhileStmt,
)
from semantic_errors import (
    SemanticDiagnostic,
    SemanticError,
    SemanticErrorKind,
)
from symbols import FunctionSymbol, Symbol


class TypeChecker:
    def __init__(self) -> None:
        self.diagnostics: list[SemanticDiagnostic] = []

    def check(self, program: Program) -> None:
        for function in program.functions:
            for parameter in function.parameters:
                if parameter.type is TypeName.VOID:
                    self._error(
                        SemanticErrorKind.VOID_PARAMETER,
                        "Parâmetros não podem ter tipo void.",
                        parameter,
                    )
            self._check_block(function.body, function)
        if self.diagnostics:
            raise SemanticError(self.diagnostics)

    def _error(self, kind: SemanticErrorKind, message: str, node: Node) -> None:
        self.diagnostics.append(SemanticDiagnostic(kind, message, node.span))

    def _check_block(self, block: Block, function: FunctionDecl) -> None:
        for statement in block.statements:
            self._check_statement(statement, function)

    def _check_statement(self, statement: Stmt, function: FunctionDecl) -> None:
        if isinstance(statement, Block):
            self._check_block(statement, function)
        elif isinstance(statement, VarDecl):
            self._check_declaration(statement)
        elif isinstance(statement, Assignment):
            target_type = self._expression_type(statement.target)
            value_type = self._expression_type(statement.value)
            if (
                target_type is not None
                and value_type is not None
                and target_type is not value_type
            ):
                self._error(
                    SemanticErrorKind.ASSIGNMENT_TYPE_MISMATCH,
                    "O valor atribuído deve ter o tipo da variável.",
                    statement.value,
                )
        elif isinstance(statement, CallStmt):
            self._expression_type(statement.call, value_required=False)
        elif isinstance(statement, IfStmt):
            self._check_condition(statement.condition)
            self._check_block(statement.then_block, function)
            if statement.else_block is not None:
                self._check_block(statement.else_block, function)
        elif isinstance(statement, WhileStmt):
            self._check_condition(statement.condition)
            self._check_block(statement.body, function)
        elif isinstance(statement, ReturnStmt):
            self._check_return(statement, function)
        elif isinstance(statement, PrintStmt):
            for item in statement.items:
                if not isinstance(item, StringLiteral):
                    self._expression_type(item)
        else:
            raise TypeError(f"Comando não reconhecido: {type(statement).__name__}")

    def _check_declaration(self, declaration: VarDecl) -> None:
        if declaration.type is TypeName.VOID:
            self._error(
                SemanticErrorKind.VOID_VARIABLE,
                "Variáveis não podem ter tipo void.",
                declaration,
            )
        if declaration.initializer is not None:
            initializer_type = self._expression_type(declaration.initializer)
            if (
                declaration.type is not TypeName.VOID
                and initializer_type is not None
                and initializer_type is not declaration.type
            ):
                self._error(
                    SemanticErrorKind.INITIALIZER_TYPE_MISMATCH,
                    "O inicializador deve ter o tipo da variável.",
                    declaration.initializer,
                )

    def _check_condition(self, condition: Expr) -> None:
        condition_type = self._expression_type(condition)
        if condition_type is not None and condition_type is not TypeName.BOOL:
            self._error(
                SemanticErrorKind.CONDITION_TYPE_MISMATCH,
                "A condição deve ter tipo bool.",
                condition,
            )

    def _check_return(self, statement: ReturnStmt, function: FunctionDecl) -> None:
        if statement.value is None:
            if function.return_type is not TypeName.VOID:
                self._error(
                    SemanticErrorKind.RETURN_MISMATCH,
                    "A função deve retornar um valor do tipo declarado.",
                    statement,
                )
            return

        value_type = self._expression_type(statement.value)
        if value_type is not None and (
            function.return_type is TypeName.VOID
            or value_type is not function.return_type
        ):
            self._error(
                SemanticErrorKind.RETURN_MISMATCH,
                "O retorno deve corresponder ao tipo declarado da função.",
                statement.value,
            )

    def _expression_type(
        self, expression: Expr, *, value_required: bool = True
    ) -> TypeName | None:
        # None é apenas um estado interno, não anota tipos inválidos na AST
        expression.metadata.pop("type", None)
        expression_type: TypeName | None
        if isinstance(expression, IntLiteral):
            if not 0 <= expression.value <= 2**63 - 1:
                self._error(
                    SemanticErrorKind.INTEGER_LITERAL_OUT_OF_RANGE,
                    "O literal inteiro deve estar entre 0 e 2**63 - 1.",
                    expression,
                )
                return None
            expression_type = TypeName.INT
        elif isinstance(expression, BoolLiteral):
            expression_type = TypeName.BOOL
        elif isinstance(expression, IdentifierExpr):
            symbol = expression.metadata["symbol"]
            assert isinstance(symbol, Symbol)
            # A declaração void já recebe seu próprio diagnóstico.
            expression_type = symbol.type if symbol.type is not TypeName.VOID else None
        elif isinstance(expression, UnaryExpr):
            expression_type = self._unary_type(expression)
        elif isinstance(expression, BinaryExpr):
            expression_type = self._binary_type(expression)
        elif isinstance(expression, CallExpr):
            expression_type = self._call_type(expression, value_required)
        else:
            raise TypeError(f"Expressão não reconhecida: {type(expression).__name__}")

        if expression_type is not None:
            expression.metadata["type"] = expression_type
        return expression_type

    def _unary_type(self, expression: UnaryExpr) -> TypeName | None:
        operand_type = self._expression_type(expression.operand)
        if operand_type is None:
            return None
        required_type = (
            TypeName.INT
            if expression.operator is UnaryOperator.NEGATE
            else TypeName.BOOL
        )
        if operand_type is not required_type:
            self._error(
                SemanticErrorKind.INVALID_UNARY_OPERAND,
                "O operando é incompatível com o operador unário.",
                expression,
            )
            return None
        return required_type

    def _binary_type(self, expression: BinaryExpr) -> TypeName | None:
        left_type = self._expression_type(expression.left)
        right_type = self._expression_type(expression.right)
        if left_type is None or right_type is None:
            return None

        operator = expression.operator
        if operator in {
            BinaryOperator.ADD,
            BinaryOperator.SUBTRACT,
            BinaryOperator.MULTIPLY,
            BinaryOperator.DIVIDE,
            BinaryOperator.REMAINDER,
        }:
            valid = left_type is right_type is TypeName.INT
            result_type = TypeName.INT
        elif operator in {
            BinaryOperator.LESS,
            BinaryOperator.LESS_EQUAL,
            BinaryOperator.GREATER,
            BinaryOperator.GREATER_EQUAL,
        }:
            valid = left_type is right_type is TypeName.INT
            result_type = TypeName.BOOL
        elif operator in {BinaryOperator.EQUAL, BinaryOperator.NOT_EQUAL}:
            valid = left_type is right_type and left_type in {TypeName.INT, TypeName.BOOL}
            result_type = TypeName.BOOL
        else:
            valid = left_type is right_type is TypeName.BOOL
            result_type = TypeName.BOOL

        if not valid:
            self._error(
                SemanticErrorKind.INVALID_BINARY_OPERANDS,
                "Os operandos são incompatíveis com o operador binário.",
                expression,
            )
            return None
        return result_type

    def _call_type(self, expression: CallExpr, value_required: bool) -> TypeName | None:
        symbol = expression.metadata["symbol"]
        assert isinstance(symbol, FunctionSymbol)
        argument_types = [self._expression_type(argument) for argument in expression.arguments]
        valid = len(argument_types) == len(symbol.parameter_types)
        if not valid:
            self._error(
                SemanticErrorKind.ARITY_MISMATCH,
                "A quantidade de argumentos difere da assinatura da função.",
                expression,
            )

        for argument, actual_type, expected_type in zip(
            expression.arguments, argument_types, symbol.parameter_types
        ):
            if actual_type is None or expected_type is TypeName.VOID:
                valid = False
            elif actual_type is not expected_type:
                self._error(
                    SemanticErrorKind.ARGUMENT_TYPE_MISMATCH,
                    "O argumento deve ter o tipo do parâmetro correspondente.",
                    argument,
                )
                valid = False

        if symbol.type is TypeName.VOID and value_required:
            self._error(
                SemanticErrorKind.VOID_VALUE_USED,
                "Uma chamada void não pode ser usada como valor.",
                expression,
            )
            return None
        return symbol.type if valid else None


def check_types(program: Program) -> None:
    """Determina tipos de expressões e valida seus contextos."""

    TypeChecker().check(program)