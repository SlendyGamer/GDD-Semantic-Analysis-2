from __future__ import annotations

from ast_nodes import (
    Assignment,
    BinaryExpr,
    Block,
    CallExpr,
    CallStmt,
    Expr,
    FunctionDecl,
    IdentifierExpr,
    IfStmt,
    IntLiteral,
    BoolLiteral,
    Node,
    Parameter,
    PrintStmt,
    Program,
    ReturnStmt,
    Stmt,
    StringLiteral,
    TypeName,
    UnaryExpr,
    VarDecl,
    WhileStmt,
)
from semantic_errors import (
    SemanticDiagnostic,
    SemanticError,
    SemanticErrorKind,
)
from symbols import FunctionSymbol, Scope, Symbol, SymbolKind

class NameResolver:
    def __init__(self) -> None:
        self.diagnostics: list[SemanticDiagnostic] = []
        self.functions: dict[str, FunctionSymbol] = {}
        self.current_scope: Scope | None = None
        
    def resolve(self, program: Program) -> None:
        self._collect_functions(program)
        self._check_main(program)
        for func in program.functions:
            self._resolve_function(func)
        if self.diagnostics:
            raise SemanticError(self.diagnostics)
        
    def _collect_functions(self, program: Program) -> None:
        for func in program.functions:
            functionName = func.name
            
            if functionName in self.functions:
                self._error(
                    SemanticErrorKind.DUPLICATE_FUNCTION,
                    f"Função '{functionName}' não pode estar duplicada.",
                    func.span,
                )
                continue
            
            parameterTypes = tuple(param.type for param in func.parameters)
            symbol = FunctionSymbol(
                name = functionName,
                kind = SymbolKind.FUNCTION,
                type = func.return_type,
                declaration = func,
                parameter_types = parameterTypes,
            )
            self.functions[functionName] = symbol
            func.metadata["symbol"] = symbol
            
    def _check_main(self, program: Program) -> None:
        mainFunction = self.functions.get("main")
        if mainFunction is None:
            self._error(
                SemanticErrorKind.INVALID_MAIN,
                f"Função main não foi encontrada",
                program.span,
            )
            return
        
        if (mainFunction.type is not TypeName.INT or len(mainFunction.parameter_types) != 0):
            self._error(
                SemanticErrorKind.INVALID_MAIN,
                f"Função {mainFunction} deve ser do tipo int",
                mainFunction.declaration.span,
            )
            
    def _resolve_function(self, function: FunctionDecl) -> None:
        functionScope = Scope(parent = None)
        self.current_scope = functionScope
        
        for param in function.parameters:
            self._declare_parameter(param)
            
        function.body.metadata["scope"] = functionScope
        for stat in function.body.statements:
            self._resolve_statement(stat)
        
        self.current_scope = None
    
    def _declare_parameter(self, parameter: Parameter) -> None:
        assert self.current_scope is not None
        
        if parameter.name in self.current_scope.symbols:
            self._error (
                SemanticErrorKind.DUPLICATE_DECLARATION,
                f"O parâmetro {parameter.name} já foi declarado",
                parameter.span,
            )
            return
        
        symbol = Symbol (
            name = parameter.name,
            kind = SymbolKind.PARAMETER,
            type = parameter.type,
            declaration = parameter,
        )
        self.current_scope.symbols[parameter.name] = symbol
        parameter.metadata["symbol"] = symbol
        
    def _resolve_block(self, block: Block) -> None:
        newScope = Scope(parent = self.current_scope)
        previousScope = self.current_scope
        self.current_scope = newScope
        
        block.metadata["scope"] = newScope
        for stat in block.statements:
            self._resolve_statement(stat)
        
        self.current_scope = previousScope
        
    def _resolve_statement(self, stat: Stmt) -> None:
        if isinstance(stat, VarDecl):
            self._resolve_var_decl(stat)
        elif isinstance(stat, Assignment):
            self._resolve_assignment(stat)
        elif isinstance(stat, CallStmt):
            self._resolve_call_expr(stat.call)
        elif isinstance(stat, IfStmt):
            self._resolve_if(stat)
        elif isinstance(stat, WhileStmt):
            self._resolve_while(stat)
        elif isinstance(stat, ReturnStmt):
            self._resolve_return(stat)
        elif isinstance(stat, PrintStmt):
            self._resolve_print(stat)
        elif isinstance(stat, Block):
            self._resolve_block(stat)
        else:
            raise TypeError(f"Statement do tipo {type(stat)} não tratado")
        
    
    def _resolve_var_decl(self, decl: VarDecl) -> None:
        assert self.current_scope is not None
        
        if decl.name in self.current_scope.symbols:
            self._error(
                SemanticErrorKind.DUPLICATE_DECLARATION,
                f"variável {decl.name} já declarada neste escopo",
                decl.span,
            )
        else:
            symbol = Symbol(
                name = decl.name,
                kind = SymbolKind.VARIABLE,
                type = decl.type,
                declaration = decl,
            )
            self.current_scope.symbols[decl.name] = symbol
            decl.metadata["symbol"] = symbol
            
        if decl.initializer is not None:
            self._resolve_expr(decl.initializer)

    def _resolve_assignment(self, assign: Assignment) -> None:
        self._resolve_expr(assign.target)
        self._resolve_expr(assign.value)
        
    def _resolve_if(self, stat: IfStmt) -> None:
        self._resolve_expr(stat.condition)
        self._resolve_block(stat.then_block)
        if stat.else_block is not None:
            self._resolve_block(stat.else_block)
        
    def _resolve_while(self, stat: WhileStmt) -> None:
        self._resolve_expr(stat.condition)
        self._resolve_block(stat.body)
        
    def _resolve_return(self, stat: ReturnStmt) -> None:
        if stat.value is not None:
            self._resolve_expr(stat.value)
        
    def _resolve_print(self, stat: PrintStmt) -> None:
        for item in stat.items:
            if isinstance(item, Expr):
                self._resolve_expr(item)
                
    def _resolve_expr(self, expr: Expr) -> None:
        if isinstance(expr, IdentifierExpr):
            self._resolve_identifier(expr)
        elif isinstance(expr, CallExpr):
            self._resolve_call_expr(expr)
        elif isinstance(expr, BinaryExpr):
            self._resolve_expr(expr.left)
            self._resolve_expr(expr.right)
        elif isinstance(expr, UnaryExpr):
            self._resolve_expr(expr.operand)
        elif isinstance(expr, (IntLiteral, BoolLiteral, StringLiteral)):
            pass
        else:
            raise TypeError(f"expressão não tratada do tipo {type(expr)}")
        
    def _resolve_identifier(self, expr: IdentifierExpr) -> None:
        symbol = self._lookup_variable(expr.name)
        if symbol is None:
            self._error(
                SemanticErrorKind.UNDECLARED_VARIABLE,
                f"variável {expr.name} não declarada",
                expr.span,
            )
            return
        
        expr.metadata["symbol"] = symbol
        
    def _resolve_call_expr(self, expr: CallExpr) -> None:
        symbol = self.functions.get(expr.name)
        if symbol is None:
            self._error(
                SemanticErrorKind.UNDECLARED_FUNCTION,
                f"função {expr.name} não declarada",
                expr.span,
            )
        else:
            expr.metadata["symbol"] = symbol
            
        for arg in expr.arguments:
            self._resolve_expr(arg)
            
    def _lookup_variable(self, name: str) -> Symbol | None:
        scope = self.current_scope
        while scope is not None:
            if name in scope.symbols:
                return scope.symbols[name]
            scope = scope.parent
        return None
    
    def _error(self, kind: SemanticErrorKind, message: str, span) -> None:
        self.diagnostics.append(SemanticDiagnostic(kind = kind, message = message, span = span))
    
def resolve_names(program: Program) -> None:
    """Construa escopos, símbolos e vínculos entre usos e declarações."""

    # 1. Colete todas as assinaturas de função.
    # 2. Valide a existência e a assinatura de main.
    # 3. Percorra os corpos em ordem, criando um escopo para cada bloco.
    # 4. Anote declarações, usos e blocos na AST.
    # 5. Acumule os diagnósticos desta passagem antes de lançar SemanticError.
    NameResolver().resolve(program)