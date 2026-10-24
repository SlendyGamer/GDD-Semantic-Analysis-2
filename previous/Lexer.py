from __future__ import annotations
import lexer_utils as lu
import enum
import sys
from dataclasses import dataclass
from typing import Iterator

# desativa o limite de conversão decimal (deve ser tratado só na análise semântica)
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


class TokenKind(enum.Enum):
    """Classe já implementada: nomes e números não devem ser alterados."""

    EOF = -1

    IDENTIFIER = 1
    INT_LITERAL = 2
    STRING_LITERAL = 3

    KW_INT = 10
    KW_BOOL = 11
    KW_VOID = 12
    KW_TRUE = 13
    KW_FALSE = 14
    KW_IF = 15
    KW_ELSE = 16
    KW_WHILE = 17
    KW_RETURN = 18
    KW_PRINT = 19

    PLUS = 20
    MINUS = 21
    STAR = 22
    SLASH = 23
    PERCENT = 24
    LESS = 25
    LESS_EQUAL = 26
    GREATER = 27
    GREATER_EQUAL = 28
    EQUAL_EQUAL = 29
    NOT_EQUAL = 30
    LOGICAL_AND = 31
    LOGICAL_OR = 32
    LOGICAL_NOT = 33
    ASSIGN = 34

    LEFT_PAREN = 40
    RIGHT_PAREN = 41
    LEFT_BRACE = 42
    RIGHT_BRACE = 43
    COMMA = 44
    SEMICOLON = 45


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    lexeme: str
    value: int | str | bool | None
    line: int
    column: int

    def __str__(self) -> str:
        return (
            f"<{self.kind.value}, {self.kind.name}, {self.lexeme!r}, "
            f"{self.value!r}, {self.line}, {self.column}>"
        )


class LexerError(Exception):
    def __init__(self, message: str, line: int, column: int):
        super().__init__(message)
        self.message = message
        self.line = line
        self.column = column

    def __str__(self) -> str:
        return f"erro léxico em {self.line}:{self.column}: {self.message}"


KEYWORDS: dict[str, TokenKind] = {
    "int":    TokenKind.KW_INT,
    "bool":   TokenKind.KW_BOOL,
    "void":   TokenKind.KW_VOID,
    "true":   TokenKind.KW_TRUE,
    "false":  TokenKind.KW_FALSE,
    "if":     TokenKind.KW_IF,
    "else":   TokenKind.KW_ELSE,
    "while":  TokenKind.KW_WHILE,
    "return": TokenKind.KW_RETURN,
    "print":  TokenKind.KW_PRINT,
}

SINGLE_OP: dict[str, TokenKind] = {
    "+": TokenKind.PLUS,
    "-": TokenKind.MINUS,
    "*": TokenKind.STAR,
    "/": TokenKind.SLASH,
    "%": TokenKind.PERCENT,
    "<": TokenKind.LESS,
    ">": TokenKind.GREATER,
    "!": TokenKind.LOGICAL_NOT,
    "=": TokenKind.ASSIGN,
    "(": TokenKind.LEFT_PAREN,
    ")": TokenKind.RIGHT_PAREN,
    "{": TokenKind.LEFT_BRACE,
    "}": TokenKind.RIGHT_BRACE,
    ",": TokenKind.COMMA,
    ";": TokenKind.SEMICOLON
}


class Lexer:
    """Converte texto-fonte MicroC em uma sequência de tokens."""

    def __init__(self, source: str):
        self.source = source
        # TODO: inicialize aqui o estado exigido por sua estratégia.
        self.line = 1  # linha atual
        self.column = 1  # caracter da linha atual
        self.pos = 0  # posicao absoluta do caracter

    def getline(self):
        """Retorna posição da linha atual"""
        return self.line

    def getcolumn(self):
        """Retorna posição da coluna atual"""
        return self.column

    def getpos(self):
        """Retorna posição absoluta atual"""
        return self.pos

    def setline(self, line: int):
        """Define o valor da linha como line"""
        if not isinstance(line, int) or line < 1:
            raise ValueError("Valor da linha deve ser um inteiro igual ou superior a 1")

        self.line = line

    def setcolumn(self, column: int):
        """Define o valor da coluna como column"""
        if not isinstance(column, int) or column < 1:
            raise ValueError("Valor da coluna deve ser um inteiro igual ou superior a 1")

        self.column = column

    def advline(self, line: int = 1):
        """Avança a posição da linha line vezes"""
        if not isinstance(line, int) or line < 1:
            raise ValueError("Valor de avanço da linha deve ser um inteiro igual ou superior a 1")
        self.line += line

    def advcolumn(self, column: int = 1):
        """Avança a posição da coluna column vezes"""
        if not isinstance(column, int) or column < 1:
            raise ValueError("Valor de avanço da coluna deve ser um inteiro igual ou superior a 1")
        self.column += column

    def advpos(self, pos: int = 1):
        """Avança a posição absoluta pos vezes"""
        if not isinstance(pos, int) or pos < 1:
            raise ValueError("Valor de avanço da posição deve ser um inteiro igual ou superior a 1")
        self.pos += pos

    def _peek(self, reach: int = 0) -> str | None:
        """Retorna o caractere em pos + n (ou None, caso tenha acabado o texto)"""
        target = self.pos + reach
        if target >= len(self.source):
            return None
        return self.source[target]

    def _advance(self) -> str | None:
        """Consome o caractere atual, atualiza posição do cursor (linha e coluna), detecta CR e retorna o caractere
        (ou None, caso esteja no fim)"""
        char = self._peek()
        if char is None:
            return None

        self.advpos()
        if char == "\n":
            self.advline()
            self.setcolumn(1)
        else:
            self.advcolumn()

        return char

    def skip_non_code(self) -> None:
        """Consome aquilo que não for código útil (espaços, tabs e CR)"""
        while True:
            char = self._peek()
            if char is None:
                return
            if char in (" ", "\t", "\n", "\r"):
                self._advance()
                continue
            if char == "/":
                if self._peek(1) == "/":
                    self._advance()
                    self._advance()
                    while True:
                        comm = self._peek()
                        if comm is None or comm == "\n":
                            break
                        if ord(comm) > 127:
                            raise LexerError(f"Caractere invalido {comm!r}", self.getline(), self.getcolumn())
                        self._advance()
                    continue
                elif self._peek(1) == "*":
                    comm_start_line = self.getline()
                    comm_start_column = self.getcolumn()
                    self._advance()
                    self._advance()
                    while True:
                        comm = self._peek()
                        if comm is None:
                            raise LexerError("Comentário em bloco não fechado", comm_start_line, comm_start_column)
                        if ord(comm) > 127:
                            raise LexerError(f"Caractere invalido {comm!r}", self.getline(), self.getcolumn())
                        if comm == "*" and self._peek(1) == "/":
                            self._advance()
                            self._advance()
                            break
                        self._advance()
                    continue
            return

    def id_or_word(self, start_line: int, start_col: int) -> Token:
        """Consome linguagem [A-Za-z_][A-Za-z0-9_]*, identificando se é um identificador ou palavra"""
        buffer: list[str] = [self._advance()]

        while True:
            char = self._peek()
            if char is not None and (lu.isnumletter(char) or char == '_'):
                buffer.append(self._advance())
            else:
                break

        lexeme = "".join(buffer)
        ktype = KEYWORDS.get(lexeme)
        if ktype is TokenKind.KW_TRUE:
            value: int | str | bool | None = True
        elif ktype is TokenKind.KW_FALSE:
            value = False
        elif ktype is not None:
            value = None
        else:
            ktype = TokenKind.IDENTIFIER
            value = lexeme

        return Token(ktype, lexeme, value, start_line, start_col)

    def _int(self, start_line: int, start_col: int) -> Token:
        """Consome Linguagem [0-9]+, preserva zeros no lexema e gera um token para int ao final"""
        buffer: list[str] = []

        while True:
            char = self._peek()
            if char is not None and lu.isnum(char):
                buffer.append(self._advance())
            else:
                break

        lexeme = "".join(buffer)
        value = int(lexeme)

        return Token(TokenKind.INT_LITERAL, lexeme, value, start_line, start_col)

    def _op_or_punct(self, start_line: int, start_col: int) -> Token:
        """Consome linguagem válida para esses caracteres [+-*/%<>=!&|(){},;], retornando um token que identifica a
        operação lógica sendo realizada ou a pontuação. Segue lista de prioridade por maior prefixo"""
        char = self._advance()

        if char == "<" and self._peek() == "=":
            self._advance()
            return Token(TokenKind.LESS_EQUAL, "<=", None, start_line, start_col)
        if char == ">" and self._peek() == "=":
            self._advance()
            return Token(TokenKind.GREATER_EQUAL, ">=", None, start_line, start_col)
        if char == "=" and self._peek() == "=":
            self._advance()
            return Token(TokenKind.EQUAL_EQUAL, "==", None, start_line, start_col)
        if char == "!" and self._peek() == "=":
            self._advance()
            return Token(TokenKind.NOT_EQUAL, "!=", None, start_line, start_col)
        if char == "&" and self._peek() == "&":
            self._advance()
            return Token(TokenKind.LOGICAL_AND, "&&", None, start_line, start_col)
        if char == "|" and self._peek() == "|":
            self._advance()
            return Token(TokenKind.LOGICAL_OR, "||", None, start_line, start_col)
        if char in SINGLE_OP:
            return Token(SINGLE_OP[char], char, None, start_line, start_col)

        raise LexerError(f"Caractere inválido -> {char!r}", start_line, start_col)

    def _string(self, start_line: int, start_col: int) -> Token:
        """Consome "<string>", incluindo escapes escritos (\\n \\t \\" \\\\ lexeme consome aspas e barras e o value
        exportado é o conteúdo restante (dentro das aspas)"""
        lexeme_buffer: list[str] = [self._advance()]
        value_buffer: list[str] = []

        while True:
            char = self._peek()
            if char is None:
                raise LexerError("String não finalizada", start_line, start_col)
            if char in ("\n", "\r"):
                raise LexerError("Não é permitido quebra de linha ou retorno de carro dentro de Strings", self.getline(), self.getcolumn())

            if char == '"':
                lexeme_buffer.append(self._advance())
                break
            if char == "\\":
                slash_line, slash_col = self.getline(), self.getcolumn()
                lexeme_buffer.append(self._advance())

                esc_char = self._peek()
                if esc_char is None:
                    raise LexerError("String não finalizada", start_line, start_col)

                if esc_char == "n":
                    lexeme_buffer.append(self._advance())
                    value_buffer.append("\n")
                elif esc_char == "t":
                    lexeme_buffer.append(self._advance())
                    value_buffer.append("\t")
                elif esc_char == '"':
                    lexeme_buffer.append(self._advance())
                    value_buffer.append('"')
                elif esc_char == "\\":
                    lexeme_buffer.append(self._advance())
                    value_buffer.append("\\")
                else:
                    raise LexerError(f"Valor inválido para escape -> {esc_char!r}", slash_line, slash_col)
                continue
            if ord(char) > 127:
                raise LexerError(f"Caractere inválido -> {char!r}", self.getline(), self.getcolumn())

            lexeme_buffer.append(self._advance())
            value_buffer.append(char)

        lexeme = "".join(lexeme_buffer)
        value = "".join(value_buffer)
        return Token(TokenKind.STRING_LITERAL, lexeme, value, start_line, start_col)

    def tokens(self) -> Iterator[Token]:
        """Produz todos os tokens significativos e um único EOF ao final."""
        while True:
            # pula comentarios e caracteres vazios
            self.skip_non_code()

            # checa por EOF
            if self._peek() is None:
                yield Token(TokenKind.EOF, "", None, self.line, self.column)
                return

            start_line, start_col = self.getline(), self.getcolumn()
            char = self._peek()
            # verifica nomes e identificadores
            if lu.isletter(char) or char == '_':
                yield self.id_or_word(start_line, start_col)
            # verifica numeros
            elif lu.isnum(char):
                yield self._int(start_line, start_col)
            # verifica operadores
            elif char in SINGLE_OP or char in "&|":
                yield self._op_or_punct(start_line, start_col)
            elif char == '"':
                yield self._string(start_line, start_col)
            else:
                raise LexerError(f"Caractere inválido -> {char!r}", start_line, start_col)

    def scan(self) -> list[Token]:
        return list(self.tokens())