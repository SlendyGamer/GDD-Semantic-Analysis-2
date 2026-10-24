def isnum(char):
    """Retorna True caso o caractere analisado seja um digito numérico inteiro [0-9], caso contrário, retorna False"""
    return '0' <= char <= '9'


def isletter(char):
    """Retorna True caso o caractere analisado seja uma letra [a-z][A-Z], caso contrário, retorna False"""
    return ('a' <= char <= 'z') or ('A' <= char <= 'Z')


def isnumletter(char):
    """Retorna True caso o caractere analisado seja um digito numérico inteiro [0-9] ou letra [a-Z][A-Z],
    caso contrário, retorna False"""
    return isnum(char) or isletter(char)