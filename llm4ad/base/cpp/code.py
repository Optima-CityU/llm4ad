import dataclasses
@dataclasses.dataclass
class CPPFunction:
    """A parsed CPP function."""

@dataclasses.dataclass(frozen=True)
class CPPProgram:
    preface: str
    functions: list[CPPFunction]

class CPPTextFunctionProgramConverter:
    @classmethod
    def text_to_function(cls, program_str: str) -> CPPFunction | None:
        try:
            program = cls.text_to_program(program_str)
            if len(program.functions) != 1:
                raise ValueError(f'Only one function expected, got {len(program.functions)}'
                                 f':\n{program.functions}')
            return program.functions[0]
        except ValueError as value_err:
            raise value_err
        except:
            return None

    @classmethod
    def text_to_program(cls, program_str: str) -> CPPProgram | None:
        """Returns Program object by parsing input text using Python AST.
        """
        try:
            # We assume that the program is composed of some preface (e.g. imports,
            # classes, assignments, ...) followed by a sequence of functions.
            return CPPProgram(preface='', functions=[program_str])
        except:
            return None