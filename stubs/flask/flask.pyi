# flask.pyi
from typing import Any, Callable, TypeVar, Union, Optional
from typing_extensions import Literal

F = TypeVar('F', bound=Callable[..., Any])

class Flask:
    def route(
        self,
        rule: str,
        methods: Optional[List[str]] = None,
        **options: Any
    ) -> Callable[[F], F]: ...

    def run(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        debug: Optional[bool] = None,
        **options: Any
    ) -> None: ...