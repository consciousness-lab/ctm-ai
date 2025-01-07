from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

from werkzeug.datastructures import FileStorage, Headers, MultiDict
from werkzeug.wrappers import Response as BaseResponse

# Flask types
F = TypeVar('F', bound=Callable[..., Any])
ResponseValue = Union[str, bytes, BaseResponse, Dict[str, Any]]

# Define Response as a proper type alias
Response = TypeVar('Response', bound=BaseResponse)

class Flask:
    config: Dict[str, Any]
    debug: bool
    
    def __init__(
        self,
        import_name: str,
        static_url_path: Optional[str] = None,
        static_folder: Optional[str] = None,
        static_host: Optional[str] = None,
        host_matching: bool = False,
        subdomain_matching: bool = False,
        template_folder: Optional[str] = None,
        instance_path: Optional[str] = None,
        instance_relative_config: bool = False,
        root_path: Optional[str] = None
    ) -> None: ...

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

# Global request object
class Request:
    method: str
    args: MultiDict[str, str]
    form: MultiDict[str, str]
    files: MultiDict[str, FileStorage]
    headers: Headers
    
    def get_json(self, force: bool = False, silent: bool = False, cache: bool = True) -> Any: ...
    
    def getlist(self, key: str) -> List[Any]: ...

request: Request

# Response functions
def jsonify(*args: Any, **kwargs: Any) -> BaseResponse: ...
def make_response(*args: Any) -> BaseResponse: ...
def send_from_directory(
    directory: str,
    filename: str,
    **kwargs: Any
) -> BaseResponse: ...

# Other Flask exports
def json(
    *,
    skipkeys: bool = False,
    ensure_ascii: bool = True,
    check_circular: bool = True,
    allow_nan: bool = True,
    cls: Optional[Type[Any]] = None,
    indent: Union[None, int, str] = None,
    separators: Optional[Tuple[str, str]] = None,
    default: Optional[Callable[[Any], Any]] = None,
    sort_keys: bool = False,
    **kw: Any,
) -> str: ...