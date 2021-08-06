import inspect
import typing
from typing import Optional, Text, Callable, Tuple, Dict, Type, Any, Set

import dataclasses
import typing_utils

from rasa.engine.exceptions import GraphSchemaValidationException
from rasa.engine.graph import (
    GraphSchema,
    GraphComponent,
    SchemaNode,
    Fingerprintable,
    ExecutionContext,
)
from rasa.engine.storage.resource import Resource
from rasa.nlu import components


@dataclasses.dataclass
class ParamInfo:
    type_annotation: Any
    is_kwargs: bool
    has_default: bool


def keywords_expected_types() -> Dict[Text, Any]:
    from rasa.engine.storage.storage import ModelStorage

    return {
        "resource": Resource,
        "execution_context": ExecutionContext,
        "model_storage": ModelStorage,
        "config": Dict[Text, Any],
    }


def validate(
    schema: GraphSchema, language: Optional[Text], is_train_graph: bool
) -> None:
    """Validates a graph schema.

    This checks that the graph structure is correct (e.g. all nodes pass the correct
    things into each other) and validates the single graph components.

    Args:
        schema: The schema which needs validating.
        language: Used to validate if all components support the language the assistant
            is used in. If the language is `None`, all components are assumed to be
            compatible.
        is_train_graph: Whether the graph is used for training.

    Raises:
        # TODO: Distinguish error types.
        GraphSchemaValidationException: If the validation failed.
    """
    for node_name, node in schema.nodes.items():
        _validate_interface_usage(node_name, node)
        _validate_supported_languages(language, node, node_name)
        _validate_required_packages(node, node_name)

        run_fn = _get_fn(node_name, node.uses, node.fn)
        run_fn_params, run_fn_return_type = _get_parameter_information(run_fn)
        _validate_run_fn(
            node_name, node, run_fn_params, run_fn_return_type, is_train_graph
        )

        constructor = _get_fn(node_name, node.uses, node.constructor_name)
        create_fn_params, _ = _get_parameter_information(constructor)
        _validate_constructor(node_name, node, create_fn_params)

        _validate_needs(
            node_name, node, schema, create_fn_params, run_fn_params,
        )


def _validate_interface_usage(node_name: Text, node: SchemaNode) -> None:
    if not issubclass(node.uses, GraphComponent):
        raise GraphSchemaValidationException(
            f"Node '{node_name}' uses class '{node.uses.__name__}'. This module does "
            f"not implement the '{GraphComponent.__name__}' interface and can "
            f"hence not be used within the graph. Please use a different "
            f"component or implement the '{GraphComponent}' interface for "
            f"'{node.uses.__name__}'."
        )


def _validate_supported_languages(
    language: Optional[Text], node: SchemaNode, node_name: Text
) -> None:
    supported_languages = node.uses.supported_languages()
    if (
        language
        and supported_languages is not None
        and language not in supported_languages
    ):
        raise GraphSchemaValidationException(
            f"Node '{node_name}' does not support the currently specified "
            f"language '{language}'."
        )


def _validate_required_packages(node: SchemaNode, node_name: Text) -> None:
    missing_packages = components.find_unavailable_packages(
        node.uses.required_packages()
    )
    if missing_packages:
        raise GraphSchemaValidationException(
            f"Node '{node_name}' requires the following packages which are "
            f"currently not installed: {', '.join(missing_packages)}."
        )


def _get_fn(node_name: Text, uses: Type, method_name: Text) -> Callable:
    fn = getattr(uses, method_name, None)
    if fn is None:
        raise GraphSchemaValidationException(
            f"Node '{node_name}' uses graph component '{uses.__name__}' which does not "
            f"have the specified "
            f"method '{method_name}'. Please make sure you're either using "
            f"the right graph component or specifying a valid method "
            f"for this component."
        )

    return fn


def _get_parameter_information(
    fn: Callable,
) -> Tuple[Dict[Text, ParamInfo], Optional[Type]]:
    type_hints = typing.get_type_hints(fn)

    return_type = type_hints.pop("return", inspect.Parameter.empty)

    params = inspect.signature(fn).parameters

    type_info = {}
    for param_name, type_annotation in type_hints.items():
        inspect_info = params[param_name]
        type_info[param_name] = ParamInfo(
            type_annotation=type_annotation,
            is_kwargs=inspect_info.kind == inspect.Parameter.VAR_KEYWORD,
            has_default=inspect_info.default != inspect.Parameter.empty,
        )

    return type_info, return_type


def _validate_run_fn(
    node_name: Text,
    node: SchemaNode,
    run_fn_params: Dict[Text, ParamInfo],
    run_fn_return_type: Any,
    is_train_graph: bool,
) -> None:
    _validate_types_of_reserved_keywords(run_fn_params, node_name, node, node.fn)
    _validate_run_fn_return_type(node, run_fn_return_type, is_train_graph)

    for param_name in _required_args(run_fn_params):
        if param_name not in node.needs:
            raise GraphSchemaValidationException(
                f"Node '{node_name}' uses a component '{node.uses.__name__}' which "
                f"needs the param '{param_name}' to be provided to its method "
                f"'{node.fn}'. Please make sure to specify the parameter in "
                f"the node's 'needs' section."
            )


def _required_args(fn_params: Dict[Text, ParamInfo]) -> Set[Text]:
    keywords = set(keywords_expected_types())
    return {
        param_name
        for param_name, param in fn_params.items()
        if not param.has_default and not param.is_kwargs and param_name not in keywords
    }


def _validate_run_fn_return_type(
    node: SchemaNode, return_type: Type, is_training: bool
) -> None:
    if return_type == inspect.Parameter.empty:
        raise GraphSchemaValidationException(
            f"'{node.uses.__name__}.{node.fn}' does not have a type annotation for "
            f"its return value. Type annotations are required for all graph "
            f"components to validate the graph's structure."
        )

    if not isinstance(return_type, Fingerprintable):
        if isinstance(return_type, typing.ForwardRef):
            raise GraphSchemaValidationException(
                f"It seems you used a forward reference to annotate "
                f"'{node.uses.__name__}.{node.fn}'. Please remove the forward "
                f"reference by removing the quotes around the type "
                f"(e.g. 'def foo() -> \"int\"' becomes 'def foo() -> int'."
            )

        if is_training:
            raise GraphSchemaValidationException(
                f"'{node.uses.__name__}.{node.fn}' does not return a fingerprintable "
                f"output. This is required for caching. Please make sure you're "
                f"using a return type which implements the "
                f"'{Fingerprintable.__name__}' protocol."
            )


def _validate_types_of_reserved_keywords(
    params: Dict[Text, ParamInfo], node_name: Text, node: SchemaNode, fn_name: Text
) -> None:
    for param_name, param in params.items():
        if param_name in keywords_expected_types():
            if not typing_utils.issubtype(
                param.type_annotation, keywords_expected_types()[param_name]
            ):
                raise GraphSchemaValidationException(
                    f"Node '{node_name}' uses a graph component "
                    f"'{node.uses.__name__}' which has an incompatible type "
                    f"'{param.type_annotation}' for the '{param_name}' parameter in "
                    f"its '{fn_name}' method."
                )


def _validate_constructor(
    node_name: Text, node: SchemaNode, create_fn_params: Dict[Text, ParamInfo],
) -> None:
    _validate_types_of_reserved_keywords(
        create_fn_params, node_name, node, node.constructor_name
    )

    for param_name in _required_args(create_fn_params):
        if node.eager:
            raise GraphSchemaValidationException(
                f"Node '{node_name}' has a constructor which has a "
                f"required parameter '{param_name}'. Extra parameters can only "
                f"supplied to be the constructor if the node is being run "
                f"in lazy mode."
            )
        if not node.eager and param_name not in node.needs:
            raise GraphSchemaValidationException(
                f"Node '{node_name}' uses a component '{node.uses.__name__}' which "
                f"needs the param '{param_name}' to be provided to its method "
                f"'{node.constructor_name}'. Please make sure to specify the "
                f"parameter in the node's 'needs' section."
            )


def _validate_needs(
    node_name: Text,
    node: SchemaNode,
    graph: GraphSchema,
    create_fn_params: Dict[Text, ParamInfo],
    run_fn_params: Dict[Text, ParamInfo],
) -> None:
    has_kwargs = any(param.is_kwargs for param in run_fn_params.values())
    available_args = run_fn_params.copy()

    if node.eager is False:
        has_kwargs = has_kwargs or any(
            param.is_kwargs for param in create_fn_params.values()
        )
        available_args.update(create_fn_params)

    for param_name, parent_name in node.needs.items():
        if not has_kwargs and param_name not in available_args:
            raise GraphSchemaValidationException(
                f"Node '{node_name}' is configured to retrieve a value for the "
                f"param '{param_name}' by its parent node '{parent_name}' although "
                f"its method '{node.fn}' does not accept a parameter with this "
                f"name. Please make sure your node's 'needs' section is "
                f"correctly specified."
            )

        parent = graph.nodes[parent_name]

        parent_run_fn = _get_fn(parent_name, parent.uses, parent.fn)
        _, parent_return_type = _get_parameter_information(parent_run_fn)

        required_type = available_args.get(param_name)
        needs_passed_to_kwargs = has_kwargs and required_type is None
        if not needs_passed_to_kwargs and not typing_utils.issubtype(
            parent_return_type, required_type.type_annotation
        ):
            raise GraphSchemaValidationException(
                f"Parent of node '{node_name}' returns type "
                f"'{parent_return_type}' but type '{required_type.type_annotation}' "
                f"was expected by component '{node.uses.__name__}'."
            )
