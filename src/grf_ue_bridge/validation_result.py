"""Validation、Audit 和 Cleanup 共用的结果契约。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CheckStatus(str, Enum):
    """单个检查的标准状态。"""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


_VALID_STATUSES = {status.value for status in CheckStatus}
_CANONICAL_FIELDS = {"passed", "exit_code", "errors", "warnings", "checks"}
_LEGACY_FIELDS = {"ok", "failed_checks"}


def _json_safe(value: Any) -> Any:
    """把结果详情递归转换为 JSON 可编码的基础类型。"""
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def _status_value(status: Any) -> str:
    value = status.value if isinstance(status, CheckStatus) else status
    if value not in _VALID_STATUSES:
        raise ValueError(
            "check status must be one of: " + ", ".join(sorted(_VALID_STATUSES))
        )
    return value


@dataclass
class ValidationResult:
    """表示一组检查的最终结果。"""

    passed: bool = True
    exit_code: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_check(
        self,
        name: str,
        status: Any,
        required: bool = False,
        message: Optional[str] = None,
        **detail: Any,
    ) -> Dict[str, Any]:
        """加入检查，并为失败的 required 检查补充唯一错误信息。"""
        status_value = _status_value(status)
        check: Dict[str, Any] = {
            "status": status_value,
            "required": bool(required),
        }
        if message is not None:
            check["message"] = message
        check.update(detail)
        self.checks[name] = check

        if status_value == CheckStatus.FAILED.value and required:
            failure_message = message or f"{name} check failed"
            if failure_message not in self.errors:
                self.errors.append(failure_message)
        return check

    def finalize(self) -> "ValidationResult":
        """根据 errors 和 required failed checks 计算最终结论。"""
        required_failure = any(
            isinstance(check, Mapping)
            and check.get("status") == CheckStatus.FAILED.value
            and bool(check.get("required", False))
            for check in self.checks.values()
        )
        self.passed = not self.errors and not required_failure
        self.exit_code = 0 if self.passed else 1
        return self

    def to_dict(self) -> Dict[str, Any]:
        """输出只包含 canonical 字段且可 JSON 序列化的字典。"""
        return {
            "passed": bool(self.passed),
            "exit_code": int(self.exit_code),
            "errors": _json_safe(self.errors),
            "warnings": _json_safe(self.warnings),
            "checks": _json_safe(self.checks),
        }


def _failed_result(message: str) -> ValidationResult:
    result = ValidationResult(errors=[message])
    result.finalize()
    return result


def _read_string_list(report: Mapping[str, Any], key: str) -> Optional[List[str]]:
    value = report.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return list(value)


def _read_canonical_report(report: Mapping[str, Any]) -> ValidationResult:
    missing = sorted(_CANONICAL_FIELDS.difference(report))
    if missing:
        return _failed_result(
            "invalid canonical validation report: missing " + ", ".join(missing)
        )
    if type(report["passed"]) is not bool:
        return _failed_result("invalid canonical validation report: passed must be bool")
    if type(report["exit_code"]) is not int or report["exit_code"] not in (0, 1):
        return _failed_result(
            "invalid canonical validation report: exit_code must be 0 or 1"
        )

    errors = _read_string_list(report, "errors")
    warnings = _read_string_list(report, "warnings")
    checks = report["checks"]
    if errors is None or warnings is None or not isinstance(checks, Mapping):
        return _failed_result("invalid canonical validation report: invalid field types")

    normalized_checks: Dict[str, Dict[str, Any]] = {}
    for name, check in checks.items():
        if not isinstance(check, Mapping):
            return _failed_result(
                f"invalid canonical validation report: check {name!r} is not an object"
            )
        status = check.get("status")
        if status not in _VALID_STATUSES:
            return _failed_result(
                f"invalid canonical validation report: check {name!r} has invalid status"
            )
        if "required" in check and type(check["required"]) is not bool:
            return _failed_result(
                f"invalid canonical validation report: check {name!r} has invalid required"
            )
        normalized_checks[str(name)] = _json_safe(check)

    required_failure = any(
        check.get("status") == CheckStatus.FAILED.value
        and bool(check.get("required", False))
        for check in normalized_checks.values()
    )
    expected_passed = not errors and not required_failure
    expected_exit_code = 0 if expected_passed else 1
    if report["passed"] != expected_passed or report["exit_code"] != expected_exit_code:
        return _failed_result(
            "invalid canonical validation report: passed and exit_code are inconsistent"
        )

    return ValidationResult(
        passed=report["passed"],
        exit_code=report["exit_code"],
        errors=errors,
        warnings=warnings,
        checks=normalized_checks,
    )


def _read_legacy_report(report: Mapping[str, Any]) -> ValidationResult:
    if set(report).intersection(_CANONICAL_FIELDS):
        return _failed_result("ambiguous validation report: mixed canonical and legacy fields")
    if "ok" not in report or "failed_checks" not in report:
        return _failed_result("ambiguous legacy validation report")
    if type(report["ok"]) is not bool or not isinstance(report["failed_checks"], list):
        return _failed_result("invalid legacy validation report")
    warnings = report.get("warnings", [])
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        return _failed_result("invalid legacy validation report: warnings must be a list")

    failed_checks = report["failed_checks"]
    errors: List[str] = []
    if not report["ok"]:
        errors.append("legacy validation report has ok=false")
    if failed_checks:
        errors.extend(str(item) for item in failed_checks)
    if report["ok"] and failed_checks:
        errors.insert(0, "legacy validation report has non-empty failed_checks")

    result = ValidationResult(warnings=list(warnings), errors=errors)
    result.finalize()
    return result


def validation_result_from_report(report: Any) -> ValidationResult:
    """读取 canonical 或明确的 legacy report，异常输入一律 fail-safe。"""
    if isinstance(report, ValidationResult):
        normalized = ValidationResult(
            errors=list(report.errors),
            warnings=list(report.warnings),
            checks=_json_safe(report.checks),
        )
        normalized.finalize()
        return _read_canonical_report(normalized.to_dict())
    if not isinstance(report, Mapping):
        return _failed_result("invalid validation report: expected an object")

    has_canonical = bool(set(report).intersection(_CANONICAL_FIELDS))
    has_legacy = bool(set(report).intersection(_LEGACY_FIELDS))
    if has_canonical and has_legacy:
        return _failed_result("ambiguous validation report: mixed canonical and legacy fields")
    if has_canonical:
        return _read_canonical_report(report)
    if has_legacy:
        return _read_legacy_report(report)
    return _failed_result("invalid validation report: no recognized result fields")
