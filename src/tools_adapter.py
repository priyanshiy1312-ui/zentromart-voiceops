"""Bridge between the agent and the dataset's mock tools (`tools/tools.py`).

Real signatures (from the dataset):
    lookup_order(order_id, verification_value=None)
        - order_id only  -> {'verification_required': True, ...} (no private data)
        - + 4-digit phone suffix OR 6-digit delivery pincode -> {'found': True, 'verified': True, ...}
    create_ticket(summary, order_id=None, reason=None, intent=None) -> {'created': True, 'ticket_id': ...}
"""
import inspect
import json
import sys

from . import config
from .data import DataStore

SECRET_KEYS = ("verification_value", "phone_last4", "pincode", "phone", "proof")


def _import_dataset_tools():
    root = str(config.DATASET_DIR.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from tools import create_ticket, lookup_order  # tools/__init__.py re-exports

        return lookup_order, create_ticket
    except Exception:
        from tools.tools import create_ticket, lookup_order

        return lookup_order, create_ticket


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str, ensure_ascii=False))


def mask_args(args):
    """Never write verification secrets into logs / sample outputs."""
    return {k: ("***" if k in SECRET_KEYS and v else v) for k, v in (args or {}).items()}


def _verification_value(args):
    for key in ("verification_value", "phone_last4", "phone", "pincode", "proof"):
        v = args.get(key)
        if v not in (None, ""):
            return "".join(ch for ch in str(v) if ch.isdigit()) or None
    return None


class ToolBox:
    """Everything the agent is allowed to call. `run` never raises: errors come back as data."""

    def __init__(self, store=None):
        self.store = store or DataStore()
        self._lookup, self._ticket = _import_dataset_tools()
        self._ticket_params = set(inspect.signature(self._ticket).parameters)

    def run(self, action, args):
        """Returns (result, ok)."""
        args = args or {}
        try:
            if action == "lookup_order":
                oid = str(args.get("order_id", "")).strip()
                value = _verification_value(args)
                res = _jsonable(self._lookup(oid, value) if value else self._lookup(oid))
                return res, isinstance(res, dict) and res.get("verified") is True
            if action == "create_ticket":
                kwargs = {"summary": str(args.get("summary", "")).strip()}
                for key in ("order_id", "reason", "intent"):
                    if args.get(key) and key in self._ticket_params:
                        kwargs[key] = args[key]
                res = _jsonable(self._ticket(**kwargs))
                return res, isinstance(res, dict) and res.get("created") is True
            if action == "check_pincode":
                res = self.store.check_pincode(args.get("pincode", ""))
                return res, "error" not in res
            if action == "product_info":
                res = self.store.product_info(args.get("query") or args.get("sku") or "")
                return res, not (res and "error" in res[0])
            if action == "date_calc":
                res = self.store.date_calc(args.get("op"), args.get("start"), args.get("days"), args.get("end"))
                return res, "error" not in res
            return {"error": f"unknown action {action!r}"}, False
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}, False
