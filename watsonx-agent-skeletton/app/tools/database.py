"""Database tools via watsonx.data SQL toolkit."""

from __future__ import annotations

from langchain_core.tools import BaseTool
from langchain_ibm import WatsonxLLM

from app.core.client import get_watsonx_client
from app.core.llm import get_llm


def get_sql_database(connection_id: str, schema: str | None = None):
    """Create a WatsonxSQLDatabase connected via Arrow Flight.

    Parameters
    ----------
    connection_id:
        A watsonx.data connection asset ID.
    schema:
        Optional database schema name to scope queries to.
    """
    from langchain_ibm import WatsonxSQLDatabase

    kwargs: dict = {
        "connection_id": connection_id,
        "watsonx_client": get_watsonx_client(),
    }
    if schema:
        kwargs["schema"] = schema
    return WatsonxSQLDatabase(**kwargs)


def get_database_tools(
    connection_id: str,
    schema: str | None = None,
    *,
    llm: WatsonxLLM | None = None,
) -> list[BaseTool]:
    """Return watsonx SQL tools (query, info, list tables, query checker).

    Parameters
    ----------
    connection_id:
        A watsonx.data connection asset ID.
    schema:
        Optional database schema name.
    llm:
        LLM used by the query-checker tool; defaults to the standard model.
    """
    from langchain_ibm import WatsonxSQLDatabaseToolkit

    db = get_sql_database(connection_id, schema)
    toolkit = WatsonxSQLDatabaseToolkit(db=db, llm=llm or get_llm())
    return toolkit.get_tools()
