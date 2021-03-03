from pydantic import BaseModel
from fastapi import Query


class ClientDataSimple(BaseModel):
    age: int = Query(..., ge=0)
    job: str = Query(...)
    marital: str = Query(...)
    education: str = Query(...)
    default: str = Query(...)
    housing: str = Query(...)
    loan: str = Query(...)
    contact: str = Query(...)
    month: str = Query(...)
    day_of_week: str = Query(...)
    campaign: int = Query(..., ge=0)
    pdays: int = Query(..., ge=0)
    previous: int = Query(..., ge=0)
    poutcome: str = Query(...)
    emp_var_rate: int = Query(...)
    cons_price_idx: int = Query(...)
    cons_conf_idx: int = Query(...)
    euribor3m: int = Query(...)
    nr_employed: int = Query(...)
