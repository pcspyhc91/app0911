"""Google Play App 成功率預測：鎖定模型 Prediction API。"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
import hashlib
import math
import os
import threading
import uuid

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_ID = "Logistic Regression | Tuned"
MODEL_SHA256 = "792c77ccfbd89bd67dff9e7c9c155d661faff50869bce24e8a14dca67fc20882"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "46_3_results" / "46_3_locked_final_logistic_regression.joblib"
MODEL_PATH = Path(os.getenv("MODEL_ARTIFACT_PATH", str(DEFAULT_MODEL_PATH))).resolve()
REFERENCE_DATE = date(2026, 6, 30)
EARLIEST_UPDATE_DATE = date(2020, 2, 5)
THRESHOLD = 0.5

CATEGORY_OPTIONS = ["ARTIFICIAL_INTELLIGENCE","ART_AND_DESIGN","AUTO_AND_VEHICLES","BEAUTY","BOOKS_AND_REFERENCE","BUSINESS","COMICS","COMMUNICATION","DATING","EDUCATION","ENTERTAINMENT","EVENTS","FAMILY","FINANCE","FOOD_AND_DRINK","GAME","HEALTH_AND_FITNESS","HOUSE_AND_HOME","LIBRARIES_AND_DEMO","LIFESTYLE","MAPS_AND_NAVIGATION","MEDICAL","NEWS_AND_MAGAZINES","PARENTING","PERSONALIZATION","PHOTOGRAPHY","PRODUCTIVITY","SHOPPING","SOCIAL","SPORTS","TOOLS","TRAVEL_AND_LOCAL","VIDEO_PLAYERS","VIRTUAL_REALITY","WEATHER","WEB3_AND_CRYPTO"]
CONTENT_RATING_OPTIONS = ["Everyone", "Everyone 10+", "Teen", "Mature 17+", "Adults only 18+", "Unrated"]
SECONDARY_GENRE_OPTIONS = ["NO_SECONDARY_GENRE", "Action", "Board", "Card", "Casual", "Creativity", "Puzzle", "RPG", "Shooter", "Simulation", "Strategy"]
ANDROID_VERSION_OPTIONS = [4.4, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
YES_NO_OPTIONS = ["否", "是"]
EXPECTED_USER_FIELDS = {"App Category","App Price","Content Rating","Size Varies with Device","App Size in MB","Secondary Genre","Last Updated","Current Version Varies","Android Version Varies","Minimum Android Version","In-App Purchases","Ad Supported"}
PROHIBITED_FIELDS = {"App","Rating","Reviews","Installs","Success","Has_Rating","Rating_Available","Rating_Missing","Has_User_Rating","Rating_Exists","Is_Rated"}
MODEL_FEATURES = ["Category","Price","Content Rating","Size_MB","Size_Varies","Is_Paid","Genre_Secondary","Days_Since_Update","Current_Ver_Varies","Android_Ver_Varies","Min_Android_Ver","Has_In_App_Purchases","Has_Ad_Support"]
BINARY_MODEL_FEATURES = ["Size_Varies","Is_Paid","Current_Ver_Varies","Android_Ver_Varies","Has_In_App_Purchases","Has_Ad_Support"]
PREDICTION_LOCK = threading.Lock()

def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

def validation_error(code: str, field: str, message: str, hint: str) -> dict[str, str]:
    return {"error_code":code,"field":field,"message":message,"correction_hint":hint}

def validate_user_input(data: dict[str, Any]) -> tuple[list[dict[str, str]], date | None]:
    errors: list[dict[str, str]] = []
    if set(data) != EXPECTED_USER_FIELDS:
        for field in sorted(EXPECTED_USER_FIELDS-set(data)): errors.append(validation_error("INPUT_REQUIRED",field,f"{field} 為必要欄位。","請完成此欄位後重新驗證。"))
        for field in sorted(set(data)&PROHIBITED_FIELDS): errors.append(validation_error("INPUT_PROHIBITED_FIELD",field,f"{field} 不得進入預測請求。","請移除禁止欄位後重新驗證。"))
        for field in sorted(set(data)-EXPECTED_USER_FIELDS-PROHIBITED_FIELDS): errors.append(validation_error("INPUT_UNEXPECTED_FIELD",field,f"{field} 不在核准的12組介面欄位中。","請移除額外欄位。"))
        return errors, None
    if data["App Category"] not in CATEGORY_OPTIONS: errors.append(validation_error("INPUT_UNKNOWN_CATEGORY","App Category","App Category 不在正式36種類別集合中。","請從正式下拉選單重新選擇。"))
    if data["Content Rating"] not in CONTENT_RATING_OPTIONS: errors.append(validation_error("INPUT_UNKNOWN_CATEGORY","Content Rating","Content Rating 不在正式6種類別集合中。","請從正式下拉選單重新選擇。"))
    if data["Secondary Genre"] not in SECONDARY_GENRE_OPTIONS: errors.append(validation_error("INPUT_UNKNOWN_CATEGORY","Secondary Genre","Secondary Genre 不在正式11種類別集合中。","無次要類型時請選 NO_SECONDARY_GENRE。"))
    price=data["App Price"]
    if not finite(price): errors.append(validation_error("INPUT_INVALID_TYPE","App Price","App Price 必須是有限數值。","請輸入0.00～399.99。"))
    elif not 0 <= float(price) <= 399.99: errors.append(validation_error("INPUT_OUT_OF_TRAINING_RANGE","App Price","App Price 超出正式訓練範圍。","請輸入0.00～399.99。"))
    for field in ["Size Varies with Device","Current Version Varies","Android Version Varies","In-App Purchases","Ad Supported"]:
        if data[field] not in YES_NO_OPTIONS: errors.append(validation_error("INPUT_INVALID_BINARY_VALUE",field,f"{field} 只接受明確的否／是。","請重新選擇否或是。"))
    size=data["App Size in MB"]
    if data["Size Varies with Device"]=="否":
        if not finite(size): errors.append(validation_error("INPUT_REQUIRED","App Size in MB","大小固定時必須提供 App Size in MB。","請輸入5.0～1,998.9 MB。"))
        elif not 5 <= float(size) <= 1998.9: errors.append(validation_error("INPUT_OUT_OF_TRAINING_RANGE","App Size in MB","App Size in MB 超出正式訓練範圍。","請輸入5.0～1,998.9 MB。"))
    elif data["Size Varies with Device"]=="是" and size is not None: errors.append(validation_error("INPUT_STRUCTURAL_MISSINGNESS_ERROR","App Size in MB","大小因裝置而異時，App Size in MB 必須保持結構性缺失。","請勿提交大小數值。"))
    parsed_date=None
    try: parsed_date=date.fromisoformat(data["Last Updated"])
    except (TypeError, ValueError): errors.append(validation_error("INPUT_INVALID_FORMAT","Last Updated","Last Updated 必須是有效的 YYYY-MM-DD 日期。","請使用日期選擇器重新選擇。"))
    if parsed_date and parsed_date>REFERENCE_DATE: errors.append(validation_error("INPUT_RESEARCH_DATE_EXCEEDED","Last Updated","Last Updated 晚於固定研究參考日。","請選擇2026-06-30或更早日期。"))
    elif parsed_date and parsed_date<EARLIEST_UPDATE_DATE: errors.append(validation_error("INPUT_OUT_OF_TRAINING_RANGE","Last Updated","Last Updated 早於正式訓練日期範圍。","請選擇2020-02-05～2026-06-30。"))
    android=data["Minimum Android Version"]
    if data["Android Version Varies"]=="否":
        if not finite(android): errors.append(validation_error("INPUT_REQUIRED","Minimum Android Version","Android版本固定時必須選擇最低版本。","請從正式版本集合選擇。"))
        elif float(android) not in ANDROID_VERSION_OPTIONS: errors.append(validation_error("INPUT_UNKNOWN_CATEGORY","Minimum Android Version","最低Android版本不在正式允許集合中。","請選擇4.4、5.0～12.0的正式版本。"))
    elif data["Android Version Varies"]=="是" and android is not None: errors.append(validation_error("INPUT_STRUCTURAL_MISSINGNESS_ERROR","Minimum Android Version","Android版本因裝置而異時，最低版本必須保持結構性缺失。","請勿提交最低版本。"))
    return errors, parsed_date

def bit(value: str) -> int: return 1 if value=="是" else 0

def build_model_input(data: dict[str, Any], updated: date) -> pd.DataFrame:
    price=float(data["App Price"])
    row={"Category":str(data["App Category"]),"Price":price,"Content Rating":str(data["Content Rating"]),"Size_MB":np.nan if data["Size Varies with Device"]=="是" else float(data["App Size in MB"]),"Size_Varies":bit(data["Size Varies with Device"]),"Is_Paid":0 if price==0 else 1,"Genre_Secondary":str(data["Secondary Genre"]),"Days_Since_Update":int((REFERENCE_DATE-updated).days),"Current_Ver_Varies":bit(data["Current Version Varies"]),"Android_Ver_Varies":bit(data["Android Version Varies"]),"Min_Android_Ver":np.nan if data["Android Version Varies"]=="是" else float(data["Minimum Android Version"]),"Has_In_App_Purchases":bit(data["In-App Purchases"]),"Has_Ad_Support":bit(data["Ad Supported"])}
    return pd.DataFrame([row],columns=MODEL_FEATURES)

def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda:file.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def load_verified_pipeline() -> tuple[Pipeline, dict[str, Any]]:
    if not MODEL_PATH.is_file(): raise HTTPException(503,{"blocking_code":"MODEL_NOT_PROVIDED","message":"正式鎖定模型檔不存在。","correction_hint":"請將原始模型放入 46_3_results/46_3_locked_final_logistic_regression.joblib。"})
    actual=sha256(MODEL_PATH)
    if actual!=MODEL_SHA256: raise HTTPException(503,{"blocking_code":"MODEL_SHA256_MISMATCH","message":"模型檔 SHA-256 與鎖定值不一致。","correction_hint":"請改用未修改的正式模型檔。"})
    try: pipeline=joblib.load(MODEL_PATH)
    except Exception as exc: raise HTTPException(503,{"blocking_code":"MODEL_LOAD_FAILED","message":f"正式模型無法載入（{type(exc).__name__}）。"}) from exc
    if not isinstance(pipeline,Pipeline) or not pipeline.steps: raise HTTPException(503,{"blocking_code":"MODEL_CONTRACT_MISMATCH","message":"模型不是正式 Pipeline。"})
    estimator=pipeline.steps[-1][1]; params=estimator.get_params(deep=False) if isinstance(estimator,LogisticRegression) else {}
    checks={"pipeline_type_match":isinstance(pipeline,Pipeline),"estimator_type_match":isinstance(estimator,LogisticRegression),"feature_order_match":list(getattr(pipeline,"feature_names_in_",[]))==MODEL_FEATURES,"raw_feature_count_match":int(getattr(pipeline,"n_features_in_",-1))==13,"processed_feature_count_match":int(getattr(estimator,"n_features_in_",-1))==63,"class_order_match":[int(v) for v in getattr(estimator,"classes_",[])]==[0,1],"locked_parameters_match":params.get("C")==0.1 and params.get("l1_ratio")==1.0 and params.get("class_weight")=="balanced","predict_proba_available":callable(getattr(pipeline,"predict_proba",None))}
    if not all(checks.values()): raise HTTPException(503,{"blocking_code":"MODEL_CONTRACT_MISMATCH","message":"正式模型契約稽核未通過。","audit":checks})
    return pipeline,{"expected_sha256":MODEL_SHA256,"actual_sha256":actual,"sha256_match":True,**checks,"overall_status":"PASS"}

def audit_and_transform(pipeline: Pipeline, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.shape!=(1,13) or list(frame.columns)!=MODEL_FEATURES: raise HTTPException(422,{"blocking_code":"RAW_MODEL_INPUT_CONTRACT_MISMATCH"})
    for col in BINARY_MODEL_FEATURES:
        if not set(frame[col].dropna()).issubset({0,1}): raise HTTPException(422,{"blocking_code":"MODEL_INPUT_AUDIT_FAILED"})
    preprocessor=pipeline.named_steps.get("preprocessor")
    if preprocessor is None: raise HTTPException(503,{"blocking_code":"LOCKED_PREPROCESSOR_NOT_FOUND"})
    transformed=preprocessor.transform(frame); array=transformed.toarray() if hasattr(transformed,"toarray") else np.asarray(transformed); array=np.asarray(array,dtype=float)
    try: names=[str(v) for v in preprocessor.get_feature_names_out()]
    except Exception as exc: raise HTTPException(503,{"blocking_code":"PIPELINE_TRANSFORMATION_FAILED","message":type(exc).__name__}) from exc
    if array.shape!=(1,63) or len(names)!=63 or not np.isfinite(array).all(): raise HTTPException(503,{"blocking_code":"PIPELINE_TRANSFORMATION_AUDIT_FAILED"})
    return {"raw_input_shape":[1,13],"raw_feature_order_match":True,"transformed_shape":[1,63],"processed_feature_name_count":63,"finite_value_check":True,"predict_proba_call_count":0,"overall_status":"PASS"}

app=FastAPI(title="Google Play App Prediction API",version="5.5.4.13")
origins=[v.strip() for v in os.getenv("ALLOWED_ORIGINS","http://127.0.0.1:5500,http://localhost:5500,null").split(",") if v.strip()]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_credentials=False,allow_methods=["GET","POST"],allow_headers=["Content-Type"])

@app.get("/health")
def health() -> dict[str, Any]:
    try: _,audit=load_verified_pipeline(); return {"status":"READY","model_id":MODEL_ID,"model_integrity":audit["overall_status"]}
    except HTTPException as exc: return {"status":"BLOCKED","detail":exc.detail}

@app.post("/predict")
def predict(request: dict[str, Any]) -> dict[str, Any]:
    data=request; errors,updated=validate_user_input(data)
    if errors: raise HTTPException(422,{"blocking_code":"INPUT_VALIDATION_FAILED","validation_status":"FAIL","errors":errors,"predict_proba_call_count":0})
    assert updated is not None
    frame=build_model_input(data,updated)
    with PREDICTION_LOCK:
        pipeline,integrity=load_verified_pipeline(); transform=audit_and_transform(pipeline,frame)
        probabilities=np.asarray(pipeline.predict_proba(frame),dtype=float); calls=1; classes=[int(v) for v in pipeline.classes_]
    valid=probabilities.shape==(1,2) and classes==[0,1] and np.isfinite(probabilities).all() and np.logical_and(probabilities>=0,probabilities<=1).all() and np.isclose(probabilities[0].sum(),1.0)
    if not valid: raise HTTPException(503,{"blocking_code":"PREDICTION_OUTPUT_AUDIT_FAILED","predict_proba_call_count":calls})
    score=float(probabilities[0,classes.index(1)]); predicted=int(score>=THRESHOLD); symbol=">=" if predicted else "<"; rule=f"Success={predicted}"
    interpretation=f"結果解讀：本筆App輸入在鎖定模型規則下被分到{rule}。這不表示App必然{'會' if predicted else '無法'}達到10,000次累積安裝量。"
    return {"request_id":str(uuid.uuid4()),"validation_status":"PASS","model_input_status":"MODEL_INPUT_READY","model_integrity_status":"MODEL_INTEGRITY_VERIFIED","pipeline_transform_status":"PIPELINE_TRANSFORM_READY","prediction_status":"PREDICTION_READY","result_publication_audit":"PASS","predicted_class":predicted,"model_estimated_score":score,"fixed_threshold":THRESHOLD,"class_0_score":float(probabilities[0,classes.index(0)]),"class_1_score":score,"classification_explanation":f"Success=1的完整精度Model-estimated Score為 {repr(score)}，而固定門檻為 {THRESHOLD:.1f}。因為 {repr(score)} {symbol} {THRESHOLD:.1f}，所以正式分類為{rule}。畫面顯示的六位小數只供閱讀，分類仍使用未經四捨五入的完整精度分數。","result_interpretation":interpretation,"audit":{"model_id":MODEL_ID,"sha256_match":integrity["sha256_match"],"class_order":classes,"positive_class_index":classes.index(1),"predict_proba_call_count":calls,"threshold_match":THRESHOLD==0.5,"pipeline_transform":transform,"overall_status":"PASS"},"test_set_accessed":False,"training_operation_performed":False}
