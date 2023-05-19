from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
from data_validation.data_validation import DataValidation
from data_cleaning.data_cleaning import DataCleaning
from model_inference.model_inference import predict
from logger.logger import MongoLogger
import traceback

app = FastAPI()


class Data(BaseModel):
    """
    Data dictionary for data type validation
    """
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    country: str


@app.post("/")
def prediction(data: Data):
    """
    Processes the API request and returns a prediction
    """
    logger = MongoLogger()
    logger.log_to_db(level="INFO", message="entering prediction_api")
    try:
        df = pd.DataFrame(data.dict(), index=[0])  # converting api data dict to df
        dv = DataValidation(input_df=df, dataset="prediction")  # validating the data
        validation_status = dv.validate_data()  # status of validation. 1=passed, 0=failed

        if validation_status != 0:
            data_cleaning = DataCleaning()
            # cleaning the data
            df = data_cleaning.clean_column_names(df).copy()
            df = data_cleaning.shorten_column_names(df).copy()
            df = data_cleaning.clean_nan(df).copy()
            # calling the 'model_inference.model_inference.predict' function
            pred = predict(df, predict_proba=False, predict_label=True)[0].strip()

        else:
            # executes when data validation fails
            pred = "data validation failed"

    except Exception as e:
        # executes in case of any exception
        pred = e
        logger.log_to_db(level="CRITICAL", message=f"unexpected error in prediction_api: {traceback.format_exc()}")
        raise
    logger.log_to_db(level="INFO", message="exiting prediction_api")
    return {"result": pred}


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=5001, workers=4)
