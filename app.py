from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor
#from pyspark.ml.regression import FMRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
import streamlit as st
from pyspark.sql import SparkSession
import pandas as pd
import os
#from pyspark.sql.functions import *
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.abspath(__file__))

sc = SparkSession.builder.appName('airbnb_price') \
            .getOrCreate()

sc = SparkSession.builder \
    .appName("YourAppName") \
    .config("spark.sql.execution.arrow.enabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

sc.conf.set("spark.sql.execution.arrow.enabled", "false")

st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title('''New York Airbnb Predicción del Precio''')
st.subheader('Modelos de Machine Learning Regression en MLlib. \
    Dataset [link](https://www.kaggle.com/code/benroshan/belong-anywhere-ny-airbnb-price-prediction/notebook)')
st.subheader('Editado por [Juan Carlos Tovar. ](https://www.linkedin.com/in/juan-carlos-tovar-galarreta/) \
    Github repo [aquí](https://github.com/PUCP-Clases/DeployMLLib)')


#########   DATA
csv_file_path = os.path.join(current_path, "airbnb.csv")
#df = sc.read.csv("file:///home/hduser/programs/airbnb-price-pred/airbnb.csv", header=True)
df = sc.read.csv(csv_file_path, header=True)
num_rows = df.count()
num_cols = len(df.columns)
#Preprocessed data is given as input to save computation
#df4 = sc.read.load("file:///home/hduser/programs/airbnb-price-pred/processed_data.parquet")
parquet_file_path = os.path.join(current_path, "processed_data.parquet")
df4 = sc.read.load(parquet_file_path)
num_rows_p = df4.count()
num_cols_p = len(df4.columns)
splits = df4.randomSplit([0.8, 0.2], seed=12345)
train_df = splits[0]
test_df = splits[1]

##########  SIDEBAR

st.sidebar.title('Modelos de MLlib Regression')
st.sidebar.subheader('Selecciona tu modelo')
mllib_model = st.sidebar.selectbox("Modelos de Regresión", \
        ('Linear Regression', 'Gradient Boost Tree', 'Decision Tree Regressor', \
            'Random Forest Regressor'))
st.sidebar.text('80 - 20 split')

def regression_model(mllib_model, train_df, test_df):
    if mllib_model == 'Linear Regression':
        lr = LinearRegression(featuresCol = 'features', labelCol='label')   
        lr_model = lr.fit(train_df)
        fullPredictions = lr_model.transform(test_df).cache()
        lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="r2")
        r2 = lr_evaluator.evaluate(fullPredictions)
        lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="rmse")
        rmse = lr_evaluator.evaluate(fullPredictions)
        pred = [int(row['prediction']) for row in fullPredictions.select('prediction').collect()]
        actual = [int(row['label']) for row in fullPredictions.select('label').collect()]
        return r2,rmse,pred,actual

    elif mllib_model == 'Decision Tree Regressor':
        dt = DecisionTreeRegressor(featuresCol = 'features', labelCol='label')
        dt_model = dt.fit(train_df)
        dtPrediction = dt_model.transform(test_df).cache()
        dt_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="r2")
        r2 = dt_evaluator.evaluate(dtPrediction)
        dt_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="rmse")
        rmse = dt_evaluator.evaluate(dtPrediction)
        pred = [int(row['prediction']) for row in dtPrediction.select('prediction').collect()]
        actual = [int(row['label']) for row in dtPrediction.select('label').collect()]
        return r2,rmse,pred,actual

    elif mllib_model == 'Gradient Boost Tree':
        gb = GBTRegressor(featuresCol = 'features', labelCol='label')
        gb_model = gb.fit(train_df)
        gbPredictions = gb_model.transform(test_df).cache()
        gb_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="r2")
        r2 = gb_evaluator.evaluate(gbPredictions)
        gb_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="rmse")
        rmse = gb_evaluator.evaluate(gbPredictions)
        pred = [int(row['prediction']) for row in gbPredictions.select('prediction').collect()]
        actual = [int(row['label']) for row in gbPredictions.select('label').collect()]
        return r2,rmse,pred,actual   

    else: #mllib_model == 'Random Forest Regressor':
        rf = RandomForestRegressor(featuresCol = 'features', labelCol='label')
        rf_model = rf.fit(train_df)
        rfPredictions = rf_model.transform(test_df).cache()
        rf_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="r2")
        r2 = rf_evaluator.evaluate(rfPredictions)
        rf_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="rmse")
        rmse = rf_evaluator.evaluate(rfPredictions)
        pred = [int(row['prediction']) for row in rfPredictions.select('prediction').collect()]
        actual = [int(row['label']) for row in rfPredictions.select('label').collect()]
        return r2,rmse,pred,actual

    """else:
        fm = FMRegressor(featuresCol = 'features', labelCol='label')
        fm_model = fm.fit(train_df)
        fmPredictions = fm_model.transform(test_df).cache()
        fm_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="r2")
        r2 = fm_evaluator.evaluate(fmPredictions)
        fm_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label",metricName="rmse")
        rmse = fm_evaluator.evaluate(fmPredictions)
        pred = [int(row['prediction']) for row in fmPredictions.select('prediction').collect()]
        actual = [int(row['label']) for row in fmPredictions.select('label').collect()]
        return r2,rmse,pred,actual"""

def safe_to_pandas(spark_df, num_rows=10):
    # Get the data as a list of rows
    rows = spark_df.limit(num_rows).collect()
    
    # Convert to list of dictionaries
    data = [row.asDict() for row in rows]
    
    # Convert to pandas
    import pandas as pd
    return pd.DataFrame(data)

df_pd = safe_to_pandas(df, num_rows=10)
st.dataframe(df_pd)
st.text("Dataframe shape: (" + str(num_rows) + "," + str(num_cols) + ")")
st.text("Dataframe procesado shape: (" + str(num_rows_p) + "," + str(num_cols_p) + ")")
st.text('Nuestra variable objetivo es el precio y estamos dando datos vectorizados a Apache Spark MLlib')
st.text('A continuación se muestran los resultados de los datos de las pruebas.')
r2,rmse,actual,pred = regression_model(mllib_model, train_df, test_df)

st.write("Modelo: ", mllib_model)


col3, col4, col5 = st.columns((1,1,2))
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-Weight: bold;
}
</style>
""", unsafe_allow_html=True)

col3.header("R2 score")
col3.markdown(f'<p class="big-font">{"{:.2f}".format(r2)}</p>', unsafe_allow_html=True)

#
col4.header("RMS Error")
col4.markdown(f'<p class="big-font">{"{:.2f}".format(rmse)}</p>', unsafe_allow_html=True)

#
fig, ax = plt.subplots()
ax.scatter(actual, pred, color='b', s=60, alpha=0.1)
plt.plot([5,250], [5,250], color='r')
plt.xlim([0, 260])
plt.ylim([0, 260])
ax.set_xlabel('Actual')
ax.set_ylabel('Predicción')
ax.set_title('Actual vs Predicción',fontsize=20)
col5.pyplot(fig)

st.text('Todos los modelos tienen un rendimiento medio y el modelo "Gradient Boost Tree Regression" dio los mejores resultados entre ellos.')

