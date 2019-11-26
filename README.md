# oplx-py-flask-sales
Basic implementation of web service (using **flask**) to consume ML model (sales prediction).

## Execution
Command line (local): `flask run` (will execute app.py and locally activate web service at http://127.0.0.1:5000/api)  
Heroku: configuration set in `Procfile`. Web service can be tested pointing to web service using [https://oplx-py-flask-sales.herokuapp.com/api](https://oplx-py-flask-sales.herokuapp.com/api).

## Files information
- `sales_build_and_train.py` contains a simple generation of ML model (using sci-kit learn pipeline and pickle serialization).
- `app.py` is a basic example of web service for ML model consumption (use of **flask** functionality to receive data selection parameters and return dataframe information, both using JSON via POST request).
- `requirements.txt`, `runtime.txt` and `Procfile` are used during web service execution (Heroku).